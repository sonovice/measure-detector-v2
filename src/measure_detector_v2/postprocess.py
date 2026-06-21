from __future__ import annotations

import io
import statistics
from dataclasses import astuple, dataclass
from functools import cmp_to_key

from PIL import Image, ImageDraw, ImageFont

HANDWRITTEN = 0
TYPESET = 1


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class Measure:
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox


def cmp_measure_bboxes(a: Measure, b: Measure) -> int:
    a, b = a.bbox, b.bbox
    if a.x1 >= b.x1 and a.y1 >= b.y1:
        return +1
    if a.x1 < b.x1 and a.y1 < b.y1:
        return -1

    denom = min(a.y2 - a.y1, b.y2 - b.y1)
    overlap_y = min(a.y2 - b.y1, b.y2 - a.y1) / denom if denom else 0.0
    if overlap_y >= 0.5:
        return -1 if a.x1 < b.x1 else +1
    return +1 if a.x1 < b.x1 else -1


def get_geometry(a: BBox, b: BBox) -> tuple[float, float, float]:
    left = max(a.x1, b.x1)
    top = max(a.y1, b.y1)
    right = min(a.x2, b.x2)
    bottom = min(a.y2, b.y2)

    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    intersection_area = 0.0 if right < left or bottom < top else (right - left) * (bottom - top)
    return intersection_area, area_a, area_b


def remove_overlapping_measures(measures: list[Measure], thresh: float = 0.7) -> list[Measure]:
    valid = [True] * len(measures)
    for a, measure_a in enumerate(measures):
        for b, measure_b in enumerate(measures):
            if a == b:
                continue
            intersection_area, area_a, area_b = get_geometry(measure_a.bbox, measure_b.bbox)
            if intersection_area == 0:
                continue
            ioa_a = intersection_area / area_a if area_a else 0.0
            ioa_b = intersection_area / area_b if area_b else 0.0
            iou = intersection_area / (area_a + area_b - intersection_area)
            if ioa_a > thresh or ioa_b > thresh or iou > thresh:
                if measure_a.confidence > measure_b.confidence:
                    valid[b] = False
                else:
                    valid[a] = False
    return [m for m, ok in zip(measures, valid, strict=True) if ok]


def detect_page_type(measures: list[Measure]) -> tuple[str, float]:
    if not measures:
        return "unknown", 0.0

    scores = [0.0, 0.0]
    for measure in measures:
        x1, y1, x2, y2 = measure.bbox
        area = (x2 - x1) * (y2 - y1)
        if 0 <= measure.class_id < len(scores):
            scores[measure.class_id] += area * measure.confidence

    total = sum(scores)
    if total == 0:
        return "unknown", 0.0
    scores = [score / total for score in scores]
    if scores[HANDWRITTEN] > scores[TYPESET]:
        return "handwritten", scores[HANDWRITTEN]
    return "typeset", scores[TYPESET]


def unify_measures(
    measures: list[Measure],
    page_type: str,
    expand: bool = False,
    trim: bool = False,
    auto: bool = False,
) -> list[Measure]:
    if not any([expand, trim, auto]):
        return measures

    if auto:
        expand = page_type == "typeset"
        trim = page_type == "typeset"

    system_tops: list[float] = []
    system_bottoms: list[float] = []
    cur_bbox = BBox(0.0, 0.0, 1.0, 1.0)
    cur_system_top = 1.0
    cur_system_bottom = 0.0
    measure_num_to_system_idx: list[int] = []

    for measure in measures:
        bbox = measure.bbox
        denom = min(bbox.y2 - bbox.y1, cur_bbox.y2 - cur_bbox.y1)
        overlap_y = min(bbox.y2 - cur_bbox.y1, cur_bbox.y2 - bbox.y1) / denom if denom else 0.0
        if bbox.x1 > cur_bbox.x1 and overlap_y > 0.5:
            cur_system_top = min(cur_system_top, bbox.y1)
            cur_system_bottom = max(cur_system_bottom, bbox.y2)
        else:
            system_tops.append(cur_system_top)
            system_bottoms.append(cur_system_bottom)
            cur_system_top = 1.0
            cur_system_bottom = 0.0
        cur_bbox = bbox
        measure_num_to_system_idx.append(len(system_tops))

    system_tops.append(cur_system_top)
    system_bottoms.append(cur_system_bottom)

    if expand:
        for i, measure in enumerate(measures):
            x1, _, x2, _ = measure.bbox
            measure.bbox = BBox(x1, system_tops[measure_num_to_system_idx[i]], x2, system_bottoms[measure_num_to_system_idx[i]])

    if trim and len(measures) >= 2:
        c_vals: list[float] = []
        for i in range(len(measures) - 1):
            if measure_num_to_system_idx[i] == measure_num_to_system_idx[i + 1]:
                ax2 = measures[i].bbox.x2
                bx1 = measures[i + 1].bbox.x1
                if ax2 > bx1:
                    c = (ax2 - bx1) / 2
                    measures[i].bbox.x2 -= c
                    measures[i + 1].bbox.x1 += c
                    c_vals.append(c)

        if c_vals:
            c_mean = statistics.mean(c_vals)
            for i in range(1, len(measures) - 1):
                if measure_num_to_system_idx[i] != measure_num_to_system_idx[i + 1]:
                    measures[i].bbox.x2 -= c_mean
                    measures[i + 1].bbox.x1 += c_mean
            measures[0].bbox.x1 += c_mean
            measures[-1].bbox.x2 -= c_mean

    return measures


def sort_measures(measures: list[Measure]) -> list[Measure]:
    return sorted(measures, key=cmp_to_key(cmp_measure_bboxes))


def draw_debug_image(rgb_img, measures: list[Measure]) -> bytes:
    pil = Image.fromarray(rgb_img).convert("RGBA")
    h, w = rgb_img.shape[:2]
    colors = ["#aa75ff", "#1ec7c7"]
    font = ImageFont.load_default()

    for i, measure in enumerate(measures):
        overlay = Image.new("RGBA", pil.size)
        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = measure.bbox
        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h
        color = colors[measure.class_id] if 0 <= measure.class_id < len(colors) else "#ffcc00"
        draw.rectangle((x1, y1, x2, y2), fill=f"{color}40", outline=color, width=2)

        text = f"{i + 1} ({int(measure.confidence * 100)}%)"
        try:
            tbx1, tby1, tbx2, tby2 = draw.textbbox((x1, y1), text, font=font)
            tw, th = tbx2 - tbx1, tby2 - tby1
        except Exception:
            tw, th = (len(text) * 6, 10)
        draw.rectangle((x1, y1, x1 + tw + 1, y1 + th + 1), outline=color, fill=color, width=2)
        draw.text((x1 + 2, y1 + 1), text, fill="#ffffff", font=font)
        pil = Image.alpha_composite(pil, overlay)

    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="jpeg")
    return buf.getvalue()
