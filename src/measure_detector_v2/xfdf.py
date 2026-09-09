"""XFDF free-text measure annotations in unrotated PDF user coordinates."""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from measure_detector_v2.inputs import _poppler
from measure_detector_v2.postprocess import BBox, Measure


@dataclass(frozen=True)
class PageGeometry:
    # pdftoppm renders the MediaBox by default, including its origin and /Rotate.
    media_box: tuple[float, float, float, float]
    rotation: int

    def rect(self, bbox: BBox) -> tuple[float, float, float, float]:
        left, bottom, right, top = self.media_box
        width, height = right - left, top - bottom

        def point(u: float, v: float) -> tuple[float, float]:
            x, y = {
                0: (u, 1 - v), 90: (v, u),
                180: (1 - u, v), 270: (1 - v, 1 - u),
            }[self.rotation]
            return left + x * width, bottom + y * height

        points = [point(x, y) for x in (bbox.x1, bbox.x2) for y in (bbox.y1, bbox.y2)]
        return (min(p[0] for p in points), min(p[1] for p in points),
                max(p[0] for p in points), max(p[1] for p in points))


def pdf_geometry(data: bytes) -> list[PageGeometry]:
    with TemporaryDirectory(prefix="measure-detector-xfdf-") as directory:
        source = Path(directory) / "input.pdf"
        source.write_bytes(data)
        info = _poppler("pdfinfo", "-box", "-f", "1", "-l", "2147483647", str(source))
    return parse_geometry(info.decode("utf-8", errors="replace"))


def parse_geometry(info: str) -> list[PageGeometry]:
    boxes = {}
    rotations = {}
    count = 0
    for line in info.splitlines():
        if line.startswith("Pages:"):
            count = int(line.split(":", 1)[1].strip())
        match = re.match(r"Page\s+(\d+)\s+(MediaBox|rot):\s+(.+)", line)
        if not match:
            continue
        page, field, value = match.groups()
        if field == "MediaBox":
            boxes[int(page)] = tuple(float(v) for v in value.split())
        else:
            rotations[int(page)] = int(value) % 360
    if count < 1:
        raise ValueError("PDF contains no pages")
    pages = []
    for number in range(1, count + 1):
        box, rotation = boxes.get(number, ()), rotations.get(number)
        if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                or box[2] <= box[0] or box[3] <= box[1] or rotation not in (0, 90, 180, 270)):
            raise ValueError(f"Invalid PDF geometry for page {number}")
        pages.append(PageGeometry(box, rotation))
    return pages


def write_xfdf(filename: str, pages: list[tuple[PageGeometry, list[Measure]]],
               pretty: bool = False) -> bytes:
    root = ET.Element("xfdf", {"xmlns": "http://ns.adobe.com/xfdf/", "xml:space": "preserve"})
    ET.SubElement(root, "f", {"href": filename})
    annots = ET.SubElement(root, "annots")
    number = 0
    for page, (geometry, measures) in enumerate(pages):
        for measure in measures:
            number += 1
            rect = geometry.rect(measure.bbox)
            width, height = rect[2] - rect[0], rect[3] - rect[1]
            if geometry.rotation in (90, 270):
                width, height = height, width
            font_size = min(150.0, height, width / (0.6 * len(str(number)) + 0.3))
            size = f"{font_size:.6f}"
            style = (f'font-family: "Courier New", monospace; text-align: center; '
                     f'font-weight: bold; font-size: {size}pt; color: #000000;')
            annot = ET.SubElement(annots, "freetext", {
                "name": f"bar-{number - 1}", "page": str(page), "subject": str(number),
                "rect": ",".join(f"{v:.6f}" for v in rect),
                "title": "BarNumber", "flags": "print,readonly,locked",
                "color": "#ffffff", "opacity": "0.1", "width": "0.0",
                "justification": "center", "rotation": str(geometry.rotation),
                "style": "cloudy", "intensity": "0.0",
            })
            ET.SubElement(annot, "contents").text = str(number)
            rich = ET.SubElement(annot, "contents-richtext")
            body = ET.SubElement(rich, "body", {
                "xmlns": "http://www.w3.org/1999/xhtml",
                "xmlns:xfa": "http://www.xfa.org/schema/xfa-data/1.0/",
                "xfa:spec": "2.0.2", "xfa:APIVersion": "Acrobat:11.0.0",
            })
            ET.SubElement(body, "p", {"style": style}).text = str(number)
            ET.SubElement(annot, "defaultstyle").text = style
            ET.SubElement(annot, "defaultappearance").text = f"0 0 0 rg /Courier#20New {size} Tf"
    if pretty:
        ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
