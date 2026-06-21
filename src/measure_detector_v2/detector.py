from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from measure_detector_v2.postprocess import (
    BBox,
    Measure,
    detect_page_type,
    draw_debug_image,
    remove_overlapping_measures,
    sort_measures,
    unify_measures,
)

CLASS_NAMES = ["handwritten", "typeset"]


class MeasureDetector:
    def __init__(
        self,
        model_path: str | Path,
        *,
        conf: float = 0.25,
        imgsz: int = 640,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf = conf
        self.imgsz = imgsz

        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        intra_threads = int(os.getenv("ORT_INTRA_OP_NUM_THREADS", "0"))
        inter_threads = int(os.getenv("ORT_INTER_OP_NUM_THREADS", "1"))
        if intra_threads > 0:
            options.intra_op_num_threads = intra_threads
        if inter_threads > 0:
            options.inter_op_num_threads = inter_threads

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        if len(shape) == 4 and isinstance(shape[2], int):
            self.imgsz = int(shape[2])

    @staticmethod
    def decode_image(data: bytes) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data uploaded")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _letterbox(self, rgb: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        h, w = rgb.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        new_w = int(round(w * r))
        new_h = int(round(h * r))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_x = (self.imgsz - new_w) / 2
        pad_y = (self.imgsz - new_h) / 2
        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas, r, (left, top)

    def _prepare(self, rgb: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        lb, ratio, pad = self._letterbox(rgb)
        arr = lb.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None]
        return np.ascontiguousarray(arr), ratio, pad

    def predict_image(
        self,
        rgb: np.ndarray,
        *,
        expand: bool = False,
        trim: bool = False,
        auto: bool = False,
    ) -> tuple[list[Measure], str, float]:
        h, w = rgb.shape[:2]
        tensor, ratio, (pad_x, pad_y) = self._prepare(rgb)
        pred = self.session.run(None, {self.input_name: tensor})[0]
        pred = np.asarray(pred)
        if pred.ndim == 3:
            pred = pred[0]

        measures: list[Measure] = []
        for row in pred:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, score, class_id = row[:6]
            score = float(score)
            if score < self.conf:
                continue
            cls = int(class_id)
            x1 = (float(x1) - pad_x) / ratio
            x2 = (float(x2) - pad_x) / ratio
            y1 = (float(y1) - pad_y) / ratio
            y2 = (float(y2) - pad_y) / ratio
            x1 = min(max(x1 / w, 0.0), 1.0)
            x2 = min(max(x2 / w, 0.0), 1.0)
            y1 = min(max(y1 / h, 0.0), 1.0)
            y2 = min(max(y2 / h, 0.0), 1.0)
            if x2 <= x1 or y2 <= y1:
                continue
            class_name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
            measures.append(
                Measure(
                    class_id=cls,
                    class_name=class_name,
                    confidence=round(score, 3),
                    bbox=BBox(
                        round(x1, 5),
                        round(y1, 5),
                        round(x2, 5),
                        round(y2, 5),
                    ),
                )
            )

        measures = sort_measures(measures)
        page_type, type_conf = detect_page_type(measures)
        measures = remove_overlapping_measures(measures)
        measures = unify_measures(measures, page_type, expand=expand, trim=trim, auto=auto)
        return measures, page_type, type_conf

    def predict_bytes(
        self,
        data: bytes,
        *,
        expand: bool = False,
        trim: bool = False,
        auto: bool = False,
        debug: bool = False,
    ):
        rgb = self.decode_image(data)
        measures, page_type, type_conf = self.predict_image(rgb, expand=expand, trim=trim, auto=auto)
        if debug:
            return draw_debug_image(rgb, measures), page_type, type_conf
        return measures, page_type, type_conf


def default_model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", "models/model.optimized.onnx"))
