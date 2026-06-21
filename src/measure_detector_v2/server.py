from __future__ import annotations

import json
from dataclasses import asdict
from time import perf_counter
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from measure_detector_v2 import __version__
from measure_detector_v2.detector import MeasureDetector, default_model_path

app = FastAPI(
    title="Measure Detector v2",
    description="CPU-only ONNX Runtime Measure Detector Microservice",
    version=__version__,
    contact={"name": "Simon Waloschek", "email": "waloschek@pm.me"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_true(val: str | None) -> bool:
    return val in {"1", "t", "T", "true", "True", "TRUE", "y", "yes", "Yes", "YES"}


@app.on_event("startup")
async def load_model() -> None:
    app.state.detector = MeasureDetector(default_model_path())


@app.get("/health")
async def get_health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": app.version})


@app.post("/json")
async def post_json(
    files: list[UploadFile] = File(...),
    expand: Optional[str] = Form(None),
    trim: Optional[str] = Form(None),
    auto: Optional[str] = Form(None),
    pretty: Optional[str] = Form(None),
) -> Response:
    start = perf_counter()
    detector: MeasureDetector = app.state.detector

    results = []
    for uf in files:
        try:
            data = await uf.read()
            measures, page_type, type_conf = detector.predict_bytes(
                data,
                expand=is_true(expand),
                trim=is_true(trim),
                auto=is_true(auto),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        results.append(
            {
                "filename": getattr(uf, "filename", None),
                "type": page_type,
                "type_confidence": round(type_conf, 3),
                "measures": [asdict(measure) for measure in measures],
            }
        )

    payload = {"process_time": round((perf_counter() - start) * 1000), "results": results}
    if is_true(pretty):
        return Response(
            content=json.dumps(payload, indent=2, ensure_ascii=False),
            media_type="application/json",
        )
    return JSONResponse(payload)


@app.post("/mei")
async def post_mei(
    files: list[UploadFile] = File(...),
    expand: Optional[str] = Form(None),
    trim: Optional[str] = Form(None),
    auto: Optional[str] = Form(None),
    pretty: Optional[str] = Form(None),
) -> Response:
    import re
    import xml.etree.ElementTree as ET

    start = perf_counter()
    detector: MeasureDetector = app.state.detector

    mei = ET.Element("mei", attrib={"xmlns": "http://www.music-encoding.org/ns/mei"})
    music = ET.SubElement(mei, "music")
    body = ET.SubElement(music, "body")
    mdiv = ET.SubElement(body, "mdiv")
    score = ET.SubElement(mdiv, "score")
    section = ET.SubElement(score, "section")
    facsimile = ET.SubElement(mei, "facsimile")

    def natural_key(name: str):
        parts = re.split(r"(\d+)", name)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    page_counter = 0
    measure_global_idx = 1

    for uf in sorted(files, key=lambda f: natural_key(getattr(f, "filename", ""))):
        try:
            data = await uf.read()
            rgb = detector.decode_image(data)
            height, width = rgb.shape[:2]
            measures, _, _ = detector.predict_image(
                rgb,
                expand=is_true(expand),
                trim=is_true(trim),
                auto=is_true(auto),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        page_counter += 1
        surface = ET.SubElement(
            facsimile,
            "surface",
            attrib={
                "xml:id": f"surface_{page_counter}",
                "n": str(page_counter),
                "ulx": "0",
                "uly": "0",
                "lrx": str(width - 1),
                "lry": str(height - 1),
            },
        )
        ET.SubElement(
            surface,
            "graphic",
            attrib={
                "xml:id": f"graphic_{page_counter}",
                "target": getattr(uf, "filename", "image"),
                "width": f"{width}px",
                "height": f"{height}px",
            },
        )

        for measure in measures:
            x1 = int(round(measure.bbox.x1 * width))
            y1 = int(round(measure.bbox.y1 * height))
            x2 = int(round(measure.bbox.x2 * width))
            y2 = int(round(measure.bbox.y2 * height))
            zone_id = f"zone_{measure_global_idx}"
            ET.SubElement(
                surface,
                "zone",
                attrib={
                    "xml:id": zone_id,
                    "type": "measure",
                    "ulx": str(x1),
                    "uly": str(y1),
                    "lrx": str(x2),
                    "lry": str(y2),
                },
            )
            ET.SubElement(
                section,
                "measure",
                attrib={
                    "xml:id": f"measure_{measure_global_idx}",
                    "n": str(measure_global_idx),
                    "label": str(measure_global_idx),
                    "facs": f"#{zone_id}",
                },
            )
            measure_global_idx += 1

    if is_true(pretty):
        try:
            ET.indent(mei, space="  ")
        except Exception:
            pass
    headers = {"X-Process-Time": str(round((perf_counter() - start) * 1000))}
    return Response(content=ET.tostring(mei, encoding="utf-8"), media_type="application/xml", headers=headers)


@app.post("/debug")
async def post_debug(
    file: UploadFile = File(...),
    expand: Optional[str] = Form(None),
    trim: Optional[str] = Form(None),
    auto: Optional[str] = Form(None),
) -> Response:
    start = perf_counter()
    detector: MeasureDetector = app.state.detector
    try:
        img, _, _ = detector.predict_bytes(
            await file.read(),
            expand=is_true(expand),
            trim=is_true(trim),
            auto=is_true(auto),
            debug=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    headers = {"X-Process-Time": str(round((perf_counter() - start) * 1000))}
    return Response(img, media_type="image/jpeg", headers=headers)


def main() -> None:
    import uvicorn

    uvicorn.run("measure_detector_v2.server:app", host="0.0.0.0", port=8123)
