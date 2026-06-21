from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnx
import onnxruntime as ort


def export_from_pt(pt_path: Path, out_path: Path, imgsz: int) -> None:
    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,
            simplify=True,
            opset=12,
            nms=False,
            device="cpu",
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != out_path.resolve():
        shutil.copy2(exported, out_path)


def maybe_onnxsim(path: Path) -> None:
    try:
        from onnxsim import simplify
    except Exception:
        return

    model = onnx.load(path)
    simplified, ok = simplify(model)
    if not ok:
        raise RuntimeError("onnxsim validation failed")
    onnx.save(simplified, path)


def maybe_onnxslim(path: Path) -> None:
    try:
        import onnxslim
    except Exception:
        return

    model = onnx.load(path)
    slimmed = onnxslim.slim(model)
    onnx.save(slimmed, path)


def check_model(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)


def write_ort_optimized(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(src), sess_options=options, providers=["CPUExecutionProvider"])
    check_model(dst)


def optimize(src: Path, dst: Path | None = None) -> Path:
    target = dst or src
    if src.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    maybe_onnxsim(target)
    maybe_onnxslim(target)
    check_model(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Export, simplify, validate, and ORT-optimize the detector ONNX model.")
    parser.add_argument("--pt", type=Path, default=None, help="Optional PyTorch YOLO checkpoint to export first.")
    parser.add_argument("--in", dest="input", type=Path, default=None, help="Existing ONNX model to optimize.")
    parser.add_argument("--out", type=Path, default=Path("models/model.onnx"))
    parser.add_argument("--optimized-out", type=Path, default=Path("models/model.optimized.onnx"))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if args.pt is None and args.input is None:
        parser.error("provide --pt or --in")

    if args.pt is not None:
        export_from_pt(args.pt, args.out, args.imgsz)
        src = args.out
    else:
        src = args.input

    assert src is not None
    model_path = optimize(src, args.out)
    write_ort_optimized(model_path, args.optimized_out)
    print(f"wrote {model_path}")
    print(f"wrote {args.optimized_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
