# Measure Detector v2

CPU-only measure detector service compatible with the original `measure-detector` API.

This package ships the new YOLO detector as ONNX and serves it with ONNX Runtime, so the runtime image does not include PyTorch, CUDA, or Ultralytics.

## API

- `GET /health`
- `POST /json`
- `POST /mei`
- `POST /debug`

`/json`, `/mei`, and `/debug` accept the same `expand`, `trim`, `auto`, and `pretty` form options as the original service. Bounding boxes are normalized to the original image size.

```shell
curl -s \
  -F 'files=@/path/to/page.jpg' \
  -F 'auto=y' \
  -F 'pretty=y' \
  http://localhost:8123/json
```

## Docker

```shell
docker build -t measure-detector-v2 .
docker run --rm -p 8123:8123 -e WORKERS=1 measure-detector-v2
```

Threading can be tuned with:

- `ORT_INTRA_OP_NUM_THREADS`, default `0` lets ONNX Runtime choose.
- `ORT_INTER_OP_NUM_THREADS`, default `1`.

## Model Optimization

The runtime model is `models/model.optimized.onnx`. To regenerate it from the local YOLO checkpoint:

```shell
uv pip install -e '.[export,optimize]'
python -m measure_detector_v2.optimize_onnx \
  --pt /home/simon/repos/measure-alignment-thesis/runs/detector/yolo26n-640-60768424/weights/best.pt \
  --out models/model.onnx \
  --optimized-out models/model.optimized.onnx \
  --imgsz 640
```

The script runs Ultralytics ONNX export with simplification/slimming where available, validates the graph with `onnx.checker`, optionally runs `onnxsim`, optionally runs `onnxslim`, and writes an ONNX Runtime graph-optimized model.

## Benchmark

```shell
measure-detector-v2-bench /path/to/page.jpg --runs 50 --warmup 5
```
