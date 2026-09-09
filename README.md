# Measure Detector v2

CPU-only measure detector service compatible with the original `measure-detector` API.

This package ships the new YOLO detector as ONNX and serves it with ONNX Runtime, so the runtime image does not include PyTorch, CUDA, or Ultralytics.

## API

- `GET /health`
- `POST /json`
- `POST /mei`
- `POST /debug`
- `POST /xfdf`

`/json`, `/mei`, and `/debug` accept the same `expand`, `trim`, `auto`, and `pretty` form options as the original service. Bounding boxes are normalized to the original image size, or to the rendered page size for PDFs.

```shell
curl -s \
  -F 'files=@/path/to/page.jpg' \
  -F 'auto=y' \
  -F 'pretty=y' \
  http://localhost:8123/json
```

PDF uploads are supported by all four endpoints. `/json` returns one result per
PDF page in a single response, keeping the original `filename` and adding a
one-based `page` field. Image results keep their existing schema. Pages are
processed in document order; `/json` preserves upload order and `/mei` naturally
sorts uploaded filenames as before.

```shell
curl -s -F 'files=@/path/to/score.pdf' -F 'pretty=y' http://localhost:8123/json
curl -s -F 'file=@/path/to/score.pdf' -F 'page=2' http://localhost:8123/debug -o page-2.jpg
```

`/mei` creates a surface for each page, with graphic targets such as
`score.pdf#page=2`. Surface dimensions and zone coordinates use the rendered
pixels. `/debug` returns a JPEG of the selected page (`page=1` by default).
Invalid PDFs, PDFs requiring a password, and out-of-range pages return HTTP 400;
non-positive debug page numbers return HTTP 422. Missing Poppler returns HTTP 503.

## XFDF annotations

The Python `/xfdf` endpoint and Rust `--format xfdf` option export **one XFDF
file for one PDF**, including annotations from every page. Pass exactly one PDF;
image inputs and multiple input documents are rejected because the XFDF `f`
element refers to a single target PDF.

```shell
curl -s -F 'files=@/path/to/score.pdf' -F 'auto=y' -F 'pretty=y' \
  http://localhost:8123/xfdf -o measures.xfdf

./target/release/measure-detector-v2 --format xfdf --auto --pretty \
  /path/to/score.pdf -o measures.xfdf
```

Import the XFDF into its source PDF with a viewer that supports XFDF annotations.
The server returns `application/vnd.adobe.xfdf`; `expand`, `trim`, `auto`, and
`pretty` work as for JSON and MEI. Invalid inputs return HTTP 400.

The output follows `examples/KA Jurgenson.xfdf`: one transparent, centered
`freetext` annotation per detected measure, titled `BarNumber`, with names
`bar-0`, `bar-1`, etc. Measure numbers start at 1 and continue across pages. These
are detection-order labels, not recognition of printed bar numbers, movement
restarts, or repeats. Rich text, default style, and default appearance use Courier
New with a font size fitted to the box, capped at 150 points, and opacity 0.1.

XFDF page indices start at **0** (JSON's `page` starts at 1). Rectangles use PDF
user coordinates with the origin at the bottom left, accounting for MediaBox
offsets and page rotation. The conversion matches the MediaBox used by the PDF
renderer, including PDFs with a different CropBox; coordinates do not depend on
rounded raster dimensions. The source PDF reference uses the supplied filename
(API) or input path (CLI). Keep it resolvable if you move the XFDF. Document IDs,
manual reminders, and repeat annotations from the example are not copied.

## PDF rendering dependency

Both implementations render PDFs at **150 dpi** using Poppler's `pdfinfo` and
`pdftoppm`, which must be available on `PATH`. The Docker image includes them.
For local Python and Rust use, install them with:

```shell
# macOS
brew install poppler
# Debian / Ubuntu
sudo apt-get install poppler-utils
```

Normal detection renders one page at a time and removes temporary files when
processing completes or fails. The Rust steady-state benchmark preloads all
pages, just as it preloads image inputs; PDF rendering is excluded from benchmark
timings.

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

## Standalone Rust CLI

The Rust CLI embeds both an ONNX model and the matching ONNX Runtime CPU shared
library into one executable **at compile time**. Selection follows the Rust
compilation target (including cross-compilation), not the machine running the
build:

| Target | Embedded runtime | Embedded model |
| --- | --- | --- |
| Linux x86_64 | `libonnxruntime.so.1.27.0` | `model.optimized.onnx` |
| macOS ARM64 (Apple Silicon) | `libonnxruntime.1.27.0.dylib` | `model.onnx` |

Other targets fail at compile time until a matching runtime asset is added.
The macOS model is optimized by ONNX Runtime during session creation; the saved
x86 model contains NCHWc operators unavailable on ARM. At startup, the executable
atomically extracts its embedded library to a cache separated by platform and
loads its embedded model from memory. It does not download a runtime and does not
need Python installed. Native loading is checked before calling `ort`, so a
library-loading failure is reported instead of hanging in `ort`'s error handling.

```shell
cd rust/measure-detector-cli
cargo build --release
./target/release/measure-detector-v2 --pretty /path/to/page.jpg
./target/release/measure-detector-v2 --format mei --pretty /path/to/pages -o measures.mei
```

Inputs can be individual image files, PDFs, folders, or a mix of these. Folders are scanned recursively unless `--no-recursive` is set. Supported extensions are `jpg`, `jpeg`, `png`, `tif`, `tiff`, `webp`, and `pdf` (case-insensitive).

A CLI invocation writes **one JSON document**, even for multiple PDFs or a
multipage PDF. Each PDF page is an entry in `results`, with its original
`filename` and one-based `page`. `-o` writes the complete document to one file:

```shell
./target/release/measure-detector-v2 --pretty /path/to/score.pdf -o results.json
```

During startup, the CLI immediately shows the active stage and elapsed time:
input scanning, PDF geometry for XFDF, loading ONNX Runtime and the detector
model, and reading input metadata. These stages run before page processing and
have no meaningful page-based ETA yet.

For multipage PDFs, the CLI then shows a progress bar on terminal stderr with the
filename, current page, completed/total pages, remaining pages, elapsed time, and
estimated remaining time. The estimate becomes available after the first page
and includes rendering plus inference. Each PDF gets its own bar. Use
`--no-progress` to hide it; redirected stderr disables it automatically. Result
data on stdout and in `-o` files is unaffected. In benchmark mode, the bar covers
PDF preloading only and finishes before the timed inference runs.

The model and ONNX Runtime remain embedded; PDF input additionally requires the
Poppler programs described above. Both supported targets can run PDF rendering,
inference, and output locally.

The embedded model is used by default. For development comparisons, pass `--model /path/to/model.onnx` to override it.

The Rust detector groups boxes into systems by vertical overlap, orders systems
from top to bottom, and measures within each system from left to right. This
avoids a non-transitive pairwise comparison that could panic on complex pages.

## Tests

With Poppler installed:

```shell
uv run --with httpx python -m unittest discover -s tests -v
cargo test --manifest-path rust/measure-detector-cli/Cargo.toml
# POSIX CLI tests for blocked-startup progress and embedded-runtime inference:
cargo build --manifest-path rust/measure-detector-cli/Cargo.toml
python3 rust/measure-detector-cli/tests/progress_startup.py
```

The shared two-page PDF fixture covers page order, RGB rendering, different page
sizes, combined JSON output, MEI coordinates, debug selection, and invalid inputs.
XFDF tests also cover XML structure, target escaping, continuous numbering, and
conversion back to PDF coordinates on rotated pages with offset page boxes.
