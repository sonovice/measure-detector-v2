# Embedded ONNX Runtime

- `libonnxruntime.so.1.27.0`: existing Linux x86_64 runtime.
- `libonnxruntime.1.27.0.dylib`: macOS ARM64 runtime, copied from the official
  `onnxruntime==1.27.0` Python wheel (PyPI), `onnxruntime/capi/`.
  SHA-256: `49422a138fff12d7fcbd544373aefbffd510e473e08e18af28a8ecc3581029bf`.

The runtime is selected using Rust target configuration and embedded with
`include_bytes!` at compile time. The executable extracts it to a target-specific
cache at startup; no runtime download or Python installation is needed.

The ONNX Runtime license and third-party notices are included alongside the
libraries. See `src/runtime.rs` for supported targets and model selection.
