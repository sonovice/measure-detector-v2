//! Embed a matching runtime and model for the compile target, not the build host.
use std::fs;

use anyhow::{Context, Result};

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod target {
    pub const LIBRARY: &[u8] = include_bytes!("../assets/libonnxruntime.so.1.27.0");
    pub const LIBRARY_NAME: &str = "libonnxruntime.so.1.27.0";
    pub const CACHE_TARGET: &str = "linux-x86_64";
    pub const MODEL: &[u8] = include_bytes!("../../../models/model.optimized.onnx");
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod target {
    pub const LIBRARY: &[u8] = include_bytes!("../assets/libonnxruntime.1.27.0.dylib");
    pub const LIBRARY_NAME: &str = "libonnxruntime.1.27.0.dylib";
    pub const CACHE_TARGET: &str = "macos-aarch64";
    // The optimized model contains x86-only NCHWc operators. Let ORT optimize
    // the portable ONNX model for this target when creating the session.
    pub const MODEL: &[u8] = include_bytes!("../../../models/model.onnx");
}

#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
compile_error!(
    "Embedded runtime supports Linux x86_64 and macOS ARM64 only; add a matching runtime asset for this target."
);

pub(crate) const EMBEDDED_MODEL: &[u8] = target::MODEL;

pub(crate) fn init() -> Result<()> {
    let directory = std::env::temp_dir()
        .join("measure-detector-v2-cli")
        .join("onnxruntime-1.27.0")
        .join(target::CACHE_TARGET);
    fs::create_dir_all(&directory)?;
    let path = directory.join(target::LIBRARY_NAME);
    if fs::metadata(&path).map_or(true, |meta| meta.len() != target::LIBRARY.len() as u64) {
        // Publish the entire library atomically, including concurrent first runs.
        let mut file = tempfile::NamedTempFile::new_in(&directory)?;
        std::io::Write::write_all(&mut file, target::LIBRARY)?;
        file.persist(&path)
            .context("failed to extract embedded ONNX Runtime")?;
    }

    // ort rc.12 can deadlock while constructing its error for dlopen failure.
    // Validate loading and the entry point first, using ordinary anyhow errors.
    // SAFETY: this is the bundled native ONNX Runtime for the compile target.
    // Keep this handle alive until ort has acquired its own reference.
    let library = unsafe { libloading::Library::new(&path) }
        .with_context(|| format!("failed to load embedded ONNX Runtime {}", path.display()))?;
    unsafe { library.get::<unsafe extern "C" fn() -> *const ()>(b"OrtGetApiBase") }
        .context("embedded ONNX Runtime has no OrtGetApiBase entry point")?;
    ort::init_from(&path)
        .map_err(|error| anyhow::anyhow!("failed to initialize embedded ONNX Runtime: {error}"))?
        .commit();
    Ok(())
}
