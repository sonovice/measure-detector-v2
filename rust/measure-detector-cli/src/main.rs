use std::cmp::Ordering;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use image::{imageops::FilterType, DynamicImage, ImageBuffer, Rgb, RgbImage};
use ort::session::Session;
use ort::value::{Tensor, ValueType};
use serde::Serialize;
use walkdir::WalkDir;

const CLASS_NAMES: [&str; 2] = ["handwritten", "typeset"];
const EMBEDDED_MODEL: &[u8] = include_bytes!("../../../models/model.optimized.onnx");
const EMBEDDED_ORT: &[u8] = include_bytes!("../assets/libonnxruntime.so.1.27.0");

#[derive(Parser, Debug)]
#[command(version, about = "CPU-only ONNX measure detector CLI")]
struct Args {
    /// Image files and/or folders. Folders are scanned recursively by default.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Optional ONNX model path. The optimized detector model is embedded by default.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,

    /// Write output to this file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Confidence threshold.
    #[arg(long, default_value_t = 0.25)]
    conf: f32,

    /// Expand measure boxes to a common system height.
    #[arg(long)]
    expand: bool,

    /// Trim overlapping adjacent measure boxes.
    #[arg(long)]
    trim: bool,

    /// Use trim+expand automatically for detected typeset pages.
    #[arg(long)]
    auto: bool,

    /// Pretty-print JSON or MEI.
    #[arg(long)]
    pretty: bool,

    /// Do not recurse into folders.
    #[arg(long)]
    no_recursive: bool,

    /// Run a steady-state benchmark instead of writing detections.
    #[arg(long)]
    bench_runs: Option<usize>,

    /// Warmup iterations per image for benchmark mode.
    #[arg(long, default_value_t = 5)]
    warmup: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Json,
    Mei,
}

#[derive(Clone, Debug, Serialize)]
struct BBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[derive(Clone, Debug, Serialize)]
struct Measure {
    class_id: usize,
    class_name: String,
    confidence: f32,
    bbox: BBox,
}

#[derive(Debug, Serialize)]
struct ImageResult {
    filename: String,
    #[serde(rename = "type")]
    page_type: String,
    type_confidence: f32,
    measures: Vec<Measure>,
}

#[derive(Debug, Serialize)]
struct JsonResponse {
    process_time: u128,
    results: Vec<ImageResult>,
}

struct LoadedImage {
    rgb: RgbImage,
}

struct Detector {
    session: Session,
    input_name: String,
    imgsz: u32,
    conf: f32,
}

impl Detector {
    fn new(model: Option<&Path>, conf: f32) -> Result<Self> {
        init_embedded_ort()?;

        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("failed to create ONNX Runtime session builder: {e}"))?;
        let session = if let Some(model) = model {
            if !model.exists() {
                bail!("model not found: {}", model.display());
            }
            builder
                .commit_from_file(model)
                .map_err(|e| anyhow::anyhow!("failed to load ONNX model {}: {e}", model.display()))?
        } else {
            builder
                .commit_from_memory(EMBEDDED_MODEL)
                .map_err(|e| anyhow::anyhow!("failed to load embedded ONNX model: {e}"))?
        };
        let input = session.inputs().first().context("model has no inputs")?;
        let input_name = input.name().to_string();
        let imgsz = match input.dtype() {
            ValueType::Tensor { shape, .. } => shape.get(2).copied().unwrap_or(640) as u32,
            _ => 640,
        };

        Ok(Self { session, input_name, imgsz, conf })
    }

    fn predict(&mut self, image: &RgbImage, expand: bool, trim: bool, auto: bool) -> Result<(Vec<Measure>, String, f32)> {
        let (tensor, ratio, pad_x, pad_y) = self.prepare(image);
        let input = Tensor::from_array((
            [1usize, 3, self.imgsz as usize, self.imgsz as usize],
            tensor.into_boxed_slice(),
        ))
        .map_err(|e| anyhow::anyhow!("failed to create input tensor: {e}"))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
            .map_err(|e| anyhow::anyhow!("ONNX Runtime inference failed: {e}"))?;
        let (_shape, output) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("failed to extract output tensor: {e}"))?;

        let (orig_w, orig_h) = image.dimensions();
        let mut measures = Vec::new();
        for row in output.chunks_exact(6) {
            let score = row[4];
            if score < self.conf {
                continue;
            }
            let class_id = row[5].round().max(0.0) as usize;
            let mut x1 = (row[0] - pad_x) / ratio;
            let mut y1 = (row[1] - pad_y) / ratio;
            let mut x2 = (row[2] - pad_x) / ratio;
            let mut y2 = (row[3] - pad_y) / ratio;
            x1 = (x1 / orig_w as f32).clamp(0.0, 1.0);
            x2 = (x2 / orig_w as f32).clamp(0.0, 1.0);
            y1 = (y1 / orig_h as f32).clamp(0.0, 1.0);
            y2 = (y2 / orig_h as f32).clamp(0.0, 1.0);
            if x2 <= x1 || y2 <= y1 {
                continue;
            }
            measures.push(Measure {
                class_id,
                class_name: CLASS_NAMES.get(class_id).unwrap_or(&"unknown").to_string(),
                confidence: round3(score),
                bbox: BBox { x1: round5(x1), y1: round5(y1), x2: round5(x2), y2: round5(y2) },
            });
        }

        measures.sort_by(cmp_measure_bboxes);
        let (page_type, type_conf) = detect_page_type(&measures);
        let mut measures = remove_overlapping_measures(&measures, 0.7);
        unify_measures(&mut measures, &page_type, expand, trim, auto);
        Ok((measures, page_type, type_conf))
    }

    fn prepare(&self, image: &RgbImage) -> (Vec<f32>, f32, f32, f32) {
        let (w, h) = image.dimensions();
        let ratio = (self.imgsz as f32 / w as f32).min(self.imgsz as f32 / h as f32);
        let new_w = (w as f32 * ratio).round() as u32;
        let new_h = (h as f32 * ratio).round() as u32;
        let resized = DynamicImage::ImageRgb8(image.clone())
            .resize_exact(new_w, new_h, FilterType::Triangle)
            .to_rgb8();
        let pad_x = ((self.imgsz - new_w) as f32 / 2.0 - 0.1).round().max(0.0);
        let pad_y = ((self.imgsz - new_h) as f32 / 2.0 - 0.1).round().max(0.0);
        let left = pad_x as u32;
        let top = pad_y as u32;

        let canvas = ImageBuffer::from_pixel(self.imgsz, self.imgsz, Rgb([114, 114, 114]));
        let side = self.imgsz as usize;
        let plane = side * side;
        let mut arr = vec![0.0_f32; 3 * plane];
        for (x, y, p) in canvas.enumerate_pixels() {
            let pixel = if x >= left && x < left + new_w && y >= top && y < top + new_h {
                resized.get_pixel(x - left, y - top)
            } else {
                p
            };
            let yi = y as usize;
            let xi = x as usize;
            let idx = yi * side + xi;
            arr[idx] = pixel[0] as f32 / 255.0;
            arr[plane + idx] = pixel[1] as f32 / 255.0;
            arr[2 * plane + idx] = pixel[2] as f32 / 255.0;
        }
        (arr, ratio, pad_x, pad_y)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let paths = collect_inputs(&args.inputs, !args.no_recursive)?;
    if paths.is_empty() {
        bail!("no supported image files found");
    }

    let start = Instant::now();
    let mut detector = Detector::new(args.model.as_deref(), args.conf)?;
    if let Some(runs) = args.bench_runs {
        return run_benchmark(&mut detector, &paths, &args, runs);
    }

    let mut results = Vec::with_capacity(paths.len());

    for path in paths {
        let image = load_image(&path)?;
        let (measures, page_type, type_confidence) =
            detector.predict(&image.rgb, args.expand, args.trim, args.auto)
                .with_context(|| format!("inference failed for {}", path.display()))?;
        results.push(ImageResult {
            filename: path.to_string_lossy().to_string(),
            page_type,
            type_confidence: round3(type_confidence),
            measures,
        });
    }

    let process_time = start.elapsed().as_millis();
    let output = match args.format {
        OutputFormat::Json => {
            let payload = JsonResponse { process_time, results };
            if args.pretty {
                serde_json::to_string_pretty(&payload)?
            } else {
                serde_json::to_string(&payload)?
            }
        }
        OutputFormat::Mei => write_mei(&results, args.pretty)?,
    };

    match args.output {
        Some(path) => fs::write(path, output)?,
        None => {
            let mut stdout = io::stdout().lock();
            if let Err(err) = stdout.write_all(output.as_bytes()).and_then(|_| stdout.write_all(b"\n")) {
                if err.kind() != io::ErrorKind::BrokenPipe {
                    return Err(err.into());
                }
            }
        }
    }
    Ok(())
}

fn run_benchmark(detector: &mut Detector, paths: &[PathBuf], args: &Args, runs: usize) -> Result<()> {
    if runs == 0 {
        bail!("--bench-runs must be greater than zero");
    }
    let mut images = Vec::with_capacity(paths.len());
    for path in paths {
        images.push((path, load_image(path)?));
    }

    for _ in 0..args.warmup {
        for (_, image) in &images {
            let _ = detector.predict(&image.rgb, args.expand, args.trim, args.auto)?;
        }
    }

    let mut times = Vec::with_capacity(runs * images.len());
    let mut n_measures = 0usize;
    for _ in 0..runs {
        for (_, image) in &images {
            let start = Instant::now();
            let (measures, _, _) = detector.predict(&image.rgb, args.expand, args.trim, args.auto)?;
            times.push(start.elapsed().as_secs_f64() * 1000.0);
            n_measures = measures.len();
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = percentile(&times, 0.50);
    let p95 = percentile(&times, 0.95);
    println!("images={}", images.len());
    println!("runs={runs}");
    println!("samples={}", times.len());
    println!("measures_last={n_measures}");
    println!("mean_ms={mean:.1}");
    println!("median_ms={median:.1}");
    println!("p95_ms={p95:.1}");
    println!("min_ms={:.1}", times[0]);
    println!("max_ms={:.1}", times[times.len() - 1]);
    Ok(())
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn collect_inputs(inputs: &[PathBuf], recursive: bool) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for input in inputs {
        if input.is_file() {
            if is_image_path(input) {
                out.push(input.clone());
            }
        } else if input.is_dir() {
            if recursive {
                for entry in WalkDir::new(input).follow_links(true).into_iter().filter_map(Result::ok) {
                    let path = entry.path();
                    if path.is_file() && is_image_path(path) {
                        out.push(path.to_path_buf());
                    }
                }
            } else {
                for entry in fs::read_dir(input)? {
                    let path = entry?.path();
                    if path.is_file() && is_image_path(&path) {
                        out.push(path);
                    }
                }
            }
        } else {
            bail!("input does not exist: {}", input.display());
        }
    }
    out.sort_by(natural_path_cmp);
    out.dedup();
    Ok(out)
}

fn is_image_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).as_deref(),
        Some("jpg" | "jpeg" | "png" | "tif" | "tiff" | "webp")
    )
}

fn load_image(path: &Path) -> Result<LoadedImage> {
    let img = image::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    Ok(LoadedImage { rgb: img.to_rgb8() })
}

fn init_embedded_ort() -> Result<()> {
    let dir = std::env::temp_dir().join("measure-detector-v2-cli").join("onnxruntime-1.27.0");
    fs::create_dir_all(&dir)?;
    let lib_path = dir.join("libonnxruntime.so.1.27.0");

    let needs_write = match fs::metadata(&lib_path) {
        Ok(meta) => meta.len() != EMBEDDED_ORT.len() as u64,
        Err(_) => true,
    };
    if needs_write {
        fs::write(&lib_path, EMBEDDED_ORT)?;
    }

    let _committed = ort::init_from(&lib_path)
        .map_err(|e| anyhow::anyhow!("failed to initialize ONNX Runtime from embedded library: {e}"))?
        .commit();
    Ok(())
}

fn cmp_measure_bboxes(a: &Measure, b: &Measure) -> Ordering {
    let a = &a.bbox;
    let b = &b.bbox;
    if a.x1 >= b.x1 && a.y1 >= b.y1 {
        return Ordering::Greater;
    }
    if a.x1 < b.x1 && a.y1 < b.y1 {
        return Ordering::Less;
    }
    let denom = (a.y2 - a.y1).min(b.y2 - b.y1);
    let overlap_y = if denom > 0.0 {
        (a.y2 - b.y1).min(b.y2 - a.y1) / denom
    } else {
        0.0
    };
    if overlap_y >= 0.5 {
        if a.x1 < b.x1 { Ordering::Less } else { Ordering::Greater }
    } else if a.x1 < b.x1 {
        Ordering::Greater
    } else {
        Ordering::Less
    }
}

fn get_geometry(a: &BBox, b: &BBox) -> (f32, f32, f32) {
    let left = a.x1.max(b.x1);
    let top = a.y1.max(b.y1);
    let right = a.x2.min(b.x2);
    let bottom = a.y2.min(b.y2);
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let intersection = if right < left || bottom < top { 0.0 } else { (right - left) * (bottom - top) };
    (intersection, area_a, area_b)
}

fn remove_overlapping_measures(measures: &[Measure], thresh: f32) -> Vec<Measure> {
    let mut valid = vec![true; measures.len()];
    for (a, ma) in measures.iter().enumerate() {
        for (b, mb) in measures.iter().enumerate() {
            if a == b {
                continue;
            }
            let (intersection, area_a, area_b) = get_geometry(&ma.bbox, &mb.bbox);
            if intersection == 0.0 {
                continue;
            }
            let ioa_a = if area_a > 0.0 { intersection / area_a } else { 0.0 };
            let ioa_b = if area_b > 0.0 { intersection / area_b } else { 0.0 };
            let iou = intersection / (area_a + area_b - intersection);
            if ioa_a > thresh || ioa_b > thresh || iou > thresh {
                if ma.confidence > mb.confidence {
                    valid[b] = false;
                } else {
                    valid[a] = false;
                }
            }
        }
    }
    measures.iter().cloned().zip(valid).filter_map(|(m, ok)| ok.then_some(m)).collect()
}

fn detect_page_type(measures: &[Measure]) -> (String, f32) {
    if measures.is_empty() {
        return ("unknown".to_string(), 0.0);
    }
    let mut scores = [0.0_f32, 0.0_f32];
    for measure in measures {
        if measure.class_id >= scores.len() {
            continue;
        }
        let b = &measure.bbox;
        let area = (b.x2 - b.x1) * (b.y2 - b.y1);
        scores[measure.class_id] += area * measure.confidence;
    }
    let total = scores[0] + scores[1];
    if total == 0.0 {
        return ("unknown".to_string(), 0.0);
    }
    scores[0] /= total;
    scores[1] /= total;
    if scores[0] > scores[1] {
        ("handwritten".to_string(), scores[0])
    } else {
        ("typeset".to_string(), scores[1])
    }
}

fn unify_measures(measures: &mut [Measure], page_type: &str, mut expand: bool, mut trim: bool, auto: bool) {
    if !(expand || trim || auto) || measures.is_empty() {
        return;
    }
    if auto {
        expand = page_type == "typeset";
        trim = page_type == "typeset";
    }

    let mut system_tops = Vec::new();
    let mut system_bottoms = Vec::new();
    let mut cur_bbox = BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
    let mut cur_system_top = 1.0;
    let mut cur_system_bottom = 0.0;
    let mut measure_to_system = Vec::with_capacity(measures.len());

    for measure in measures.iter() {
        let bbox = &measure.bbox;
        let denom = (bbox.y2 - bbox.y1).min(cur_bbox.y2 - cur_bbox.y1);
        let overlap_y = if denom > 0.0 {
            (bbox.y2 - cur_bbox.y1).min(cur_bbox.y2 - bbox.y1) / denom
        } else {
            0.0
        };
        if bbox.x1 > cur_bbox.x1 && overlap_y > 0.5 {
            cur_system_top = f32::min(cur_system_top, bbox.y1);
            cur_system_bottom = f32::max(cur_system_bottom, bbox.y2);
        } else {
            system_tops.push(cur_system_top);
            system_bottoms.push(cur_system_bottom);
            cur_system_top = 1.0;
            cur_system_bottom = 0.0;
        }
        cur_bbox = bbox.clone();
        measure_to_system.push(system_tops.len());
    }
    system_tops.push(cur_system_top);
    system_bottoms.push(cur_system_bottom);

    if expand {
        for (i, measure) in measures.iter_mut().enumerate() {
            let sys = measure_to_system[i];
            measure.bbox.y1 = system_tops[sys];
            measure.bbox.y2 = system_bottoms[sys];
        }
    }

    if trim && measures.len() >= 2 {
        let mut c_vals = Vec::new();
        for i in 0..measures.len() - 1 {
            if measure_to_system[i] == measure_to_system[i + 1] {
                let ax2 = measures[i].bbox.x2;
                let bx1 = measures[i + 1].bbox.x1;
                if ax2 > bx1 {
                    let c = (ax2 - bx1) / 2.0;
                    measures[i].bbox.x2 -= c;
                    measures[i + 1].bbox.x1 += c;
                    c_vals.push(c);
                }
            }
        }
        if !c_vals.is_empty() {
            let c_mean = c_vals.iter().sum::<f32>() / c_vals.len() as f32;
            for i in 1..measures.len() - 1 {
                if measure_to_system[i] != measure_to_system[i + 1] {
                    measures[i].bbox.x2 -= c_mean;
                    measures[i + 1].bbox.x1 += c_mean;
                }
            }
            measures[0].bbox.x1 += c_mean;
            let last = measures.len() - 1;
            measures[last].bbox.x2 -= c_mean;
        }
    }
}

fn write_mei(results: &[ImageResult], pretty: bool) -> Result<String> {
    let indent = |level: usize| if pretty { "  ".repeat(level) } else { String::new() };
    let nl = if pretty { "\n" } else { "" };
    let mut out = String::new();
    out.push_str(r#"<mei xmlns="http://www.music-encoding.org/ns/mei">"#);
    out.push_str(nl);
    out.push_str(&format!("{}<music>{nl}{}<body>{nl}{}<mdiv>{nl}{}<score>{nl}{}<section>{nl}", indent(1), indent(2), indent(3), indent(4), indent(5)));
    let mut measure_idx = 1usize;
    for result in results {
        for _ in &result.measures {
            out.push_str(&format!(
                "{}<measure xml:id=\"measure_{}\" n=\"{}\" label=\"{}\" facs=\"#zone_{}\" />{}",
                indent(6), measure_idx, measure_idx, measure_idx, measure_idx, nl
            ));
            measure_idx += 1;
        }
    }
    out.push_str(&format!("{}{}</section>{nl}{}{}</score>{nl}{}{}</mdiv>{nl}{}{}</body>{nl}{}{}</music>{nl}", indent(5), "", indent(4), "", indent(3), "", indent(2), "", indent(1), ""));
    out.push_str(&format!("{}<facsimile>{nl}", indent(1)));

    measure_idx = 1;
    for (page_idx, result) in results.iter().enumerate() {
        let (width, height) = image_dimensions(Path::new(&result.filename)).unwrap_or((0, 0));
        out.push_str(&format!(
            r#"{}<surface xml:id="surface_{}" n="{}" ulx="0" uly="0" lrx="{}" lry="{}">{}"#,
            indent(2), page_idx + 1, page_idx + 1, width.saturating_sub(1), height.saturating_sub(1), nl
        ));
        out.push_str(&format!(
            r#"{}<graphic xml:id="graphic_{}" target="{}" width="{}px" height="{}px" />{}"#,
            indent(3), page_idx + 1, xml_escape(&result.filename), width, height, nl
        ));
        for measure in &result.measures {
            let x1 = (measure.bbox.x1 * width as f32).round() as u32;
            let y1 = (measure.bbox.y1 * height as f32).round() as u32;
            let x2 = (measure.bbox.x2 * width as f32).round() as u32;
            let y2 = (measure.bbox.y2 * height as f32).round() as u32;
            out.push_str(&format!(
                r#"{}<zone xml:id="zone_{}" type="measure" ulx="{}" uly="{}" lrx="{}" lry="{}" />{}"#,
                indent(3), measure_idx, x1, y1, x2, y2, nl
            ));
            measure_idx += 1;
        }
        out.push_str(&format!("{}</surface>{nl}", indent(2)));
    }
    out.push_str(&format!("{}</facsimile>{nl}</mei>", indent(1)));
    Ok(out)
}

fn image_dimensions(path: &Path) -> Option<(u32, u32)> {
    image::image_dimensions(path).ok()
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn natural_path_cmp(a: &PathBuf, b: &PathBuf) -> Ordering {
    a.to_string_lossy().cmp(&b.to_string_lossy())
}

fn round3(v: f32) -> f32 {
    (v * 1000.0).round() / 1000.0
}

fn round5(v: f32) -> f32 {
    (v * 100000.0).round() / 100000.0
}
