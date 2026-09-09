//! XFDF annotations use unrotated PDF coordinates, not rendered pixels.
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::inputs::poppler;
use crate::{BBox, ImageResult, xml_escape};

#[derive(Clone, Copy, Debug)]
pub(crate) struct PageGeometry {
    media_box: [f64; 4],
    rotation: i32,
}

impl PageGeometry {
    fn rect(&self, bbox: &BBox) -> [f64; 4] {
        let [left, bottom, right, top] = self.media_box;
        let point = |u: f32, v: f32| {
            let (u, v) = (f64::from(u), f64::from(v));
            let (x, y) = match self.rotation {
                90 => (v, u),
                180 => (1.0 - u, v),
                270 => (1.0 - v, 1.0 - u),
                _ => (u, 1.0 - v),
            };
            (left + x * (right - left), bottom + y * (top - bottom))
        };
        let a = point(bbox.x1, bbox.y1);
        let b = point(bbox.x2, bbox.y2);
        [a.0.min(b.0), a.1.min(b.1), a.0.max(b.0), a.1.max(b.1)]
    }
}

pub(crate) fn pdf_geometry(path: &Path) -> Result<Vec<PageGeometry>> {
    let source = path.canonicalize()?;
    let output = poppler(
        Command::new("pdfinfo")
            .args(["-box", "-f", "1", "-l", "2147483647"])
            .arg(source),
    )?;
    parse_geometry(&String::from_utf8_lossy(&output.stdout))
}

fn parse_geometry(info: &str) -> Result<Vec<PageGeometry>> {
    let count: usize = info
        .lines()
        .find_map(|line| line.strip_prefix("Pages:"))
        .context("PDF page count is missing")?
        .trim()
        .parse()?;
    if count == 0 {
        bail!("PDF contains no pages");
    }
    let mut boxes = vec![None; count];
    let mut rotations = vec![None; count];
    for line in info.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() < 4 || fields[0] != "Page" {
            continue;
        }
        let Ok(page) = fields[1].parse::<usize>() else {
            continue;
        };
        if page == 0 || page > count {
            continue;
        }
        match fields[2] {
            "MediaBox:" if fields.len() == 7 => {
                let mut values = [0.0_f64; 4];
                for (index, value) in fields[3..].iter().enumerate() {
                    values[index] = value.parse()?;
                }
                boxes[page - 1] = Some(values);
            }
            "rot:" => rotations[page - 1] = Some(fields[3].parse::<i32>()?.rem_euclid(360)),
            _ => {}
        }
    }
    boxes
        .into_iter()
        .zip(rotations)
        .enumerate()
        .map(|(page, (bounds, rotation))| {
            let media_box =
                bounds.with_context(|| format!("Missing MediaBox for page {}", page + 1))?;
            let rotation = rotation.context("Missing PDF page rotation")?;
            if !media_box.iter().all(|v| v.is_finite())
                || media_box[2] <= media_box[0]
                || media_box[3] <= media_box[1]
                || ![0, 90, 180, 270].contains(&rotation)
            {
                bail!("Invalid PDF geometry for page {}", page + 1);
            }
            Ok(PageGeometry {
                media_box,
                rotation,
            })
        })
        .collect()
}

pub(crate) fn validate_input(paths: &[std::path::PathBuf]) -> Result<()> {
    if paths.len() != 1
        || !paths[0]
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
    {
        bail!("XFDF output requires exactly one PDF input");
    }
    Ok(())
}

pub(crate) fn write_xfdf(
    filename: &str,
    results: &[ImageResult],
    geometry: &[PageGeometry],
    pretty: bool,
) -> Result<String> {
    use std::fmt::Write;
    let nl = if pretty { "\n" } else { "" };
    let indent = if pretty { "  " } else { "" };
    let mut out = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>{nl}<xfdf xmlns=\"http://ns.adobe.com/xfdf/\" xml:space=\"preserve\">{nl}{indent}<f href=\"{}\"/>{nl}{indent}<annots>{nl}",
        xml_escape(filename)
    );
    let mut number = 0;
    for result in results {
        let page = result
            .page
            .context("XFDF output requires PDF page numbers")?;
        let geometry = page
            .checked_sub(1)
            .and_then(|index| geometry.get(index))
            .context("Missing geometry for PDF page")?;
        for measure in &result.measures {
            number += 1;
            let rect = geometry.rect(&measure.bbox);
            let (mut width, mut height) = (rect[2] - rect[0], rect[3] - rect[1]);
            if [90, 270].contains(&geometry.rotation) {
                std::mem::swap(&mut width, &mut height);
            }
            let size = 150.0_f64
                .min(height)
                .min(width / (0.6 * number.to_string().len() as f64 + 0.3));
            let style = format!(
                "font-family: \"Courier New\", monospace; text-align: center; font-weight: bold; font-size: {size:.6}pt; color: #000000;"
            );
            write!(
                out,
                "{indent}{indent}<freetext name=\"bar-{}\" page=\"{}\" subject=\"{number}\" rect=\"{:.6},{:.6},{:.6},{:.6}\" title=\"BarNumber\" flags=\"print,readonly,locked\" color=\"#ffffff\" opacity=\"0.1\" width=\"0.0\" justification=\"center\" rotation=\"{}\" style=\"cloudy\" intensity=\"0.0\">{nl}",
                number - 1,
                page - 1,
                rect[0],
                rect[1],
                rect[2],
                rect[3],
                geometry.rotation
            )?;
            write!(
                out,
                "{indent}{indent}{indent}<contents>{number}</contents>{nl}{indent}{indent}{indent}<contents-richtext><body xmlns=\"http://www.w3.org/1999/xhtml\" xmlns:xfa=\"http://www.xfa.org/schema/xfa-data/1.0/\" xfa:spec=\"2.0.2\" xfa:APIVersion=\"Acrobat:11.0.0\"><p style=\"{}\">{number}</p></body></contents-richtext>{nl}",
                xml_escape(&style)
            )?;
            write!(
                out,
                "{indent}{indent}{indent}<defaultstyle>{}</defaultstyle>{nl}{indent}{indent}{indent}<defaultappearance>0 0 0 rg /Courier#20New {size:.6} Tf</defaultappearance>{nl}{indent}{indent}</freetext>{nl}",
                xml_escape(&style)
            )?;
        }
    }
    write!(out, "{indent}</annots>{nl}</xfdf>")?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Args, Measure, OutputFormat};
    use clap::Parser;

    #[test]
    fn rendered_rotated_pages_recover_original_pdf_rectangle() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("rotated.pdf");
        std::fs::write(
            &path,
            include_bytes!("../../../tests/fixtures/rotated-offset.pdf"),
        )?;
        let geometries = pdf_geometry(&path)?;
        for (index, image) in crate::inputs::load_pages(&path)?.enumerate() {
            let rgb = image?.rgb;
            let mut bounds = [rgb.width(), rgb.height(), 0, 0];
            for (x, y, pixel) in rgb.enumerate_pixels() {
                if pixel[0] > 240 && pixel[1] < 10 {
                    bounds = [
                        bounds[0].min(x),
                        bounds[1].min(y),
                        bounds[2].max(x + 1),
                        bounds[3].max(y + 1),
                    ];
                }
            }
            let bbox = BBox {
                x1: bounds[0] as f32 / rgb.width() as f32,
                y1: bounds[1] as f32 / rgb.height() as f32,
                x2: bounds[2] as f32 / rgb.width() as f32,
                y2: bounds[3] as f32 / rgb.height() as f32,
            };
            assert_eq!(geometries[index].rotation, index as i32 * 90);
            for (actual, expected) in geometries[index]
                .rect(&bbox)
                .iter()
                .zip([50.0, 40.0, 150.0, 80.0])
            {
                assert!((actual - expected).abs() < 0.8, "{actual} != {expected}");
            }
        }
        Ok(())
    }

    #[test]
    fn output_matches_example_structure_and_escapes_filename() -> Result<()> {
        let geometry = vec![
            PageGeometry {
                media_box: [0.0, 0.0, 144.0, 72.0],
                rotation: 0
            };
            2
        ];
        let results: Vec<_> = (1..=2)
            .map(|page| ImageResult {
                filename: "score.pdf".into(),
                page: Some(page),
                dimensions: (300, 150),
                page_type: "typeset".into(),
                type_confidence: 0.9,
                measures: vec![Measure {
                    class_id: 1,
                    class_name: "typeset".into(),
                    confidence: 0.9,
                    bbox: BBox {
                        x1: 0.1,
                        y1: 0.2,
                        x2: 0.8,
                        y2: 0.9,
                    },
                }],
            })
            .collect();
        for pretty in [false, true] {
            let xml = write_xfdf("score & \"parts\".pdf", &results, &geometry, pretty)?;
            let doc = roxmltree::Document::parse(&xml)?;
            let root = doc.root_element();
            assert_eq!(
                root.tag_name().namespace(),
                Some("http://ns.adobe.com/xfdf/")
            );
            let file = root
                .children()
                .find(|node| node.has_tag_name(("http://ns.adobe.com/xfdf/", "f")))
                .unwrap();
            assert_eq!(file.attribute("href"), Some("score & \"parts\".pdf"));
            let annots: Vec<_> = root
                .descendants()
                .filter(|node| node.has_tag_name(("http://ns.adobe.com/xfdf/", "freetext")))
                .collect();
            assert_eq!(annots.len(), 2);
            for (index, annot) in annots.iter().enumerate() {
                assert_eq!(annot.attribute("page"), Some(index.to_string().as_str()));
                assert_eq!(
                    annot.attribute("subject"),
                    Some((index + 1).to_string().as_str())
                );
                assert_eq!(annot.attribute("title"), Some("BarNumber"));
                assert_eq!(annot.attribute("opacity"), Some("0.1"));
                let rect: Vec<f64> = annot
                    .attribute("rect")
                    .unwrap()
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
                for (actual, expected) in rect.iter().zip([14.4, 7.2, 115.2, 57.6]) {
                    assert!((actual - expected).abs() < 0.00001);
                }
                let text = annot
                    .descendants()
                    .find(|node| node.has_tag_name(("http://www.w3.org/1999/xhtml", "p")))
                    .unwrap();
                assert_eq!(text.text(), Some((index + 1).to_string().as_str()));
            }
        }
        let empty = write_xfdf("empty.pdf", &[], &geometry, false)?;
        assert!(roxmltree::Document::parse(&empty).is_ok());
        Ok(())
    }

    #[test]
    fn cli_accepts_format_and_rejects_ambiguous_targets() -> Result<()> {
        let args = Args::try_parse_from(["measure-detector-v2", "--format", "xfdf", "score.pdf"])?;
        assert!(matches!(args.format, OutputFormat::Xfdf));
        assert!(validate_input(&["score.PDF".into()]).is_ok());
        assert!(validate_input(&["image.png".into()]).is_err());
        assert!(validate_input(&["one.pdf".into(), "two.pdf".into()]).is_err());
        assert!(parse_geometry("Pages: 1\nPage 1 rot: 45\nPage 1 MediaBox: 0 0 10 20").is_err());
        Ok(())
    }
}
