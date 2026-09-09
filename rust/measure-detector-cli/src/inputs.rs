//! Page-at-a-time PDF rendering keeps normal detection memory bounded.
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use anyhow::{Context, Result, bail};
use tempfile::TempDir;

use crate::{LoadedImage, load_image};

pub(crate) struct Pages {
    source: PathBuf,
    // Own the temporary directory until iteration finishes (including early errors).
    directory: Option<TempDir>,
    next: usize,
    count: usize,
}

pub(crate) fn poppler(command: &mut Command) -> Result<Output> {
    let output = command
        .env("LC_ALL", "C")
        .output()
        .context("PDF support requires Poppler (pdfinfo and pdftoppm) on PATH")?;
    if !output.status.success() {
        bail!(
            "Cannot read PDF: invalid, damaged, or password-protected document: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output)
}

pub(crate) fn load_pages(path: &Path) -> Result<Pages> {
    let is_pdf = path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"));
    if !is_pdf {
        return Ok(Pages {
            source: path.to_owned(),
            directory: None,
            next: 1,
            count: 1,
        });
    }
    // Absolute paths also prevent filenames starting with '-' being treated as options.
    let source = path
        .canonicalize()
        .with_context(|| format!("failed to open {}", path.display()))?;
    let output = poppler(Command::new("pdfinfo").arg(&source))?;
    let info = String::from_utf8_lossy(&output.stdout);
    let count: usize = info
        .lines()
        .find_map(|line| line.strip_prefix("Pages:"))
        .context("PDF page count is missing")?
        .trim()
        .parse()
        .context("Invalid PDF page count")?;
    if count == 0 {
        bail!("PDF contains no pages");
    }
    Ok(Pages {
        source,
        directory: Some(tempfile::tempdir()?),
        next: 1,
        count,
    })
}

impl Pages {
    pub(crate) fn page_count(&self) -> usize {
        self.count
    }
}

impl Iterator for Pages {
    type Item = Result<LoadedImage>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next > self.count {
            return None;
        }
        let page = self.next;
        self.next += 1;
        let result = (|| {
            let Some(directory) = &self.directory else {
                return load_image(&self.source);
            };
            let prefix = directory.path().join("page");
            poppler(
                Command::new("pdftoppm")
                    .args([
                        "-f",
                        &page.to_string(),
                        "-l",
                        &page.to_string(),
                        "-singlefile",
                        "-r",
                        "150",
                        "-png",
                    ])
                    .arg(&self.source)
                    .arg(&prefix),
            )
            .with_context(|| format!("failed to render {} page {page}", self.source.display()))?;
            let mut image = load_image(&prefix.with_extension("png"))?;
            image.page = Some(page);
            Ok(image)
        })();
        if result.is_err() {
            self.next = self.count + 1;
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BBox, ImageResult, JsonResponse, Measure, collect_inputs, write_mei};

    #[test]
    fn multipage_pdf_produces_one_json_and_mei_with_page_dimensions() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let pdf = dir.path().join("score.PDF");
        std::fs::write(
            &pdf,
            include_bytes!("../../../tests/fixtures/two-pages.pdf"),
        )?;
        let pages = load_pages(&pdf)?.collect::<Result<Vec<_>>>()?;
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].rgb.dimensions(), (300, 150));
        assert_eq!(pages[1].rgb.dimensions(), (150, 300));
        assert_eq!(pages[0].rgb.get_pixel(75, 75).0, [255, 0, 0]);
        assert_eq!(pages[1].rgb.get_pixel(75, 225).0, [0, 0, 255]);
        let results = pages
            .into_iter()
            .map(|image| ImageResult {
                filename: "score.PDF".into(),
                page: image.page,
                dimensions: image.rgb.dimensions(),
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
        let response = JsonResponse {
            process_time: 0,
            results,
        };
        let json = serde_json::to_string(&response)?;
        let output = dir.path().join("results.json");
        std::fs::write(&output, json)?;
        let parsed: serde_json::Value = serde_json::from_slice(&std::fs::read(output)?)?;
        assert_eq!(parsed["results"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["results"][0]["page"], 1);
        assert_eq!(parsed["results"][1]["page"], 2);
        assert!(parsed["results"][0].get("dimensions").is_none());
        let mei = write_mei(&response.results, true)?;
        assert!(mei.contains("score.PDF#page=1\" width=\"300px\""));
        assert!(mei.contains("score.PDF#page=2\" width=\"150px\""));
        assert!(mei.contains("ulx=\"15\" uly=\"60\""));
        assert_eq!(collect_inputs(&[dir.path().to_owned()], true)?, vec![pdf]);
        Ok(())
    }

    #[test]
    fn image_compatibility_and_invalid_pdf() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let image = dir.path().join("image.png");
        image::RgbImage::new(20, 10).save(&image)?;
        let pages = load_pages(&image)?.collect::<Result<Vec<_>>>()?;
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].page, None);
        let pdf = dir.path().join("bad.pdf");
        std::fs::write(&pdf, b"%PDF-broken")?;
        assert!(load_pages(&pdf).is_err());
        Ok(())
    }
}
