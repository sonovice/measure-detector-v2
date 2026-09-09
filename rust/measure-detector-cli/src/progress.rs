//! Terminal-only PDF progress, separate from machine-readable output on stdout.
use std::path::Path;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressState, ProgressStyle};

pub(crate) struct PdfProgress {
    bar: Option<ProgressBar>,
}

impl PdfProgress {
    pub(crate) fn new(path: &Path, pages: usize, enabled: bool, action: &str) -> Self {
        if !enabled || pages <= 1 {
            return Self { bar: None };
        }
        // Indicatif defaults to stderr and hides the bar for non-terminal targets.
        let bar = ProgressBar::new(pages as u64);
        if bar.is_hidden() {
            return Self { bar: None };
        }
        bar.set_style(
            ProgressStyle::with_template(
                "{prefix}\n{spinner:.green} [{bar:24.cyan/blue}] {pos}/{len} pages | {msg}\nelapsed {elapsed_precise} | ETA {pdf_eta}",
            )
            .expect("valid PDF progress template")
            .with_key("pdf_eta", |state: &ProgressState, out: &mut dyn std::fmt::Write| {
                // No estimate is available before the first completed page.
                if state.pos() == 0 {
                    let _ = write!(out, "--");
                } else {
                    let seconds = state.eta().as_secs();
                    let _ = write!(out, "{:02}:{:02}:{:02}", seconds / 3600, seconds / 60 % 60, seconds % 60);
                }
            })
            .progress_chars("=>-"),
        );
        let filename = path
            .file_name()
            .unwrap_or(path.as_os_str())
            .to_string_lossy();
        let filename: String = filename
            .chars()
            .map(|c| if c.is_control() { '?' } else { c })
            .collect();
        bar.set_prefix(format!("{action} {filename}"));
        bar.set_message(format!("page 1; {pages} remaining"));
        bar.enable_steady_tick(Duration::from_millis(100));
        Self { bar: Some(bar) }
    }

    // Called only after both rendering and inference succeed, so ETA includes both.
    pub(crate) fn page_completed(&self) {
        let Some(bar) = &self.bar else { return };
        bar.inc(1);
        let remaining = bar.length().unwrap_or(0).saturating_sub(bar.position());
        if remaining == 0 {
            bar.finish_with_message("done; 0 remaining");
        } else {
            bar.set_message(format!(
                "page {}; {remaining} remaining",
                bar.position() + 1
            ));
        }
    }
}

impl Drop for PdfProgress {
    fn drop(&mut self) {
        if let Some(bar) = &self.bar {
            if !bar.is_finished() {
                // An error must not leave a live spinner or falsely report 100%.
                bar.abandon_with_message("failed");
            }
        }
    }
}

/// Make blocking setup work visible before a PDF page count/ETA is available.
pub(crate) fn stage<T>(
    enabled: bool,
    message: &str,
    work: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    if !enabled {
        return work();
    }
    let bar = ProgressBar::new_spinner();
    if bar.is_hidden() {
        return work();
    }
    bar.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg} | elapsed {elapsed_precise}")
            .expect("valid startup progress template"),
    );
    let message: String = message
        .chars()
        .map(|c| if c.is_control() { '?' } else { c })
        .collect();
    bar.set_message(message.clone());
    // Draw before entering the blocking call; do not wait for the ticker thread.
    bar.force_draw();
    bar.enable_steady_tick(Duration::from_millis(100));
    let result = work();
    match &result {
        Ok(_) => bar.finish_with_message(format!("{message}: done")),
        Err(_) => bar.abandon_with_message(format!("{message}: failed")),
    }
    result
}
