"""Decode images and render PDF pages at 150 dpi using Poppler."""
from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from measure_detector_v2.detector import MeasureDetector


def _poppler(*args: str) -> bytes:
    try:
        result = subprocess.run(args, capture_output=True, timeout=120,
                                env={**os.environ, "LC_ALL": "C"})
    except FileNotFoundError as exc:
        raise RuntimeError("PDF support requires Poppler (pdfinfo and pdftoppm)") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("PDF processing timed out") from exc
    if result.returncode:
        raise ValueError("Cannot read PDF: invalid, damaged, or password-protected document")
    return result.stdout


def decode_pages(data: bytes, filename: str | None = None, content_type: str | None = None,
                 page: int | None = None) -> Iterator[tuple[int | None, np.ndarray]]:
    """Yield RGB pages in order; image inputs have no PDF page number.

    A selected page is one-based. Close the iterator to release temporary files.
    """
    if page is not None and page < 1:
        raise ValueError("page must be greater than zero")
    is_pdf = (b"%PDF-" in data[:1024] or (filename or "").lower().endswith(".pdf")
              or content_type == "application/pdf")
    if not is_pdf:
        if page not in (None, 1):
            raise ValueError("Image inputs only have one page")
        yield None, MeasureDetector.decode_image(data)
        return
    with TemporaryDirectory(prefix="measure-detector-pdf-") as directory:
        source = Path(directory) / "input.pdf"
        source.write_bytes(data)
        info = _poppler("pdfinfo", str(source)).decode("utf-8", errors="replace")
        count = next((int(line.split(":", 1)[1].strip()) for line in info.splitlines()
                      if line.startswith("Pages:")), 0)
        if count < 1:
            raise ValueError("PDF contains no pages")
        if page is not None and page > count:
            raise ValueError(f"PDF contains only {count} pages")
        prefix = Path(directory) / "page"
        for number in ([page] if page is not None else range(1, count + 1)):
            _poppler("pdftoppm", "-f", str(number), "-l", str(number), "-singlefile",
                     "-r", "150", "-png", str(source), str(prefix))
            yield number, MeasureDetector.decode_image(prefix.with_suffix(".png").read_bytes())
