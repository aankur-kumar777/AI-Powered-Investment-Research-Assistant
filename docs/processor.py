
"""docs/processor.py


PDF text extraction utility with OCR fallback. Features:
- Fast text extraction using PyMuPDF (fitz).
- Fallback to pdfplumber when needed.
- OCR fallback using pytesseract for scanned pages.
- Small CLI to extract and save text for a PDF.


Requirements: pymupdf, pdfplumber, pillow, pytesseract
"""

"""docs/processor.py

PDF text extraction utility with OCR fallback. Features:
- Fast text extraction using PyMuPDF (fitz).
- Fallback to pdfplumber when needed.
- OCR fallback using pytesseract for scanned pages.
- Small CLI to extract and save text for a PDF.

Requirements: pymupdf, pdfplumber, pillow, pytesseract
"""
from typing import List, Optional
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract
import io
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Run pytesseract OCR on raw image bytes and return text."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return pytesseract.image_to_string(img)
    except Exception:
        logger.exception("OCR failed on image bytes")
        return ""


def extract_text_from_pdf(path: str, ocr_dpi: int = 150) -> str:
    """Extract text from a PDF file.

    Strategy:
    1. Try PyMuPDF page.get_text() (fast and usually good).
    2. If a page returns empty text, try pdfplumber's extraction for that page.
    3. If still empty, rasterize the page and OCR with pytesseract.

    Returns the aggregated text across pages.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    texts: List[str] = []

    try:
        doc = fitz.open(path)
    except Exception:
        logger.exception("Failed to open PDF with PyMuPDF: %s", path)
        raise

    for i, page in enumerate(doc):
        try:
            text = page.get_text().strip()
            if text:
                texts.append(text)
                continue
        except Exception:
            logger.exception("PyMuPDF text extraction failed on page %s", i)

        # Try pdfplumber for the specific page
        try:
            with pdfplumber.open(path) as pp:
                pdf_page = pp.pages[i]
                text_pp = (pdf_page.extract_text() or "").strip()
                if text_pp:
                    texts.append(text_pp)
                    continue
        except Exception:
            # pdfplumber may fail on malformed PDFs
            logger.debug("pdfplumber failed on page %s", i)

        # Fallback: render page to image and OCR
        try:
            pix = page.get_pixmap(dpi=ocr_dpi)
            img_bytes = pix.tobytes("png")
            ocr_text = _ocr_image_bytes(img_bytes)
            texts.append(ocr_text)
        except Exception:
            logger.exception("Rasterization/OCR failed on page %s", i)

    return " ".join(t for t in texts if t)


def save_extracted_text(pdf_path: str, out_path: Optional[str] = None) -> str:
    """Extract text from `pdf_path` and save to `out_path` (defaults to same name with .txt). Returns out_path."""
    out_path = out_path or os.path.splitext(pdf_path)[0] + ".txt"
    text = extract_text_from_pdf(pdf_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Saved extracted text to %s", out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from a PDF with OCR fallback")
    parser.add_argument("pdf", help="path to input PDF")
    parser.add_argument("--out", help="path to output text file", default=None)
    parser.add_argument("--ocr-dpi", help="rasterization DPI for OCR", type=int, default=150)

    args = parser.parse_args()
    save_extracted_text(args.pdf, args.out)
