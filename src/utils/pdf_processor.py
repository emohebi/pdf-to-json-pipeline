"""PDF page extraction utilities."""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
from config.settings import DPI
from src.utils.logger import setup_logger

logger = setup_logger("pdf_processor")


def extract_pages(pdf_path: str, dpi: int = None) -> List[Dict[str, Any]]:
    """Extract all pages from PDF as images with metadata."""
    dpi = dpi or DPI
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        pages.append({
            "page_number": page_num + 1,
            "image": pix.tobytes("png"),
            "width": pix.width,
            "height": pix.height,
            "text": page.get_text(),
        })
    doc.close()
    logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
    return pages
