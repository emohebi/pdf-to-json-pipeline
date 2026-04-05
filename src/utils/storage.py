"""Storage manager for pipeline intermediate and final outputs."""
import json
import re
from pathlib import Path
from typing import Dict, Any, List
from src.utils.logger import setup_logger
from src.utils.json_to_excel import write_excel

logger = setup_logger("storage")


def _sanitize_filename(name: str, max_length: int = 80) -> str:
    safe = re.sub(r'[^\w\-.]', '_', name)
    safe = re.sub(r'_+', '_', safe)
    safe = safe.strip('_.')
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip('_.')
    return safe or "unnamed"


def _make_safe_path(directory: Path, *parts: str, extension: str = ".json") -> Path:
    WINDOWS_MAX_PATH = 255
    dir_str = str(directory.resolve())
    available = WINDOWS_MAX_PATH - len(dir_str) - 1 - len(extension)
    available = max(available, 30)
    joined = "_".join(_sanitize_filename(p) for p in parts)
    if len(joined) > available:
        joined = joined[:available].rstrip('_.')
    return directory / f"{joined}{extension}"


class StorageManager:
    """Manages saving/loading of pipeline artifacts."""

    def __init__(self):
        from config.settings import (
            DETECTION_DIR, SECTIONS_DIR, FINAL_DIR,
            INTERMEDIATE_DIR, SAVE_INTERMEDIATES, TERM_MATCHING_DIR,
            EFFECTIVE_DATE_DIR, UOM_EXTRACTION_DIR, BATCHES_DIR,
            VERIFICATION_DIR, CPI_ADJUSTMENT_DIR
        )
        self.detection_dir = DETECTION_DIR
        self.sections_dir = SECTIONS_DIR
        self.batch_dir = BATCHES_DIR
        self.final_dir = FINAL_DIR
        self.intermediate_dir = INTERMEDIATE_DIR
        self.save_intermediates = SAVE_INTERMEDIATES
        self.term_matching_dir = TERM_MATCHING_DIR
        self.effective_date_dir = EFFECTIVE_DATE_DIR
        self.uom_extraction_dir = UOM_EXTRACTION_DIR
        self.verification_dir = VERIFICATION_DIR
        self.cpi_adjustment_dir = CPI_ADJUSTMENT_DIR

    def save_cpi_adjustment_result(self, document_id: str, report: Dict):
        path = _make_safe_path(self.cpi_adjustment_dir, document_id, "cpi_adjustment")
        self._write_json(path, report)
        logger.info(f"Saved CPI adjustment report: {path}")

    def save_detection_result(self, document_id: str, sections: Any):
        if not self.save_intermediates or sections is None:
            return
        path = _make_safe_path(self.detection_dir, document_id, "detection")
        self._write_json(path, sections)

    def save_batch_json(self, section_name: str, data: Dict):
        path = _make_safe_path(self.batch_dir, "", section_name)
        self._write_json(path, data)

    def save_section_json(self, document_id: str, section_name: str, data: Dict, confidence: float):
        if not self.save_intermediates:
            return
        path = _make_safe_path(self.sections_dir, document_id, section_name)
        self._write_json(path, data)

    def save_final_json(self, document_id: str, data: Dict):
        path = _make_safe_path(self.final_dir, document_id)
        self._write_json(path, data)
        logger.info(f"Saved final: {path}")
        with open(path, encoding="utf-8") as f:
            document = json.load(f)

        output_path = path.with_suffix(".xlsx")

        excel_path = write_excel(document, output_path)
        logger.info(f"Saved final excel: {excel_path}")

    def save_review_results(self, document_id: str, results: Dict):
        if not self.save_intermediates:
            return
        path = _make_safe_path(self.intermediate_dir, document_id, "review")
        self._write_json(path, results)

    def save_term_matching_result(self, document_id: str, report: Dict):
        path = _make_safe_path(self.term_matching_dir, document_id, "term_matching")
        self._write_json(path, report)
        logger.info(f"Saved term matching report: {path}")

    def save_effective_date_result(self, document_id: str, report: Dict):
        path = _make_safe_path(self.effective_date_dir, document_id, "effective_date")
        self._write_json(path, report)
        logger.info(f"Saved effective date report: {path}")

    def save_uom_extraction_result(self, document_id: str, report: Dict):
        path = _make_safe_path(self.uom_extraction_dir, document_id, "uom_extraction")
        self._write_json(path, report)
        logger.info(f"Saved UOM extraction report: {path}")

    def save_verification_result(self, document_id: str, report: Dict):
        """Save extraction verification report."""
        path = _make_safe_path(self.verification_dir, document_id, "verification")
        self._write_json(path, report)
        logger.info(f"Saved verification report: {path}")

    def save_plain_text(self, document_id: str, text: str):
        if not self.save_intermediates:
            return
        path = _make_safe_path(self.intermediate_dir, document_id, "plain", extension=".txt")
        path.write_text(text, encoding="utf-8")

    def get_validation_queue(self) -> List[Dict]:
        from config.settings import VALIDATION_QUEUE_DIR
        queue = []
        if VALIDATION_QUEUE_DIR.exists():
            for f in VALIDATION_QUEUE_DIR.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        queue.append(json.load(fh))
                except Exception as e:
                    logger.warning(f"Failed to read {f}: {e}")
        return queue

    def approve_document(self, document_id: str, reviewer: str = None):
        logger.info(f"Approved: {document_id} by {reviewer}")

    def reject_document(self, document_id: str, reason: str, reviewer: str = None):
        logger.info(f"Rejected: {document_id} by {reviewer}: {reason}")

    @staticmethod
    def _write_json(path: Path, data: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
