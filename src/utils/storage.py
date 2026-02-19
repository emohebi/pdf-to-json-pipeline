"""Storage manager for pipeline intermediate and final outputs."""
import json
import re
from pathlib import Path
from typing import Dict, Any, List
from src.utils.logger import setup_logger

logger = setup_logger("storage")


def _sanitize_filename(name: str, max_length: int = 80) -> str:
    """
    Sanitize a string for use as a filename component.

    Replaces any character that is not alphanumeric, hyphen, underscore,
    or period with an underscore. Collapses consecutive underscores.
    """
    # Replace any non-safe character with underscore
    safe = re.sub(r'[^\w\-.]', '_', name)
    # Collapse multiple underscores
    safe = re.sub(r'_+', '_', safe)
    # Strip leading/trailing underscores and dots
    safe = safe.strip('_.')
    # Truncate to max length
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip('_.')
    return safe or "unnamed"


def _make_safe_path(directory: Path, *parts: str, extension: str = ".json") -> Path:
    """
    Build a file path from sanitized parts, ensuring the FULL path
    stays under Windows' MAX_PATH (260 chars).

    Dynamically calculates available filename space based on the
    actual directory path length.
    """
    WINDOWS_MAX_PATH = 255
    # Resolve to get the full absolute directory path
    dir_str = str(directory.resolve())
    # Available chars for filename = MAX_PATH - dir path - separator - extension
    available = WINDOWS_MAX_PATH - len(dir_str) - 1 - len(extension)
    # Minimum usable filename length
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
            INTERMEDIATE_DIR, SAVE_INTERMEDIATES,
        )
        self.detection_dir = DETECTION_DIR
        self.sections_dir = SECTIONS_DIR
        self.final_dir = FINAL_DIR
        self.intermediate_dir = INTERMEDIATE_DIR
        self.save_intermediates = SAVE_INTERMEDIATES

    def save_detection_result(self, document_id: str, sections: Any):
        if not self.save_intermediates or sections is None:
            return
        path = _make_safe_path(self.detection_dir, document_id, "detection")
        self._write_json(path, sections)

    def save_section_json(self, document_id: str, section_name: str, data: Dict, confidence: float):
        if not self.save_intermediates:
            return
        path = _make_safe_path(self.sections_dir, document_id, section_name)
        self._write_json(path, data)

    def save_final_json(self, document_id: str, data: Dict):
        path = _make_safe_path(self.final_dir, document_id)
        self._write_json(path, data)
        logger.info(f"Saved final: {path}")

    def save_review_results(self, document_id: str, results: Dict):
        if not self.save_intermediates:
            return
        path = _make_safe_path(self.intermediate_dir, document_id, "review")
        self._write_json(path, results)

    def save_plain_text(self, document_id: str, text: str):
        if not self.save_intermediates:
            return
        path = _make_safe_path(self.intermediate_dir, document_id, "plain", extension=".txt")
        path.write_text(text, encoding="utf-8")

    def get_validation_queue(self) -> List[Dict]:
        """Get all documents pending validation."""
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