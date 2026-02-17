"""Storage manager for pipeline intermediate and final outputs."""
import json
from pathlib import Path
from typing import Dict, Any, List
from src.utils.logger import setup_logger

logger = setup_logger("storage")


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
        path = self.detection_dir / f"{document_id}_detection.json"
        self._write_json(path, sections)

    def save_section_json(self, document_id: str, section_name: str, data: Dict, confidence: float):
        if not self.save_intermediates:
            return
        safe_name = section_name.replace(" ", "_").replace("/", "_")
        path = self.sections_dir / f"{document_id}_{safe_name}.json"
        self._write_json(path, data)

    def save_final_json(self, document_id: str, data: Dict):
        path = self.final_dir / f"{document_id}.json"
        self._write_json(path, data)
        logger.info(f"Saved final: {path}")

    def save_review_results(self, document_id: str, results: Dict):
        if not self.save_intermediates:
            return
        path = self.intermediate_dir / f"{document_id}_review.json"
        self._write_json(path, results)

    def save_plain_text(self, document_id: str, text: str):
        if not self.save_intermediates:
            return
        path = self.intermediate_dir / f"{document_id}_plain.txt"
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def _write_json(path: Path, data: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
