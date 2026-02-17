"""
Stage 3.5: Review Agent - config-driven.
Reviews each section with its page images using the review_template from config.
No hardcoded section types or domain knowledge.
"""
import json
import re
from typing import Dict, List, Any

from config.settings import MODEL_MAX_TOKENS_VALIDATION
from config.config_loader import get_prompt, render_prompt
from src.tools.llm_provider import invoke_multimodal
from src.tools.bedrock_vision import prepare_images_for_bedrock
from src.utils import setup_logger, StorageManager

logger = setup_logger("review_agent")


class ReviewAgent:
    """Agent to review and validate extracted section content."""

    def __init__(self):
        self.storage = StorageManager()

    def review_document(
        self,
        section_jsons: List[Dict],
        document_id: str,
        pages_data: List[Dict] = None,
    ) -> Dict[str, Any]:
        logger.info(f"[{document_id}] Starting document review")

        try:
            aggregated = {
                "WORD_ACCURACY": [],
                "MISSING_INFORMATION_ACCURACY": [],
                "DUPLICATION_ACCURACY": [],
            }

            logger.info(f"[{document_id}] Reviewing {len(section_jsons)} sections...")
            for idx, section in enumerate(section_jsons, 1):
                section_name = section.get("section_name", f"Section {idx}")
                logger.info(f"  [{idx}/{len(section_jsons)}] Reviewing: {section_name}")

                section_plain_text = self._section_json_to_plain_text(section, section_name)

                section_images = None
                if pages_data:
                    page_range = section.get("page_range", [])
                    if page_range and len(page_range) == 2:
                        section_pages = pages_data[page_range[0] - 1 : page_range[1]]
                        section_images = prepare_images_for_bedrock(section_pages)

                review = self._validate_section_content(
                    section_plain_text, section_name, section_images, document_id
                )
                for key in aggregated:
                    if key in review and review[key]:
                        aggregated[key].extend(review[key])

            self.storage.save_review_results(document_id, aggregated)

            full_plain = self._all_sections_to_plain_text(section_jsons)
            self.storage.save_plain_text(document_id, full_plain)

            total_issues = sum(len(v) for v in aggregated.values())
            if total_issues == 0:
                logger.info(f"[{document_id}] Review passed -- no issues found")
            else:
                logger.warning(f"[{document_id}] Review found {total_issues} issue(s)")

            return aggregated

        except Exception as e:
            logger.error(f"[{document_id}] Review failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Plain text conversion helpers
    # ------------------------------------------------------------------

    def _section_json_to_plain_text(self, section: Dict, section_name: str) -> str:
        try:
            data = section.get("data", section)
            return f"=== {section_name} ===\n\n{self._extract_text(data)}".strip()
        except Exception as e:
            logger.error(f"Failed to convert section {section_name} to text: {e}")
            return f"=== {section_name} ===\n\n[Error converting section]"

    def _extract_text(self, data: Any, indent: int = 0) -> str:
        parts = []
        prefix = "  " * indent
        if isinstance(data, dict):
            for key, value in data.items():
                if key.startswith("_") or key == "image" or "_orig" in key:
                    continue
                if key == "text" and isinstance(value, str) and value.strip():
                    parts.append(f"{prefix}{value}")
                else:
                    nested = self._extract_text(value, indent)
                    if nested:
                        parts.append(nested)
        elif isinstance(data, list):
            for item in data:
                nested = self._extract_text(item, indent)
                if nested:
                    parts.append(nested)
        elif isinstance(data, str) and data.strip():
            parts.append(f"{prefix}{data}")
        return "\n".join(parts)

    def _all_sections_to_plain_text(self, section_jsons: List[Dict]) -> str:
        return "\n\n".join(
            self._section_json_to_plain_text(s, s.get("section_name", "Unknown"))
            for s in section_jsons
        )

    # ------------------------------------------------------------------
    # Validation (config-driven prompt)
    # ------------------------------------------------------------------

    def _validate_section_content(
        self,
        section_plain_text: str,
        section_name: str,
        section_images: List[str] = None,
        document_id: str = None,
    ) -> Dict[str, List[Dict]]:
        empty = {"WORD_ACCURACY": [], "MISSING_INFORMATION_ACCURACY": [], "DUPLICATION_ACCURACY": []}
        if not section_images:
            return empty

        # Build prompt from config template
        template = get_prompt("review_template")
        prompt = render_prompt(
            template,
            section_name=section_name,
            extracted_text=section_plain_text,
        )

        counter = 3
        while True:
            try:
                response = invoke_multimodal(
                    images=section_images, prompt=prompt, max_tokens=MODEL_MAX_TOKENS_VALIDATION
                )
                cleaned = re.sub(r"```(?:json)?\s*", "", response)
                cleaned = re.sub(r"```\s*$", "", cleaned)
                match = re.search(r"\{[\s\S]*\}", cleaned)
                if not match:
                    raise ValueError("No JSON object found")
                review = json.loads(match.group(0))
                for key in ("WORD_ACCURACY", "MISSING_INFORMATION_ACCURACY", "DUPLICATION_ACCURACY"):
                    review.setdefault(key, [])
                return review
            except Exception as e:
                counter -= 1
                if counter <= 0:
                    logger.error(f"[{document_id}] Validation failed for {section_name}: {e}")
                    return empty
                logger.info(f"Retrying ({counter} left)...")
