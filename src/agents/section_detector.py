"""
Stage 1: Section Detection Agent - Fully config-driven.
All prompts, merge rules, and heading aliases come from config.json.
No hardcoded section types or domain knowledge.
"""
import json
from typing import List, Dict

from config.settings import MODEL_MAX_TOKENS_DETECTION, MAX_IMAGES_PER_BATCH
from config.config_loader import (
    get_section_definitions,
    get_detection_prompt_template,
    get_document_type_name,
    build_heading_alias_rules,
    render_prompt,
    get_merge_rules,
)
from src.tools.llm_provider import invoke_multimodal
from src.tools.bedrock_vision import prepare_images_for_bedrock
from src.utils import setup_logger, StorageManager

logger = setup_logger("section_detector")


class SectionDetectionAgent:
    """Identify logical sections in PDF documents using config-driven prompts."""

    def __init__(self):
        self.section_definitions = get_section_definitions()
        self.storage = StorageManager()
        self.max_per_call = MAX_IMAGES_PER_BATCH

    def detect_sections(self, pages_data: List[Dict], document_id: str) -> List[Dict]:
        total = len(pages_data)
        logger.info(f"[{document_id}] Detecting sections in {total} pages")
        try:
            sections = self._detect_multi_batch(pages_data, document_id)
            self.storage.save_detection_result(document_id, sections)
            logger.info(f"[{document_id}] Detected {len(sections)} sections")
            return sections
        except Exception as e:
            logger.error(f"[{document_id}] Detection failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _detect_multi_batch(self, pages_data, document_id):
        total = len(pages_data)
        n_batches = (total + self.max_per_call - 1) // self.max_per_call
        logger.info(f"[{document_id}] {total} pages -> {n_batches} batches")

        batch_sections = []
        for b in range(n_batches):
            s = b * self.max_per_call
            e = min(s + self.max_per_call, total)
            batch = pages_data[s:e]
            sp, ep = s + 1, e
            logger.info(f"[{document_id}] Batch {b+1}/{n_batches}: pages {sp}-{ep}")

            images = prepare_images_for_bedrock(batch)
            prompt = self._build_prompt(sp, ep, total)
            resp = invoke_multimodal(images=images, prompt=prompt, max_tokens=MODEL_MAX_TOKENS_DETECTION)
            secs = self._parse(resp)
            for sec in secs:
                sec["start_page"] = max(sec["start_page"], sp)
                sec["end_page"] = min(sec["end_page"], ep)
            batch_sections.append({"start_page": sp, "end_page": ep, "sections": secs})

        return self._merge(batch_sections, total, document_id)

    # ------------------------------------------------------------------
    # Prompt building (fully config-driven)
    # ------------------------------------------------------------------

    def _build_prompt(self, start_page, end_page, total_pages) -> str:
        template = get_detection_prompt_template()
        return render_prompt(
            template,
            document_type_name=get_document_type_name(),
            total_pages=total_pages,
            start_page=start_page,
            end_page=end_page,
            section_types_csv=", ".join(self.section_definitions.keys()),
            section_definitions_json=json.dumps(self.section_definitions, indent=2),
            heading_alias_rules=build_heading_alias_rules(),
        )

    # ------------------------------------------------------------------
    # Merge (config-driven rules)
    # ------------------------------------------------------------------

    def _merge(self, batch_sections, total_pages, document_id):
        all_secs = []
        for bi in batch_sections:
            all_secs.extend(bi["sections"])
        if not all_secs:
            return None
        all_secs.sort(key=lambda s: s["start_page"])

        # Build lookup of merge rules from config
        merge_rules = {r["section_type"]: r for r in get_merge_rules()}

        merged = []
        current = all_secs[0]
        for nxt in all_secs[1:]:
            adjacent = current["end_page"] >= nxt["start_page"] - 1
            if adjacent and self._should_merge(current, nxt, merge_rules):
                current["end_page"] = max(current["end_page"], nxt["end_page"])
                current["confidence"] = (current.get("confidence", 0.8) + nxt.get("confidence", 0.8)) / 2
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
        logger.info(f"[{document_id}] Merged {len(all_secs)} -> {len(merged)} sections")
        return merged

    @staticmethod
    def _should_merge(cur, nxt, rules):
        """Decide whether to merge two adjacent sections based on config rules."""
        if cur["section_type"] != nxt["section_type"]:
            return False
        rule = rules.get(cur["section_type"])
        if rule:
            if rule.get("merge_same_type_only"):
                return True  # same type is sufficient
            if not rule.get("merge_requires_same_name", True):
                return True
        # Default: merge only if same type AND same name
        return cur["section_name"] == nxt["section_name"]

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def _parse(self, response: str) -> List[Dict]:
        response = response.strip()
        for pfx in ("```json", "```"):
            if response.startswith(pfx):
                response = response[len(pfx):]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        try:
            secs = json.loads(response)
            if not isinstance(secs, list):
                raise ValueError("Not a JSON array")
            return secs
        except json.JSONDecodeError as e:
            logger.error(f"Parse failed: {e}")
            raise
