"""
Stage 2: Section Extraction Agent - FULLY GENERIC.
All prompts, section-specific behavior, and document classification
are driven entirely by config.json. No hardcoded section types.
"""
import json
import re
from typing import Dict, List, Tuple, Any, Union

from config.settings import MODEL_MAX_TOKENS_EXTRACTION
from config.config_loader import (
    get_extraction_preamble,
    get_section_extraction_prompt,
    get_extraction_general_rules,
    get_document_classification_config,
    join_prompt,
    render_prompt,
)
from src.tools.llm_provider import invoke_multimodal
from src.tools.bedrock_vision import prepare_images_for_bedrock
from src.utils import setup_logger, StorageManager

logger = setup_logger("section_extractor")


# ============================================================================
# HELPERS
# ============================================================================

def clean_json_response(response: str) -> str:
    response = response.strip()
    for pfx in ("```json", "```"):
        if response.startswith(pfx):
            response = response[len(pfx):]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    response = re.sub(r",\s*([}\]])", r"\1", response)
    for i, ch in enumerate(response):
        if ch in "{[":
            response = response[i:]
            break
    for i in range(len(response) - 1, -1, -1):
        if response[i] in "}]":
            response = response[: i + 1]
            break
    return response


def _check_dict_empty(data: Dict) -> bool:
    if not data:
        return True
    empty = total = 0
    for v in data.values():
        total += 1
        if v is None or v == "" or (isinstance(v, str) and not v.strip()):
            empty += 1
        elif isinstance(v, list) and not v:
            empty += 1
        elif isinstance(v, dict) and _check_dict_empty(v):
            empty += 1
    return empty > total * 0.5 if total else True


def calculate_confidence_score(extracted: Any, section_type: str) -> Tuple[float, List[str]]:
    issues, conf = [], 1.0
    if extracted is None:
        return 0.0, ["No data"]
    if isinstance(extracted, list):
        if not extracted:
            conf -= 0.3; issues.append("Empty array")
        else:
            empty_n = sum(1 for it in extracted if isinstance(it, dict) and _check_dict_empty(it))
            if empty_n:
                conf -= (empty_n / len(extracted)) * 0.2
                issues.append(f"{empty_n}/{len(extracted)} items empty")
    elif isinstance(extracted, dict):
        if not extracted:
            conf -= 0.3; issues.append("Empty object")
        elif _check_dict_empty(extracted):
            conf -= 0.2; issues.append("Some fields empty")
    return max(0.0, min(1.0, conf)), issues


# ============================================================================
# GENERIC SECTION EXTRACTION AGENT
# ============================================================================

class SectionExtractionAgent:
    """
    Fully generic section extractor.
    Reads ALL prompts and section-specific behavior from config.json.
    """

    def __init__(self, section_schema: Dict):
        self.section_schema = section_schema
        self.storage = StorageManager()
        self._cls_config = get_document_classification_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_section(
        self,
        section_pages: List[Dict],
        section_info: Dict,
        next_section_name: str,
        document_id: str,
        image_mapping_data: List[Dict] = None,
        image_prompt_text: str = "",
    ) -> Dict:
        section_type = section_info["section_type"]
        section_name = section_info["section_name"]
        start_page = section_info["start_page"]
        end_page = section_info["end_page"]

        logger.info(f"[{document_id}] Extracting: {section_name} (pages {start_page}-{end_page})")
        response = ""
        try:
            images_b64 = prepare_images_for_bedrock(section_pages)

            # --- Document classification (config-driven) ---
            doc_type_guidance = ""
            if self._needs_classification(section_type):
                doc_type = self._classify_document(images_b64, section_info, document_id)
                doc_type_guidance = self._get_type_guidance(doc_type)

            # --- Build prompt ---
            prompt = self._build_prompt(
                section_info, next_section_name,
                image_prompt_text, doc_type_guidance,
            )

            # --- Invoke LLM ---
            response = invoke_multimodal(images=images_b64, prompt=prompt, max_tokens=MODEL_MAX_TOKENS_EXTRACTION)
            response = response.replace(" -- ", " - ")
            section_json = self._parse(response)

            confidence, issues = calculate_confidence_score(section_json, section_type)
            result = {
                "section_name": section_name,
                "page_range": [start_page, end_page],
                "data": section_json,
                "_metadata": {
                    "section_type": section_type,
                    "confidence": confidence,
                    "quality_issues": issues,
                },
            }
            self.storage.save_section_json(document_id, section_name, result, confidence)
            logger.info(f"[{document_id}] Extracted {section_name} (confidence: {confidence:.2f})")
            return result

        except Exception as e:
            logger.error(f"[{document_id}] Failed: {section_name}: {e}")
            logger.error(f"RESPONSE: {response[:500] if response else '<empty>'}")
            raise

    # ------------------------------------------------------------------
    # Document classification (entirely config-driven)
    # ------------------------------------------------------------------

    def _needs_classification(self, section_type: str) -> bool:
        """Check if this section type requires document classification."""
        if not self._cls_config.get("enabled", False):
            return False
        return section_type in self._cls_config.get("applies_to", [])

    def _classify_document(self, images_b64: List[str], section_info: Dict, document_id: str) -> str:
        """Run document classification using the config-defined prompt."""
        types_cfg = self._cls_config.get("types", {})
        default_type = self._cls_config.get("default_type", "Unknown")

        # Build type hints from config
        type_hints_parts = []
        for type_name, type_cfg in types_cfg.items():
            type_hints_parts.append(f'- "{type_name}": {type_cfg.get("detection_hints", "")}')
        type_hints = "\n".join(type_hints_parts)
        type_names = " / ".join(types_cfg.keys())

        # Render the classification prompt template
        template = join_prompt(self._cls_config.get("prompt_template", []))
        prompt = render_prompt(
            template,
            section_name=section_info["section_name"],
            start_page=section_info["start_page"],
            end_page=section_info["end_page"],
            type_hints=type_hints,
            type_names=type_names,
        )

        response = invoke_multimodal(images=images_b64, prompt=prompt, max_tokens=MODEL_MAX_TOKENS_EXTRACTION)

        # Match response to a configured type by keyword
        response_lower = response.lower().strip()
        for type_name, type_cfg in types_cfg.items():
            keyword = type_cfg.get("match_keyword", type_name.lower())
            if keyword in response_lower:
                logger.info(f"[{document_id}] Classified as: {type_name}")
                return type_name

        logger.info(f"[{document_id}] Classification unclear, using default: {default_type}")
        return default_type

    def _get_type_guidance(self, doc_type: str) -> str:
        """Get the extraction guidance for a classified document type."""
        types_cfg = self._cls_config.get("types", {})
        type_cfg = types_cfg.get(doc_type, {})
        guidance = type_cfg.get("extraction_guidance", [])
        return join_prompt(guidance) if guidance else ""

    # ------------------------------------------------------------------
    # Prompt building (fully config-driven)
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        section_info: Dict,
        next_section_name: str,
        image_info: str,
        doc_type_guidance: str,
    ) -> str:
        section_type = section_info["section_type"]

        # 1. System preamble (from config)
        preamble = render_prompt(
            get_extraction_preamble(),
            schema=json.dumps(self.section_schema, indent=2),
        )

        # 2. Section-specific or default extraction template (from config)
        body_template = get_section_extraction_prompt(section_type)

        body = render_prompt(
            body_template,
            section_name=section_info["section_name"],
            section_type=section_type,
            start_page=section_info["start_page"],
            end_page=section_info["end_page"],
            next_section_name=next_section_name,
            image_info=image_info or "",
            schema=json.dumps(self.section_schema, indent=2),
            doc_type_guidance=doc_type_guidance,
        )

        # 3. General rules (from config)
        rules = get_extraction_general_rules()

        return f"{preamble}\n\n{body}\n\n{rules}"

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def _parse(self, response: str) -> Union[Dict, List]:
        response = clean_json_response(response)
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            raise


# ============================================================================
# Fallback image mapping formatter (used when prompt text not pre-built)
# ============================================================================

def format_image_mapping_fallback(image_mappings: List[Dict]) -> str:
    if not image_mappings:
        return ""
    lines = ["\n\nIMAGES IN THIS SECTION:", "-" * 60]
    for img in image_mappings:
        idx = img.get("sorted_index", img.get("index", 0))
        lines.append(f"[{idx}] Page {img['page']}, {img['grid']} (Y:{img['y_percent']:.0f}%)")
        lines.append(f"    Description: {img['description'][:50]}...")
        lines.append(f"    PATH: {img['path']}")
        lines.append("")
    lines += ["-" * 60, "Match images by PAGE and POSITION, then copy exact PATH."]
    return "\n".join(lines)
