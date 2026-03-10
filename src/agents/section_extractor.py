"""
Stage 2: Section Extraction Agent - FULLY GENERIC.
All prompts, section-specific behavior, and document classification
are driven entirely by config.json. No hardcoded section types.

Supports batch processing: sections with more than BATCH_SIZE pages
are split into batches, extracted independently, and merged.
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

BATCH_SIZE = 2


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


def calculate_confidence_score(
    extracted: Any, section_type: str
) -> Tuple[float, List[str]]:
    issues, conf = [], 1.0
    if extracted is None:
        return 0.0, ["No data"]
    if isinstance(extracted, list):
        if not extracted:
            conf -= 0.3
            issues.append("Empty array")
        else:
            empty_n = sum(
                1 for it in extracted
                if isinstance(it, dict) and _check_dict_empty(it)
            )
            if empty_n:
                conf -= (empty_n / len(extracted)) * 0.2
                issues.append(f"{empty_n}/{len(extracted)} items empty")
    elif isinstance(extracted, dict):
        if not extracted:
            conf -= 0.3
            issues.append("Empty object")
        elif _check_dict_empty(extracted):
            conf -= 0.2
            issues.append("Some fields empty")
    return max(0.0, min(1.0, conf)), issues


def _deep_merge(base: Dict, overlay: Dict) -> Dict:
    """
    Deep-merge overlay into base.
    - Lists are concatenated.
    - Dicts are recursively merged.
    - Scalars: overlay wins if base value is empty/None, otherwise
      base is kept (first-batch priority for top-level fields like
      titles and descriptions).
    """
    merged = dict(base)
    for key, val in overlay.items():
        if key not in merged:
            merged[key] = val
        elif isinstance(merged[key], list) and isinstance(val, list):
            merged[key] = merged[key] + val
        elif isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            # Keep base value if it's non-empty; otherwise take overlay
            if (
                merged[key] is None
                or merged[key] == ""
                or (isinstance(merged[key], str) and not merged[key].strip())
            ):
                merged[key] = val
    return merged


def _build_image_manifest(start_page: int, num_pages: int) -> str:
    """
    Build an explicit image-to-page mapping string so the LLM knows
    exactly which image corresponds to which page and can verify it
    processed all of them.

    Uses 'PDF Page' terminology to disambiguate from printed page
    numbers that may appear in document headers/footers.
    """
    lines = [f"IMAGE-TO-PAGE MAPPING ({num_pages} images):"]
    for i in range(num_pages):
        lines.append(f"  Image {i + 1} = PDF Page {start_page + i}")
    lines.append("")
    lines.append(
        "CRITICAL: The page numbers above refer to PDF page positions "
        "(Image 1 is the FIRST image shown, Image 2 is the SECOND, etc). "
        "The document may have DIFFERENT printed page numbers in its "
        "headers or footers - IGNORE those. Use the image order above."
    )
    lines.append("")
    lines.append(
        f"You MUST extract content from EVERY image (all {num_pages}). "
        f"Process them in order: Image 1, Image 2, ..., Image {num_pages}."
    )
    return "\n".join(lines)


# ============================================================================
# GENERIC SECTION EXTRACTION AGENT
# ============================================================================

class SectionExtractionAgent:
    """
    Fully generic section extractor.
    Reads ALL prompts and section-specific behavior from config.json.

    When a section spans more than BATCH_SIZE pages, the pages are
    split into batches and extracted separately, then merged.
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
        total_pages = len(section_pages)

        logger.info(
            f"[{document_id}] Extracting: {section_name} "
            f"(pages {start_page}-{end_page}, {total_pages} pages)"
        )

        response = ""
        try:
            # --- Document classification (config-driven) ---
            doc_type_guidance = ""
            if self._needs_classification(section_type):
                # Classify using first batch of pages
                cls_pages = section_pages[:BATCH_SIZE]
                images_cls = prepare_images_for_bedrock(cls_pages)
                doc_type = self._classify_document(
                    images_cls, section_info, document_id
                )
                doc_type_guidance = self._get_type_guidance(doc_type)

            # --- Single batch or multi-batch ---
            if total_pages <= BATCH_SIZE:
                section_json = self._extract_single(
                    section_pages, section_info,
                    next_section_name, image_prompt_text,
                    doc_type_guidance,
                )
            else:
                section_json = self._extract_batched(
                    section_pages, section_info,
                    next_section_name, image_prompt_text,
                    doc_type_guidance, document_id,
                )

            confidence, issues = calculate_confidence_score(
                section_json, section_type
            )
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
            self.storage.save_section_json(
                document_id, section_name, result, confidence
            )
            logger.info(
                f"[{document_id}] Extracted {section_name} "
                f"(confidence: {confidence:.2f})"
            )
            return result

        except Exception as e:
            logger.error(
                f"[{document_id}] Failed: {section_name}: {e}"
            )
            logger.error(
                f"RESPONSE: {response[:500] if response else '<empty>'}"
            )
            raise

    # ------------------------------------------------------------------
    # Single-batch extraction (≤ BATCH_SIZE pages)
    # ------------------------------------------------------------------

    def _extract_single(
        self,
        section_pages: List[Dict],
        section_info: Dict,
        next_section_name: str,
        image_prompt_text: str,
        doc_type_guidance: str,
    ) -> Union[Dict, List]:
        """Extract a section in one LLM call."""
        images_b64 = prepare_images_for_bedrock(section_pages)

        prompt = self._build_prompt(
            section_info, next_section_name,
            image_prompt_text, doc_type_guidance,
        )

        # Add image manifest so the LLM knows exactly which image
        # is which page and can verify completeness
        manifest = _build_image_manifest(
            section_info["start_page"], len(section_pages)
        )
        prompt = manifest + "\n\n" + prompt

        response = invoke_multimodal(
            images=images_b64, prompt=prompt,
            max_tokens=MODEL_MAX_TOKENS_EXTRACTION,
        )
        response = response.replace(" -- ", " - ")
        return self._parse(response)

    # ------------------------------------------------------------------
    # Multi-batch extraction (> BATCH_SIZE pages)
    # ------------------------------------------------------------------

    def _extract_batched(
        self,
        section_pages: List[Dict],
        section_info: Dict,
        next_section_name: str,
        image_prompt_text: str,
        doc_type_guidance: str,
        document_id: str,
    ) -> Union[Dict, List]:
        """
        Split section into batches of BATCH_SIZE pages, extract each
        batch, and merge results.

        Each batch receives context about:
          - Which batch it is (N of M)
          - Which absolute pages it covers
          - An explicit image-to-page manifest
          - The expected output schema (same for every batch)

        Merge strategy:
          - If results are lists → concatenate all lists
          - If results are dicts → deep-merge (lists inside concatenated,
            first-batch scalars take priority)
        """
        total_pages = len(section_pages)
        start_page = section_info["start_page"]

        # Build batches
        batches = []
        for i in range(0, total_pages, BATCH_SIZE):
            batch_pages = section_pages[i : i + BATCH_SIZE]
            batch_start = start_page + i
            batch_end = batch_start + len(batch_pages) - 1
            batches.append((batch_pages, batch_start, batch_end))

        n_batches = len(batches)
        logger.info(
            f"[{document_id}] Splitting '{section_info['section_name']}' "
            f"into {n_batches} batches of up to {BATCH_SIZE} pages"
        )

        batch_results: List[Union[Dict, List]] = []

        for batch_idx, (batch_pages, b_start, b_end) in enumerate(
            batches, 1
        ):
            logger.info(
                f"[{document_id}]   Batch {batch_idx}/{n_batches}: "
                f"pages {b_start}-{b_end} "
                f"({len(batch_pages)} pages)"
            )

            # Build a batch-specific section_info with adjusted pages
            batch_info = dict(section_info)
            batch_info["start_page"] = b_start
            batch_info["end_page"] = b_end

            # Only the last batch mentions the next section name.
            # For non-last batches, use "END OF SHOWN PAGES" so the
            # prompt template renders a clear boundary instead of an
            # empty string (which confuses the LLM).
            batch_next = (
                next_section_name
                if batch_idx == n_batches
                else "END OF SHOWN PAGES"
            )

            # Build the base prompt
            prompt = self._build_prompt(
                batch_info, batch_next,
                image_prompt_text, doc_type_guidance,
            )

            # Add image manifest for this batch
            manifest = _build_image_manifest(b_start, len(batch_pages))

            # Add batch context — ONLY reference this batch's pages.
            # Do NOT mention the full section page range, as this causes
            # the LLM to anchor to pages beyond the current batch and
            # either hallucinate content or mislabel which image is which page.
            if batch_idx == 1:
                heading_instruction = (
                    "This is the FIRST batch — extract the section heading "
                    "normally from the top of the content."
                )
            else:
                heading_instruction = (
                    "This is a CONTINUATION batch (not the first). "
                    "If there is not a section heading visible then"
                    "Set 'heading' to '' (empty string) and 'heading_level' to ''. "
                    "Put ALL visible text into the 'body' array. "
                    "Do NOT pick up text from the top of the page as a heading — "
                    "it is body content continuing from the previous section."
                )

            batch_context = (
                f"\n\nBATCH PROCESSING CONTEXT:\n"
                f"You are shown {len(batch_pages)} page images "
                f"(pages {b_start}-{b_end}).\n"
                f"This is batch {batch_idx} of {n_batches} for this section.\n"
                f"{heading_instruction}\n"
                f"Extract ONLY the content visible in these {len(batch_pages)} images. "
                f"Do NOT invent or assume content from pages you cannot see.\n"
                f"Use the SAME output schema -- your output will be "
                f"merged with other batches.\n"
            )

            prompt = manifest + "\n\n" + prompt + batch_context

            images_b64 = prepare_images_for_bedrock(batch_pages)

            try:
                response = invoke_multimodal(
                    images=images_b64, prompt=prompt,
                    max_tokens=MODEL_MAX_TOKENS_EXTRACTION,
                )
                response = response.replace(" -- ", " - ")
                batch_json = self._parse(response)
                self.storage.save_batch_json(f"{section_info["section_name"]}_{batch_idx}", batch_json)
                batch_results.append(batch_json)
                logger.info(
                    f"[{document_id}]   Batch {batch_idx} extracted OK"
                )
            except Exception as e:
                logger.error(
                    f"[{document_id}]   Batch {batch_idx} failed: {e}"
                )
                # Continue with other batches rather than failing entirely
                continue

        if not batch_results:
            raise RuntimeError(
                f"All {n_batches} batches failed for "
                f"'{section_info['section_name']}'"
            )

        # Merge batch results
        return self._merge_batch_results(batch_results, document_id)

    def _merge_batch_results(
        self,
        results: List[Union[Dict, List]],
        document_id: str,
    ) -> Union[Dict, List]:
        """
        Merge results from multiple batches.

        - All lists → concatenate
        - All dicts → deep-merge (lists concatenated, first scalar wins)
        - Mixed → wrap dicts as single-element lists, then concatenate
        """
        if not results:
            return {}

        # Check types
        all_lists = all(isinstance(r, list) for r in results)
        all_dicts = all(isinstance(r, dict) for r in results)

        if all_lists:
            merged = []
            for r in results:
                merged.extend(r)
            logger.info(
                f"[{document_id}] Merged {len(results)} batches "
                f"(list concat): {len(merged)} items"
            )
            return merged

        if all_dicts:
            merged = {}
            for r in results:
                merged = _deep_merge(merged, r)
            logger.info(
                f"[{document_id}] Merged {len(results)} batches "
                f"(dict deep-merge)"
            )
            return merged

        # Mixed: normalize to lists and concatenate
        merged = []
        for r in results:
            if isinstance(r, list):
                merged.extend(r)
            elif isinstance(r, dict):
                merged.append(r)
            else:
                merged.append(r)
        logger.info(
            f"[{document_id}] Merged {len(results)} batches "
            f"(mixed → list): {len(merged)} items"
        )
        return merged

    # ------------------------------------------------------------------
    # Document classification (entirely config-driven)
    # ------------------------------------------------------------------

    def _needs_classification(self, section_type: str) -> bool:
        """Check if this section type requires document classification."""
        if not self._cls_config.get("enabled", False):
            return False
        return section_type in self._cls_config.get("applies_to", [])

    def _classify_document(
        self, images_b64: List[str], section_info: Dict,
        document_id: str,
    ) -> str:
        """Run document classification using the config-defined prompt."""
        types_cfg = self._cls_config.get("types", {})
        default_type = self._cls_config.get("default_type", "Unknown")

        # Build type hints from config
        type_hints_parts = []
        for type_name, type_cfg in types_cfg.items():
            type_hints_parts.append(
                f'- "{type_name}": '
                f'{type_cfg.get("detection_hints", "")}'
            )
        type_hints = "\n".join(type_hints_parts)
        type_names = " / ".join(types_cfg.keys())

        # Render the classification prompt template
        template = join_prompt(
            self._cls_config.get("prompt_template", [])
        )
        prompt = render_prompt(
            template,
            section_name=section_info["section_name"],
            start_page=section_info["start_page"],
            end_page=section_info["end_page"],
            type_hints=type_hints,
            type_names=type_names,
        )

        response = invoke_multimodal(
            images=images_b64, prompt=prompt,
            max_tokens=MODEL_MAX_TOKENS_EXTRACTION,
        )

        # Match response to a configured type by keyword
        response_lower = response.lower().strip()
        for type_name, type_cfg in types_cfg.items():
            keyword = type_cfg.get("match_keyword", type_name.lower())
            if keyword in response_lower:
                logger.info(
                    f"[{document_id}] Classified as: {type_name}"
                )
                return type_name

        logger.info(
            f"[{document_id}] Classification unclear, "
            f"using default: {default_type}"
        )
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
        lines.append(
            f"[{idx}] Page {img['page']}, {img['grid']} "
            f"(Y:{img['y_percent']:.0f}%)"
        )
        lines.append(f"    Description: {img['description'][:50]}...")
        lines.append(f"    PATH: {img['path']}")
        lines.append("")
    lines += [
        "-" * 60,
        "Match images by PAGE and POSITION, then copy exact PATH.",
    ]
    return "\n".join(lines)