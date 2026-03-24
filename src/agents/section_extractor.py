"""
Stage 2: Section Extraction Agent - FULLY GENERIC.
All prompts, section-specific behavior, and document classification
are driven entirely by config.json. No hardcoded section types.

Uses ordered content-block schema: each section has a single 'content'
array of typed blocks (paragraph, table, subsection) in reading order.
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
from src.agents.batch_merge import (
    merge_batch_results,
    normalize_batch_result,
    get_trailing_context,
)

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
    if isinstance(extracted, dict):
        content = extracted.get("content", [])
        if not content:
            conf -= 0.3
            issues.append("Empty content array")
    elif isinstance(extracted, list):
        if not extracted:
            conf -= 0.3
            issues.append("Empty list")
    return max(0.0, min(1.0, conf)), issues


def _build_image_manifest(start_page: int, num_pages: int) -> str:
    lines = [f"IMAGE-TO-PAGE MAPPING ({num_pages} images):"]
    for i in range(num_pages):
        lines.append(f"  Image {i + 1} = PDF Page {start_page + i}")
    lines.append("")
    lines.append(
        "CRITICAL: The page numbers above refer to PDF page positions. "
        "IGNORE any printed page numbers in document headers/footers."
    )
    lines.append(
        f"You MUST extract content from ALL {num_pages} images in order."
    )
    return "\n".join(lines)


# ============================================================================
# SECTION EXTRACTION AGENT
# ============================================================================

class SectionExtractionAgent:

    def __init__(self, section_schema: Dict):
        self.section_schema = section_schema
        self.storage = StorageManager()
        self._cls_config = get_document_classification_config()

    def extract_section(
        self, section_pages: List[Dict], section_info: Dict,
        next_section_name: str, document_id: str,
        image_mapping_data: List[Dict] = None, image_prompt_text: str = "",
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
            doc_type_guidance = ""
            if self._needs_classification(section_type):
                cls_pages = section_pages[:BATCH_SIZE]
                images_cls = prepare_images_for_bedrock(cls_pages)
                doc_type = self._classify_document(images_cls, section_info, document_id)
                doc_type_guidance = self._get_type_guidance(doc_type)

            if total_pages <= BATCH_SIZE:
                section_json = self._extract_single(
                    section_pages, section_info, next_section_name,
                    image_prompt_text, doc_type_guidance,
                )
            else:
                section_json = self._extract_batched(
                    section_pages, section_info, next_section_name,
                    image_prompt_text, doc_type_guidance, document_id,
                )

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

    def _extract_single(self, section_pages, section_info, next_section_name,
                         image_prompt_text, doc_type_guidance):
        images_b64 = prepare_images_for_bedrock(section_pages)
        prompt = self._build_prompt(section_info, next_section_name, image_prompt_text, doc_type_guidance)
        manifest = _build_image_manifest(section_info["start_page"], len(section_pages))
        prompt = manifest + "\n\n" + prompt
        response = invoke_multimodal(images=images_b64, prompt=prompt, max_tokens=MODEL_MAX_TOKENS_EXTRACTION)
        response = response.replace(" -- ", " - ")
        return self._parse(response)

    # ------------------------------------------------------------------

    def _extract_batched(self, section_pages, section_info, next_section_name,
                          image_prompt_text, doc_type_guidance, document_id):
        total_pages = len(section_pages)
        start_page = section_info["start_page"]

        batches = []
        for i in range(0, total_pages, BATCH_SIZE):
            bp = section_pages[i:i + BATCH_SIZE]
            bs = start_page + i
            be = bs + len(bp) - 1
            batches.append((bp, bs, be))

        n_batches = len(batches)
        logger.info(f"[{document_id}] Splitting '{section_info['section_name']}' into {n_batches} batches")

        batch_results = []
        trailing_contexts = []
        prev_ctx: Dict[str, Any] = {}

        for batch_idx, (batch_pages, b_start, b_end) in enumerate(batches, 1):
            logger.info(f"[{document_id}]   Batch {batch_idx}/{n_batches}: pages {b_start}-{b_end}")

            batch_info = dict(section_info)
            batch_info["start_page"] = b_start
            batch_info["end_page"] = b_end
            batch_next = next_section_name if batch_idx == n_batches else "END OF SHOWN PAGES"

            prompt = self._build_prompt(batch_info, batch_next, image_prompt_text, doc_type_guidance)
            manifest = _build_image_manifest(b_start, len(batch_pages))

            if batch_idx == 1:
                hint = "This is the FIRST batch -- extract the section heading normally."
            else:
                tsub = prev_ctx.get("trailing_subsection", "")
                if tsub:
                    hint = (
                        "This is a CONTINUATION batch. "
                        "Set top-level 'heading' to '' and 'heading_level' to ''.\n\n"
                        f"CRITICAL: The previous batch ended inside subsection \"{tsub}\".\n"
                        f"If text at the top continues that topic, place it inside a "
                        f"subsection block with heading=\"{tsub}\" in the content array.\n"
                        f"Do NOT put continuation content as top-level paragraph blocks "
                        f"before the subsection block."
                    )
                else:
                    hint = (
                        "This is a CONTINUATION batch. "
                        "Set 'heading' to '' and 'heading_level' to ''. "
                        "Add all content as blocks in the content array."
                    )

            batch_context = (
                f"\n\nBATCH PROCESSING CONTEXT:\n"
                f"Pages {b_start}-{b_end}, batch {batch_idx}/{n_batches}.\n"
                f"{hint}\n"
                f"Extract ONLY content visible in these images.\n"
                f"Use the SAME output schema.\n"
            )
            prompt = manifest + "\n\n" + prompt + batch_context
            images_b64 = prepare_images_for_bedrock(batch_pages)

            try:
                response = invoke_multimodal(images=images_b64, prompt=prompt, max_tokens=MODEL_MAX_TOKENS_EXTRACTION)
                response = response.replace(" -- ", " - ")
                batch_json = self._parse(response)
                batch_json = normalize_batch_result(batch_json)
                self.storage.save_batch_json(f"{section_info['section_name']}_{batch_idx}", batch_json)
                batch_results.append(batch_json)
                prev_ctx = get_trailing_context(batch_json)
                trailing_contexts.append(prev_ctx)
                logger.info(f"[{document_id}]   Batch {batch_idx} OK")
                if prev_ctx.get("trailing_subsection"):
                    logger.info(f"[{document_id}]   Trailing: \"{prev_ctx['trailing_subsection']}\"")
            except Exception as e:
                logger.error(f"[{document_id}]   Batch {batch_idx} failed: {e}")
                prev_ctx = {}
                trailing_contexts.append({})

        if not batch_results:
            raise RuntimeError(f"All batches failed for '{section_info['section_name']}'")

        return merge_batch_results(batch_results, trailing_contexts, document_id)

    # ------------------------------------------------------------------

    def _needs_classification(self, section_type):
        if not self._cls_config.get("enabled", False):
            return False
        return section_type in self._cls_config.get("applies_to", [])

    def _classify_document(self, images_b64, section_info, document_id):
        types_cfg = self._cls_config.get("types", {})
        default_type = self._cls_config.get("default_type", "Unknown")
        type_hints = "\n".join(
            f'- "{n}": {c.get("detection_hints", "")}' for n, c in types_cfg.items()
        )
        template = join_prompt(self._cls_config.get("prompt_template", []))
        prompt = render_prompt(template, section_name=section_info["section_name"],
                               start_page=section_info["start_page"], end_page=section_info["end_page"],
                               type_hints=type_hints, type_names=" / ".join(types_cfg.keys()))
        response = invoke_multimodal(images=images_b64, prompt=prompt, max_tokens=MODEL_MAX_TOKENS_EXTRACTION)
        rl = response.lower().strip()
        for name, cfg in types_cfg.items():
            if cfg.get("match_keyword", name.lower()) in rl:
                return name
        return default_type

    def _get_type_guidance(self, doc_type):
        cfg = self._cls_config.get("types", {}).get(doc_type, {})
        g = cfg.get("extraction_guidance", [])
        return join_prompt(g) if g else ""

    # ------------------------------------------------------------------

    def _build_prompt(self, section_info, next_section_name, image_info, doc_type_guidance):
        section_type = section_info["section_type"]
        preamble = render_prompt(get_extraction_preamble(), schema=json.dumps(self.section_schema, indent=2))
        body_template = get_section_extraction_prompt(section_type)
        body = render_prompt(body_template, section_name=section_info["section_name"],
                             section_type=section_type, start_page=section_info["start_page"],
                             end_page=section_info["end_page"], next_section_name=next_section_name,
                             image_info=image_info or "", schema=json.dumps(self.section_schema, indent=2),
                             doc_type_guidance=doc_type_guidance)
        rules = get_extraction_general_rules()
        return f"{preamble}\n\n{body}\n\n{rules}"

    # ------------------------------------------------------------------

    def _parse(self, response: str) -> Union[Dict, List]:
        response = clean_json_response(response)
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            raise


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
