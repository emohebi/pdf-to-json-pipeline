"""
Stage 1: Section Detection Agent - TOC-First Strategy.

Detection flow:
  1. Scan early pages for a Table of Contents (TOC).
  2. If TOC found:
     a. Extract top-level section entries from the TOC.
     b. Map printed page numbers to absolute PDF page indices.
     c. Add front_matter for pages before the first section.
     d. Optionally add back_matter for trailing pages.
  3. If NO TOC found:
     a. Fall back to batch-based visual detection (original approach).
  4. Merge and validate the final section list.

All prompts, merge rules, and heading aliases come from config.json.
"""
import json
from typing import List, Dict, Optional

from config.settings import MODEL_MAX_TOKENS_DETECTION, MAX_IMAGES_PER_BATCH
from config.config_loader import (
    get_section_definitions,
    get_detection_prompt_template,
    get_document_type_name,
    build_heading_alias_rules,
    render_prompt,
    get_merge_rules,
)
from src.agents.toc_detector import TOCDetector
from src.agents.page_number_resolver import PageNumberResolver
from src.tools.llm_provider import invoke_multimodal
from src.tools.bedrock_vision import prepare_images_for_bedrock
from src.utils import setup_logger, StorageManager

logger = setup_logger("section_detector")


class SectionDetectionAgent:
    """
    Identify logical sections in PDF documents.

    Uses a TOC-first strategy: if a Table of Contents is found, it is
    treated as the authoritative source for section boundaries.
    Falls back to batch-based visual detection when no TOC is present.
    """

    def __init__(self):
        self.section_definitions = get_section_definitions()
        self.storage = StorageManager()
        self.max_per_call = MAX_IMAGES_PER_BATCH

    # ==================================================================
    # Public API
    # ==================================================================

    def detect_sections(
        self, pages_data: List[Dict], document_id: str
    ) -> Optional[List[Dict]]:
        total = len(pages_data)
        logger.info(f"[{document_id}] Detecting sections in {total} pages")

        try:
            # --- Strategy 1: TOC-based detection ---
            sections = self._try_toc_detection(pages_data, document_id)

            if sections:
                logger.info(
                    f"[{document_id}] TOC-based detection succeeded: "
                    f"{len(sections)} sections"
                )
            else:
                # --- Strategy 2: Fallback to visual batch detection ---
                logger.info(
                    f"[{document_id}] Falling back to visual batch detection"
                )
                sections = self._detect_multi_batch(pages_data, document_id)

            if sections:
                # Ensure all pages are covered (fill gaps)
                sections = self._ensure_full_coverage(sections, total, document_id)
                self.storage.save_detection_result(document_id, sections)
                logger.info(
                    f"[{document_id}] Final detection: {len(sections)} sections"
                )

            return sections

        except Exception as e:
            logger.error(f"[{document_id}] Detection failed: {e}")
            return None

    # ==================================================================
    # Strategy 1: TOC-based detection
    # ==================================================================

    def _try_toc_detection(
        self, pages_data: List[Dict], document_id: str
    ) -> Optional[List[Dict]]:
        """
        Attempt to detect sections using the Table of Contents.
        Returns a list of section dicts or None if TOC not found/usable.
        """
        toc_detector = TOCDetector()
        toc_entries = toc_detector.detect_toc(pages_data, document_id)

        if not toc_entries:
            return None

        # Build page number mapping, anchored to the TOC location
        resolver = PageNumberResolver(
            pages_data, toc_abs_start=toc_detector.toc_start_page
        )
        stats = resolver.get_mapping_stats()
        logger.info(
            f"[{document_id}] Page mapping: {stats['mapped_pages']}/"
            f"{stats['total_pages']} pages resolved "
            f"({stats['coverage_pct']}%), offset={stats['offset']}"
        )

        # Resolve TOC entries to absolute page ranges
        total = len(pages_data)
        sections = resolver.resolve_toc_entries(toc_entries, total)

        if not sections:
            logger.warning(
                f"[{document_id}] TOC entries could not be resolved to pages"
            )
            return None

        # ---- Verify boundaries by checking actual page images ----
        sections = self._verify_section_boundaries(
            sections, pages_data, document_id
        )

        # Add front_matter for pages before the first section
        first_start = sections[0]["start_page"]
        if first_start > 1:
            sections.insert(0, {
                "section_type": "front_matter",
                "section_name": "Front Matter",
                "start_page": 1,
                "end_page": first_start - 1,
                "confidence": 0.95,
                "_source": "toc_inferred",
            })

        # Validate: no section should exceed total pages
        for sec in sections:
            sec["end_page"] = min(sec["end_page"], total)
            sec["start_page"] = max(sec["start_page"], 1)

        # Log the detected sections
        for sec in sections:
            logger.info(
                f"  [{sec['section_type']}] '{sec['section_name']}' "
                f"pages {sec['start_page']}-{sec['end_page']}"
            )

        return sections

    # ==================================================================
    # Boundary verification (end-page only)
    # ==================================================================

    def _verify_section_boundaries(
        self,
        sections: List[Dict],
        pages_data: List[Dict],
        document_id: str,
    ) -> List[Dict]:
        """
        Verify and correct the END PAGE of each section.

        The TOC gives correct START pages. The problem is with end pages:
        the naive formula `end = next_section_start - 1` can be wrong
        because the current section's content may continue onto the page
        where the next section's heading appears.

        Example:
          TOC: "Definitions" -> page 3, "Services" -> page 7
          Naive: Definitions = 3-6, Services = 7+
          Reality: Definitions content continues onto page 7;
                   the "Services" heading appears partway through page 7.
          Correct: Definitions = 3-7, Services = 7+  (page 7 shared)

        Approach:
          For each boundary, send the boundary page (next_start) to
          the LLM and ask: "Does this page contain content from the
          PREVIOUS section before the next section's heading?"
          If yes -> current section's end_page = next_start (shared page).
          If no  -> current section's end_page = next_start - 1 (no overlap).
        """
        if len(sections) < 2:
            return sections

        logger.info(
            f"[{document_id}] Verifying {len(sections) - 1} "
            "section boundary end-pages"
        )

        for i in range(len(sections) - 1):
            current_sec = sections[i]
            next_sec = sections[i + 1]

            boundary_page = next_sec["start_page"]  # TOC start (authoritative)
            current_name = current_sec["section_name"]
            next_name = next_sec["section_name"]

            # Nothing to verify if sections are already adjacent with no gap
            # (current end == boundary - 1 is the naive default we want to check)
            if boundary_page < 1 or boundary_page > len(pages_data):
                continue

            logger.info(
                f"  Boundary {i + 1}: '{current_name}' ends at ? | "
                f"'{next_name}' starts at page {boundary_page}"
            )

            # Send the single boundary page to check for shared content
            extends = self._check_section_continues_on_page(
                pages_data[boundary_page - 1],
                boundary_page,
                current_name,
                next_name,
                document_id,
            )

            if extends:
                current_sec["end_page"] = boundary_page
                logger.info(
                    f"    -> '{current_name}' extends onto page {boundary_page} "
                    f"(shared with '{next_name}')"
                )
            else:
                current_sec["end_page"] = boundary_page - 1
                logger.info(
                    f"    -> '{current_name}' ends at page {boundary_page - 1} "
                    f"(clean break)"
                )

        # Safety: ensure end >= start for all sections
        for sec in sections:
            if sec["end_page"] < sec["start_page"]:
                sec["end_page"] = sec["start_page"]

        return sections

    def _check_section_continues_on_page(
        self,
        page_data: Dict,
        abs_page: int,
        current_section_name: str,
        next_section_name: str,
        document_id: str,
    ) -> bool:
        """
        Check whether the current section's content continues onto
        this page BEFORE the next section's heading appears.

        Returns True if the page has content from the current section
        (i.e., the next section heading is NOT at the very top of the page,
        meaning there is preceding content belonging to the current section).
        Returns False if the next section heading is at the top of the page
        or the page belongs entirely to the next section.
        """
        images = prepare_images_for_bedrock([page_data])

        prompt = (
            f"I am showing you a single document page (absolute page {abs_page}).\n\n"
            "CONTEXT:\n"
            f"- The section '{current_section_name}' is expected to end "
            f"around this page.\n"
            f"- The section '{next_section_name}' is expected to start "
            f"on this page.\n\n"
            "TASK: Look at this page and determine:\n"
            f"Does content belonging to '{current_section_name}' appear on "
            f"this page BEFORE the heading '{next_section_name}'?\n\n"
            "RULES:\n"
            f"- If the heading '{next_section_name}' is at the VERY TOP of "
            "the page (first element, no prior content), answer false — the "
            "previous section does NOT continue onto this page.\n"
            f"- If there is ANY content (text, tables, figures) from "
            f"'{current_section_name}' ABOVE or BEFORE the heading "
            f"'{next_section_name}', answer true — the previous section "
            "DOES continue onto this page.\n"
            f"- If the heading '{next_section_name}' does not appear on this "
            "page at all, answer true — the previous section still continues.\n\n"
            "Return ONLY a JSON object (no markdown, no extra text):\n"
            "{\n"
            '  "continues": true or false\n'
            "}"
        )

        try:
            response = invoke_multimodal(
                images=images,
                prompt=prompt,
                max_tokens=128,
            )
            data = self._parse_json_obj(response)
            return data.get("continues", False)

        except Exception as e:
            logger.warning(
                f"[{document_id}] Boundary check failed for page {abs_page}: {e}"
            )
            # Default: assume no overlap (safe fallback)
            return False

    @staticmethod
    def _parse_json_obj(response: str) -> Dict:
        """Parse a JSON object from an LLM response."""
        import re
        response = response.strip()
        for pfx in ("```json", "```"):
            if response.startswith(pfx):
                response = response[len(pfx):]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"No JSON object in response: {response[:200]}")

    # ==================================================================
    # Strategy 2: Visual batch detection (fallback)
    # ==================================================================

    def _detect_multi_batch(self, pages_data, document_id):
        """Original batch-based detection using visual analysis."""
        total = len(pages_data)
        n_batches = (total + self.max_per_call - 1) // self.max_per_call
        logger.info(f"[{document_id}] {total} pages -> {n_batches} batches")

        batch_sections = []
        for b in range(n_batches):
            s = b * self.max_per_call
            e = min(s + self.max_per_call, total)
            batch = pages_data[s:e]
            sp, ep = s + 1, e  # Absolute page numbers (1-based)
            logger.info(
                f"[{document_id}] Batch {b+1}/{n_batches}: "
                f"absolute pages {sp}-{ep}"
            )

            images = prepare_images_for_bedrock(batch)
            prompt = self._build_prompt(sp, ep, total)
            resp = invoke_multimodal(
                images=images,
                prompt=prompt,
                max_tokens=MODEL_MAX_TOKENS_DETECTION,
            )
            secs = self._parse(resp)

            # Clamp returned pages to valid batch range
            for sec in secs:
                sec["start_page"] = max(sec.get("start_page", sp), sp)
                sec["end_page"] = min(sec.get("end_page", ep), ep)

            batch_sections.append(
                {"start_page": sp, "end_page": ep, "sections": secs}
            )

        return self._merge_batches(batch_sections, total, document_id)

    def _build_prompt(self, start_page, end_page, total_pages) -> str:
        """Build detection prompt from config template."""
        template = get_detection_prompt_template()
        return render_prompt(
            template,
            document_type_name=get_document_type_name(),
            total_pages=total_pages,
            start_page=start_page,
            end_page=end_page,
            section_types_csv=", ".join(self.section_definitions.keys()),
            section_definitions_json=json.dumps(
                self.section_definitions, indent=2
            ),
            heading_alias_rules=build_heading_alias_rules(),
        )

    # ==================================================================
    # Merge logic
    # ==================================================================

    def _merge_batches(self, batch_sections, total_pages, document_id):
        """Merge sections from multiple visual detection batches."""
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
                current["end_page"] = max(
                    current["end_page"], nxt["end_page"]
                )
                current["confidence"] = (
                    current.get("confidence", 0.8)
                    + nxt.get("confidence", 0.8)
                ) / 2
            else:
                merged.append(current)
                current = nxt
        merged.append(current)

        logger.info(
            f"[{document_id}] Merged {len(all_secs)} -> {len(merged)} sections"
        )
        return merged

    @staticmethod
    def _should_merge(cur, nxt, rules):
        """Decide whether to merge two adjacent sections based on config."""
        if cur["section_type"] != nxt["section_type"]:
            return False
        rule = rules.get(cur["section_type"])
        if rule:
            if rule.get("merge_same_type_only"):
                return True
            if not rule.get("merge_requires_same_name", True):
                return True
        return cur["section_name"] == nxt["section_name"]

    # ==================================================================
    # Coverage validation
    # ==================================================================

    def _ensure_full_coverage(
        self, sections: List[Dict], total_pages: int, document_id: str
    ) -> List[Dict]:
        """
        Ensure every page from 1 to total_pages belongs to exactly one section.
        Fill gaps with 'unhandled_content' sections.
        """
        if not sections:
            return sections

        sections.sort(key=lambda s: s["start_page"])
        filled = []
        expected_start = 1

        for sec in sections:
            # Fill gap before this section
            if sec["start_page"] > expected_start:
                gap = {
                    "section_type": "unhandled_content",
                    "section_name": "Unhandled Content",
                    "start_page": expected_start,
                    "end_page": sec["start_page"] - 1,
                    "confidence": 0.5,
                    "_source": "gap_fill",
                }
                filled.append(gap)
                logger.info(
                    f"[{document_id}] Filled gap: pages "
                    f"{gap['start_page']}-{gap['end_page']}"
                )

            filled.append(sec)
            expected_start = sec["end_page"] + 1

        # Fill trailing gap
        if expected_start <= total_pages:
            # Check if the last section looks like it should extend
            last = filled[-1]
            if last["section_type"] in ("back_matter", "unhandled_content"):
                last["end_page"] = total_pages
            else:
                gap = {
                    "section_type": "back_matter",
                    "section_name": "Back Matter",
                    "start_page": expected_start,
                    "end_page": total_pages,
                    "confidence": 0.7,
                    "_source": "trailing_fill",
                }
                filled.append(gap)
                logger.info(
                    f"[{document_id}] Added trailing back_matter: pages "
                    f"{gap['start_page']}-{gap['end_page']}"
                )

        return filled

    # ==================================================================
    # Parse
    # ==================================================================

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