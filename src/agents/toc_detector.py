"""
Table of Contents Detection Agent.

Scans the first N pages of a PDF for a Table of Contents,
extracts top-level section entries with their printed page numbers,
then resolves those to absolute PDF page indices.

This gives a much more reliable section map than visual batch detection
because the TOC is the document's own authoritative section listing.
"""
import json
import re
from typing import List, Dict, Optional, Tuple

from config.settings import MODEL_MAX_TOKENS_DETECTION, MAX_IMAGES_PER_BATCH
from config.config_loader import (
    get_section_definitions,
    get_document_type_name,
    build_heading_alias_rules,
    render_prompt,
    get_prompt,
)
from src.tools.llm_provider import invoke_multimodal
from src.tools.bedrock_vision import prepare_images_for_bedrock
from src.utils import setup_logger

logger = setup_logger("toc_detector")

# How many pages from the start to scan for a TOC
DEFAULT_TOC_SCAN_PAGES = 15


class TOCDetector:
    """
    Detect and parse a Table of Contents from the early pages of a PDF.

    Strategy:
      1. Send the first N page images to the LLM and ask:
         "Is there a Table of Contents? If yes, extract ONLY top-level entries."
      2. If a TOC is found, return structured entries with printed page numbers.
      3. A separate resolver maps printed page numbers -> absolute PDF indices.
    """

    def __init__(self, toc_scan_pages: int = DEFAULT_TOC_SCAN_PAGES):
        self.toc_scan_pages = toc_scan_pages
        self.section_definitions = get_section_definitions()
        self.toc_start_page: Optional[int] = None  # absolute page where TOC begins
        self.toc_end_page: Optional[int] = None    # absolute page where TOC ends

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_toc(
        self,
        pages_data: List[Dict],
        document_id: str,
    ) -> Optional[List[Dict]]:
        """
        Scan early pages for a TOC and extract top-level entries.

        Returns:
            List of dicts with keys:
                - section_name: str  (exact text from TOC)
                - printed_page: int  (page number as printed in the TOC)
                - toc_level: int     (1 = top-level section)
            or None if no TOC found.
        """
        scan_count = min(self.toc_scan_pages, len(pages_data))
        scan_pages = pages_data[:scan_count]

        logger.info(
            f"[{document_id}] Scanning first {scan_count} pages for Table of Contents"
        )

        # --- Step 1: Find which pages contain a TOC ---
        toc_page_range = self._find_toc_pages(scan_pages, document_id)
        if toc_page_range is None:
            logger.info(f"[{document_id}] No Table of Contents found")
            return None

        toc_start, toc_end = toc_page_range
        self.toc_start_page = toc_start
        self.toc_end_page = toc_end
        logger.info(
            f"[{document_id}] TOC found on absolute pages {toc_start}-{toc_end}"
        )

        # --- Step 2: Extract entries from the TOC pages ---
        toc_pages = pages_data[toc_start - 1 : toc_end]
        entries = self._extract_toc_entries(toc_pages, toc_start, toc_end, document_id)

        if not entries:
            logger.warning(f"[{document_id}] TOC found but no entries extracted")
            return None

        logger.info(
            f"[{document_id}] Extracted {len(entries)} top-level TOC entries"
        )
        for e in entries:
            logger.info(
                f"  -> '{e['section_name']}' (printed page {e['printed_page']})"
            )

        return entries

    # ------------------------------------------------------------------
    # Step 1: Locate which pages contain a TOC
    # ------------------------------------------------------------------

    def _find_toc_pages(
        self, scan_pages: List[Dict], document_id: str
    ) -> Optional[Tuple[int, int]]:
        """
        Ask the LLM which (if any) of the scanned pages contain a
        Table of Contents. Returns (start_abs, end_abs) or None.
        """
        images = prepare_images_for_bedrock(scan_pages)

        # Limit to MAX_IMAGES_PER_BATCH
        if len(images) > MAX_IMAGES_PER_BATCH:
            images = images[:MAX_IMAGES_PER_BATCH]

        scan_end = len(images)

        prompt = (
            "Look at these document pages (absolute page 1 to "
            f"{scan_end}).\n\n"
            "TASK: Determine whether any of these pages contain a "
            "Table of Contents (TOC), Contents page, or similar listing "
            "of document sections with page numbers.\n\n"
            "IMPORTANT: Use ABSOLUTE page position (the first image is page 1, "
            "the second image is page 2, etc.), NOT the printed page numbers.\n\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            '  "has_toc": true or false,\n'
            '  "toc_start_page": <absolute page number where TOC begins>,\n'
            '  "toc_end_page": <absolute page number where TOC ends>\n'
            "}\n\n"
            "If there is no TOC, return:\n"
            '{"has_toc": false, "toc_start_page": null, "toc_end_page": null}\n\n'
            "Return valid JSON only, no markdown or extra text."
        )

        try:
            response = invoke_multimodal(
                images=images,
                prompt=prompt,
                max_tokens=512,
            )
            data = self._parse_json_object(response)

            if not data.get("has_toc", False):
                return None

            start = data.get("toc_start_page")
            end = data.get("toc_end_page")
            if start is None or end is None:
                return None

            return (int(start), int(end))

        except Exception as e:
            logger.warning(f"[{document_id}] TOC page detection failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Step 2: Extract entries from TOC pages
    # ------------------------------------------------------------------

    def _extract_toc_entries(
        self,
        toc_pages: List[Dict],
        toc_start: int,
        toc_end: int,
        document_id: str,
    ) -> List[Dict]:
        """
        Given the actual TOC page images, extract ALL entries with their
        hierarchy level and printed page numbers. Then filter to top-level only.
        """
        images = prepare_images_for_bedrock(toc_pages)

        section_types_csv = ", ".join(self.section_definitions.keys())
        section_defs_json = json.dumps(self.section_definitions, indent=2)
        heading_alias_rules = build_heading_alias_rules()

        prompt = (
            "These images show a Table of Contents from a "
            f"{get_document_type_name()}.\n\n"
            "TASK: Extract ALL entries listed in this Table of Contents.\n\n"
            "For each entry, determine:\n"
            "1. The EXACT heading text as printed in the TOC.\n"
            "2. The PAGE NUMBER (see critical rules below).\n"
            "3. The hierarchy level:\n"
            "   - level 1 = top-level section (main division of the document)\n"
            "   - level 2 = subsection (indented or numbered under a level-1 entry)\n"
            "   - level 3 = sub-subsection\n\n"
            "CRITICAL - DISTINGUISHING PAGE NUMBERS FROM CLAUSE NUMBERS:\n"
            "TOC entries often have TWO numbers. You MUST pick the right one:\n"
            "  - The CLAUSE/SECTION number appears on the LEFT, before or as part\n"
            "    of the heading text (e.g., '12. Assignment', 'Section 5.3').\n"
            "    DO NOT use this as the page number.\n"
            "  - The PAGE NUMBER appears on the RIGHT side, typically after\n"
            "    dot-leaders (......) or whitespace, at the end of the line.\n"
            "    THIS is the number you must extract as printed_page.\n\n"
            "Example:\n"
            "  '12. Assignment .................. 43'\n"
            "     ^^ clause number (IGNORE)       ^^ page number (USE THIS)\n\n"
            "  '5.3 Liability ................... 28'\n"
            "   ^^^ clause number (IGNORE)       ^^ page number (USE THIS)\n\n"
            "If there is only ONE number and it appears at the end of the line,\n"
            "it is the page number. If an entry has no page number at the end,\n"
            "set printed_page to null.\n\n"
            "Page numbers should be MONOTONICALLY INCREASING (or at least\n"
            "non-decreasing) as you go down the TOC. If the numbers you are\n"
            "extracting jump backward (e.g., 41, 42, 12, 12), you are likely\n"
            "extracting clause numbers instead of page numbers. Re-examine.\n\n"
            "HOW TO DETERMINE HIERARCHY LEVEL:\n"
            "- Level 1 entries are typically: flush left, larger/bolder font, "
            "or have single-level numbering (1, 2, 3 or I, II, III or A, B, C).\n"
            "- Level 2+ entries are typically: indented, smaller font, "
            "or have multi-level numbering (1.1, 1.2, 2.1, A.1, etc.).\n"
            "- If entries use dot-leaders (......) to the page number, "
            "use indentation/numbering to determine level.\n\n"
            "EXPECTED SECTION TYPES for level-1 entries:\n"
            f"{section_defs_json}\n\n"
            f"{heading_alias_rules}\n\n"
            "Return ONLY a JSON array (no markdown, no extra text):\n"
            "[\n"
            "  {\n"
            f'    "section_type": "one of: {section_types_csv}, or null if unsure",\n'
            '    "section_name": "EXACT heading text from TOC",\n'
            '    "printed_page": <PAGE number from the RIGHT side of the TOC line>,\n'
            '    "toc_level": <1, 2, or 3>\n'
            "  }\n"
            "]\n\n"
            "RULES:\n"
            "- Extract the EXACT text -- do NOT paraphrase.\n"
            "- Include ALL entries, even if you are unsure of their type.\n"
            "- printed_page MUST be the page number on the RIGHT side of the line,\n"
            "  NOT the clause/section number on the LEFT.\n"
            "- If the page number uses Roman numerals, convert to integer.\n"
            "- Use ONLY ASCII characters.\n"
            "- Return valid JSON only."
        )

        try:
            response = invoke_multimodal(
                images=images,
                prompt=prompt,
                max_tokens=MODEL_MAX_TOKENS_DETECTION,
            )
            all_entries = self._parse_json_array(response)

            # Filter to top-level entries only
            top_level = [e for e in all_entries if e.get("toc_level", 1) == 1]

            # If filtering removed everything, the model may not have set levels
            # correctly -- fall back to returning all
            if not top_level and all_entries:
                logger.warning(
                    f"[{document_id}] No level-1 entries found; "
                    "using all entries as top-level"
                )
                top_level = all_entries

            # Ensure printed_page is int
            for entry in top_level:
                pp = entry.get("printed_page")
                if pp is not None:
                    try:
                        entry["printed_page"] = int(pp)
                    except (ValueError, TypeError):
                        entry["printed_page"] = None

            # Remove entries with no page number
            top_level = [e for e in top_level if e.get("printed_page") is not None]

            return top_level

        except Exception as e:
            logger.error(f"[{document_id}] TOC entry extraction failed: {e}")
            return []

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_object(response: str) -> Dict:
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
        raise ValueError(f"No JSON object found in response: {response[:200]}")

    @staticmethod
    def _parse_json_array(response: str) -> List[Dict]:
        response = response.strip()
        for pfx in ("```json", "```"):
            if response.startswith(pfx):
                response = response[len(pfx):]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        match = re.search(r"\[[\s\S]*\]", response)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"No JSON array found in response: {response[:200]}")