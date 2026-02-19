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

from config.settings import MAX_IMAGES_PER_BATCH
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
        scan_from: int = 1,
    ) -> Optional[List[Dict]]:
        """
        Scan pages starting from `scan_from` for a TOC and extract
        top-level entries.

        Args:
            pages_data: ALL pages in the document.
            document_id: Identifier for logging.
            scan_from: Absolute page number to start scanning from (1-based).
                       Pages before this are ignored.

        Returns:
            List of dicts with keys:
                - section_name: str  (exact text from TOC)
                - printed_page: int  (page number as printed in the TOC)
                - toc_level: int     (1 = top-level section)
            or None if no TOC found.
        """
        total = len(pages_data)
        scan_end_abs = min(scan_from + self.toc_scan_pages - 1, total)
        scan_pages = pages_data[scan_from - 1 : scan_end_abs]

        if not scan_pages:
            return None

        logger.info(
            f"[{document_id}] Scanning pages {scan_from}-{scan_end_abs} "
            f"for Table of Contents"
        )

        # --- Step 1: Find which pages contain a TOC ---
        toc_page_range = self._find_toc_pages(
            scan_pages, document_id, abs_offset=scan_from
        )
        if toc_page_range is None:
            logger.info(
                f"[{document_id}] No Table of Contents found "
                f"in pages {scan_from}-{scan_end_abs}"
            )
            return None

        toc_start, toc_end = toc_page_range
        self.toc_start_page = toc_start

        # --- Step 1b: Verify/extend TOC end page ---
        # The LLM often underestimates the TOC end. Scan forward
        # page by page to find the true end.
        toc_end = self._verify_toc_end(
            pages_data, toc_start, toc_end, document_id
        )
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
        self, scan_pages: List[Dict], document_id: str,
        abs_offset: int = 1,
    ) -> Optional[Tuple[int, int]]:
        """
        Ask the LLM which (if any) of the scanned pages contain a
        Table of Contents. Returns (start_abs, end_abs) or None.

        Uses image indices (1..N) and translates to absolute pages.
        """
        images = prepare_images_for_bedrock(scan_pages)

        # Limit to MAX_IMAGES_PER_BATCH
        if len(images) > MAX_IMAGES_PER_BATCH:
            images = images[:MAX_IMAGES_PER_BATCH]

        n_images = len(images)

        prompt = (
            f"I am showing you {n_images} document page images.\n"
            f"They are numbered Image 1 through Image {n_images}.\n\n"
            "TASK: Determine whether any of these images contain a "
            "Table of Contents (TOC), Contents page, or similar listing "
            "of document sections with page numbers.\n\n"
            "IMPORTANT: Use IMAGE NUMBERS (1 to "
            f"{n_images}), NOT any page numbers printed on the document.\n"
            "Ignore any page numbers you see printed on the pages.\n\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            '  "has_toc": true or false,\n'
            f'  "toc_start_image": <image number 1-{n_images} where TOC begins>,\n'
            f'  "toc_end_image": <image number 1-{n_images} where TOC ends>\n'
            "}\n\n"
            "If there is no TOC, return:\n"
            '{"has_toc": false, "toc_start_image": null, "toc_end_image": null}\n\n'
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

            start_img = data.get("toc_start_image")
            end_img = data.get("toc_end_image")
            if start_img is None or end_img is None:
                return None

            start_img = int(start_img)
            end_img = int(end_img)

            # Validate image range
            if start_img < 1 or end_img > n_images or start_img > end_img:
                logger.warning(
                    f"[{document_id}] Invalid TOC image range "
                    f"{start_img}-{end_img} (expected 1-{n_images})"
                )
                start_img = max(1, min(start_img, n_images))
                end_img = max(start_img, min(end_img, n_images))

            # Translate image indices to absolute page numbers
            start_abs = start_img + abs_offset - 1
            end_abs = end_img + abs_offset - 1

            logger.info(
                f"[{document_id}] TOC images {start_img}-{end_img} -> "
                f"absolute pages {start_abs}-{end_abs}"
            )

            return (start_abs, end_abs)

        except Exception as e:
            logger.warning(f"[{document_id}] TOC page detection failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Step 1b: Verify / extend the TOC end page
    # ------------------------------------------------------------------

    def _verify_toc_end(
        self,
        pages_data: List[Dict],
        toc_start: int,
        toc_end_estimate: int,
        document_id: str,
        max_extend: int = 5,
    ) -> int:
        """
        Verify and extend the TOC end page.

        The LLM often underestimates where the TOC ends. Check each
        page after the estimated end to see if it's still a TOC page.

        Args:
            toc_start: Absolute page where TOC starts.
            toc_end_estimate: LLM's estimate of where TOC ends.
            max_extend: Maximum pages to check beyond the estimate.

        Returns:
            The true absolute end page of the TOC.
        """
        total = len(pages_data)
        toc_end = toc_end_estimate

        for abs_page in range(toc_end_estimate + 1,
                              min(toc_end_estimate + max_extend + 1, total + 1)):
            page_data = pages_data[abs_page - 1]
            images = prepare_images_for_bedrock([page_data])

            prompt = (
                "I am showing you a single document page image.\n\n"
                "QUESTION: Is this page a Table of Contents (TOC) page?\n\n"
                "A TOC page contains a structured listing of section names "
                "or headings with corresponding page numbers, often with "
                "dot leaders (......) connecting the heading to the number.\n\n"
                "Answer YES if this page:\n"
                "- Lists section/chapter headings with page numbers\n"
                "- Is a continuation of a TOC from a previous page\n"
                "- Contains entries like 'Section Name ......... 12'\n\n"
                "Answer NO if this page:\n"
                "- Contains actual document content (clauses, paragraphs, text)\n"
                "- Is a cover page, title page, or separator page\n"
                "- Has no structured listing of sections with page numbers\n\n"
                "Return ONLY a JSON object:\n"
                '{"is_toc_page": true or false}'
            )

            try:
                response = invoke_multimodal(
                    images=images, prompt=prompt, max_tokens=64,
                )
                data = self._parse_json_object(response)

                if data.get("is_toc_page", False):
                    toc_end = abs_page
                    logger.info(
                        f"[{document_id}] Page {abs_page} is also a TOC page "
                        f"(extending end from {toc_end_estimate})"
                    )
                else:
                    # First non-TOC page — stop
                    break

            except Exception as e:
                logger.warning(
                    f"[{document_id}] TOC end check failed on page "
                    f"{abs_page}: {e}"
                )
                break

        if toc_end != toc_end_estimate:
            logger.info(
                f"[{document_id}] TOC end extended: "
                f"{toc_end_estimate} -> {toc_end}"
            )

        return toc_end

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
                max_tokens=4096,  # TOC extraction needs more tokens than detection
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

        # Try to parse the complete array first
        match = re.search(r"\[[\s\S]*\]", response)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # If that fails (truncated response), try to salvage complete
        # JSON objects from the truncated array.
        # Find the opening '[' and extract all complete {...} objects.
        bracket_pos = response.find("[")
        if bracket_pos == -1:
            raise ValueError(
                f"No JSON array found in response: {response[:200]}"
            )

        truncated = response[bracket_pos:]
        salvaged = []
        # Find each complete JSON object within the truncated array
        depth = 0
        obj_start = None
        for i, ch in enumerate(truncated):
            if ch == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and obj_start is not None:
                    obj_str = truncated[obj_start : i + 1]
                    try:
                        salvaged.append(json.loads(obj_str))
                    except json.JSONDecodeError:
                        pass
                    obj_start = None

        if salvaged:
            logger.warning(
                f"JSON array was truncated. Salvaged {len(salvaged)} "
                "complete entries."
            )
            return salvaged

        raise ValueError(
            f"No JSON array found in response: {response[:200]}"
        )