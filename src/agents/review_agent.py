"""
Stage 3.5: Review Agent - Page-by-Page Extraction Accuracy Review.

Compares the extracted JSON content against the source PDF page images
on a per-page basis, producing detailed statistics for each page showing:
  - Coverage percentage (how much of the page content was captured)
  - Missed text / paragraphs / tables
  - Incorrectly extracted content
  - Overall extraction quality score

The agent aligns extracted sections (which have page_range metadata)
back to individual PDF pages, then sends each page image + the
corresponding extracted text to the LLM for a detailed comparison.
"""
import json
import re
from typing import Dict, List, Any, Optional, Tuple

from config.settings import MODEL_MAX_TOKENS_VALIDATION
from config.config_loader import get_prompt, render_prompt
from src.tools.llm_provider import invoke_multimodal
from src.tools.bedrock_vision import prepare_images_for_bedrock
from src.utils import setup_logger, StorageManager

logger = setup_logger("review_agent")


# ── Prompt for page-level review ──────────────────────────────────────

PAGE_REVIEW_PROMPT = """\
You are a document extraction quality reviewer.

I am showing you a PDF page image (page {page_number} of {total_pages}).

Below is the text that was EXTRACTED by an automated pipeline from the
section(s) that span this page. The extracted text may include content
from OTHER pages of the same section — this is expected. Your job is to
focus ONLY on what is visible on THIS page image.

=== EXTRACTED TEXT (from section(s) covering this page) ===
{extracted_text}
=== END OF EXTRACTED TEXT ===

TASK:
1. Read ALL visible text on the page image carefully.
2. For each piece of visible body content on the page, check whether it
   appears SOMEWHERE in the extracted text above.
3. Identify what was correctly captured, what was missed, and what was wrong.

IMPORTANT: The extracted text above may contain content from neighbouring
pages of the same section. Text that appears in the extraction but is NOT
visible on THIS page is NOT an error — simply ignore it. Only assess
content that IS visible on page {page_number}.

=== WHAT TO IGNORE (do NOT count as missed or incorrect) ===
The following page elements should be COMPLETELY IGNORED when assessing
extraction quality. They are intentionally excluded by the pipeline:
- Page numbers (e.g. "Page 5", "5", "- 5 -", or any standalone number
  in a header/footer area)
- Running page headers and footers (e.g. "CONFIDENTIAL", "DRAFT",
  company names, document titles, or dates that repeat on every page
  in the header/footer margin area)
- Confidentiality banners or watermarks (e.g. "CONFIDENTIAL",
  "COMMERCIAL IN CONFIDENCE", "PRIVATE AND CONFIDENTIAL")
- Logos, decorative lines, and visual-only elements
- Copyright notices in footers
- Extracted text that belongs to OTHER pages (not visible on this page)

Do NOT report any of the above as "missed_content" or "incorrect_content".

=== WHAT COUNTS AS CORRECT (tolerance rules) ===
The following differences between the extracted text and the page image
should be treated as CORRECT extraction — do NOT report them as errors:
- Case differences: "DEFINITIONS" vs "Definitions" vs "definitions"
- Minor punctuation differences: straight quotes vs curly quotes,
  en-dash vs em-dash vs hyphen, missing or extra trailing periods
- Unicode vs ASCII equivalents: smart quotes vs straight quotes,
  bullet characters vs dashes, non-breaking spaces vs regular spaces
- Whitespace differences: extra spaces, line breaks, indentation
- Minor formatting: numbered lists "(a)" vs "a)" vs "(a).",
  bullet style "- " vs "* " vs "o "
- Clause number formatting: "1.1" vs "1.1." vs "1.1 " (trailing dot/space)
- Word order within a sentence being slightly different due to line wrapping

These are artefacts of the extraction process, not actual errors.

=== ASSESSMENT CRITERIA ===

COVERAGE: What percentage of the page's MEANINGFUL BODY content was extracted?
- For each paragraph, clause, table row, heading, or text block VISIBLE
  on page {page_number}, check if it appears in the extracted text above.
- EXCLUDE all ignored elements (headers, footers, page numbers, banners).
- 100% = every visible body element on this page was found in the extraction.
- 0% = nothing visible on this page was found in the extraction.

MISSED CONTENT: List specific BODY text/elements VISIBLE on page {page_number}
that do NOT appear anywhere in the extracted text. Be specific — quote
the first few words of each missed item.
Do NOT include ignored elements (headers, footers, page numbers, banners).

INCORRECT CONTENT: List any content where the extraction has WRONG VALUES
compared to what is shown on page {page_number}:
- Wrong words (e.g. "shall" extracted as "should")
- Wrong numbers (e.g. "$500" extracted as "$5000")
- Genuinely hallucinated text that appears nowhere on this page AND is
  not from a neighbouring page of the same section
Do NOT list case, punctuation, whitespace, or formatting differences.
Do NOT list text that is from other pages as hallucinated.

TABLE ACCURACY: If page {page_number} contains tables:
- Were all visible table rows captured in the extraction?
- Were headers correct?
- Were cell values accurate?

Return ONLY a JSON object (no markdown fences):
{{
  "page_number": {page_number},
  "coverage_pct": <0-100, percentage of visible body content found in extraction>,
  "total_elements_on_page": <count of distinct body text blocks/paragraphs/rows VISIBLE on this page, EXCLUDING headers/footers/page numbers>,
  "elements_extracted": <count of those found in the extracted text>,
  "elements_missed": <count of those NOT found in the extracted text>,
  "has_tables": <true if this page contains tables>,
  "table_accuracy_pct": <0-100 if tables exist on this page, null if no tables>,
  "missed_content": [
    {{
      "type": "paragraph | table_row | heading | list_item | caption | other",
      "description": "<first 80 chars of the missed text or brief description>"
    }}
  ],
  "incorrect_content": [
    {{
      "type": "wrong_text | wrong_number | hallucinated",
      "description": "<brief description of the error>",
      "extracted": "<what the extraction says>",
      "actual": "<what the page actually shows>"
    }}
  ],
  "notes": "<any other observations about extraction quality on this page>"
}}

REMEMBER:
- Focus ONLY on content VISIBLE on page {page_number}.
- Text in the extraction from other pages is NOT an error — ignore it.
- Do NOT report page headers, footers, page numbers, or banners as missed.
- Do NOT report case, punctuation, or whitespace differences as errors.

Return the JSON now:"""


# ── Summary prompt ────────────────────────────────────────────────────

SUMMARY_PROMPT = """\
You are a document extraction quality reviewer.

Below are per-page review results for a document extraction.
Provide an overall summary assessment.

PAGE REVIEWS:
{page_reviews_json}

Return ONLY a JSON object:
{{
  "overall_coverage_pct": <weighted average coverage across all pages>,
  "total_pages_reviewed": <count>,
  "pages_with_perfect_coverage": <count of pages with 100% coverage>,
  "pages_with_missed_content": <count of pages with missed content>,
  "pages_with_errors": <count of pages with incorrect content>,
  "most_common_miss_types": ["<top 3 types of missed content>"],
  "most_common_error_types": ["<top 3 types of errors>"],
  "quality_grade": "A | B | C | D | F",
  "quality_description": "<1-2 sentence summary of overall quality>",
  "recommendations": ["<top 3 suggestions to improve extraction>"]
}}

Quality grade guide:
  A = 95%+ coverage, minimal errors
  B = 85-95% coverage, few errors
  C = 70-85% coverage, some errors
  D = 50-70% coverage, many errors
  F = below 50% coverage

Return the JSON now:"""


class ReviewAgent:
    """
    Agent to review extraction quality by comparing extracted JSON
    against source PDF page images on a per-page basis.
    """

    def __init__(self):
        self.storage = StorageManager()

    # ==================================================================
    # Public API
    # ==================================================================

    def review_document(
        self,
        section_jsons: List[Dict],
        document_id: str,
        pages_data: List[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Review extracted sections against source PDF pages.

        Args:
            section_jsons: Extracted section results with page_range
                           and data fields.
            document_id:   Document identifier for logging.
            pages_data:    PDF page dicts from pdf_processor.extract_pages().

        Returns:
            Review report with per-page statistics and summary.
        """
        logger.info(f"[{document_id}] Starting page-by-page review")

        if not pages_data:
            logger.warning(
                f"[{document_id}] No page data provided — "
                f"cannot perform visual review"
            )
            return self._empty_report(document_id)

        total_pages = len(pages_data)

        # Step 1: Build page-to-text mapping from extracted sections
        page_text_map = self._build_page_text_map(
            section_jsons, total_pages, pages_data
        )

        logger.info(
            f"[{document_id}] Mapped extracted text to "
            f"{len(page_text_map)} pages"
        )

        # Step 2: Review each page
        page_reviews: List[Dict] = []

        for page_num in range(1, total_pages + 1):
            extracted_text = page_text_map.get(page_num, "")

            logger.info(
                f"[{document_id}]   Reviewing page {page_num}/{total_pages} "
                f"({len(extracted_text)} chars of extracted text)"
            )

            review = self._review_page(
                pages_data[page_num - 1],
                page_num,
                total_pages,
                extracted_text,
                document_id,
            )
            page_reviews.append(review)

        # Step 3: Compute aggregate statistics
        stats = self._compute_statistics(page_reviews)

        # Step 4: Build report
        report = {
            "document_id": document_id,
            "total_pages": total_pages,
            "page_reviews": page_reviews,
            "statistics": stats,
        }

        # Step 5: Save
        self.storage.save_review_results(document_id, report)

        # Also save plain text version for reference
        full_plain = self._all_sections_to_plain_text(section_jsons)
        self.storage.save_plain_text(document_id, full_plain)

        logger.info(
            f"[{document_id}] Review complete: "
            f"avg coverage {stats['average_coverage_pct']:.1f}%, "
            f"grade {stats['quality_grade']}"
        )

        return report

    # ==================================================================
    # Page-to-text mapping
    # ==================================================================

    def _build_page_text_map(
        self,
        section_jsons: List[Dict],
        total_pages: int,
        pages_data: List[Dict] = None,
    ) -> Dict[int, str]:
        """
        Build a mapping from page number -> extracted text for that page.

        For each section, the FULL extracted text is assigned to EVERY
        page in its page_range. The LLM (which can see the actual page
        image) determines which parts appear on the page it is reviewing.

        The prompt instructs the LLM to only assess content visible on
        the page image and to ignore extracted text from other pages.
        """
        page_texts: Dict[int, List[str]] = {
            p: [] for p in range(1, total_pages + 1)
        }

        for section in section_jsons:
            name = section.get("section_name", "Unknown")
            page_range = section.get("page_range", [])

            # Handle both section formats:
            #  1. Pipeline format: {"section_name": ..., "data": {"heading": ..., "content": [...]}}
            #  2. Flat format:     {"section_name": ..., "heading": ..., "content": [...]}
            data = section.get("data", {})
            if not isinstance(data, dict) or (
                not data.get("content") and not data.get("heading")
                and not data.get("body") and not data.get("text")
            ):
                data = section

            if not page_range or len(page_range) < 2:
                continue

            start_page, end_page = page_range[0], page_range[1]

            # Extract all text from this section
            section_text = self._extract_section_text(data, name)

            if not section_text.strip():
                continue

            # Assign the FULL section text to every page in the range.
            for page_num in range(start_page, end_page + 1):
                if 1 <= page_num <= total_pages:
                    page_texts[page_num].append(section_text)

        # Combine all text for each page
        return {
            page: "\n\n".join(texts)
            for page, texts in page_texts.items()
            if texts
        }

    def _extract_section_text(self, data: Any, section_name: str) -> str:
        """
        Extract readable text from a section's data dict.

        Follows the same approach as json_to_excel.py:
          1. Extract the root heading explicitly from the dict.
          2. Process the 'content' array where every block has a 'type'.

        No generic dict walking — only known fields are accessed.
        No added formatting wrappers — raw text only.
        """
        parts: List[str] = []

        if isinstance(data, dict):
            # Root heading
            heading = data.get("heading", "")
            if isinstance(heading, str) and heading.strip():
                parts.append(heading.strip())

            # Main content array (new schema)
            content = data.get("content", [])
            if isinstance(content, list) and content:
                self._flatten_content(content, parts)

            # Old schema fallback: body, subsections, tables
            body = data.get("body", [])
            if isinstance(body, list) and body:
                for item in body:
                    if isinstance(item, str) and item.strip():
                        parts.append(item.strip())

            subsections = data.get("subsections", [])
            if isinstance(subsections, list) and subsections:
                for sub in subsections:
                    if isinstance(sub, dict):
                        sh = sub.get("heading", "")
                        if isinstance(sh, str) and sh.strip():
                            parts.append(sh.strip())
                        sb = sub.get("body", sub.get("content", []))
                        if isinstance(sb, list):
                            self._flatten_content(sb, parts)

            tables = data.get("tables", [])
            if isinstance(tables, list) and tables:
                for tbl in tables:
                    if isinstance(tbl, dict):
                        self._flatten_table(tbl, parts)

            # Old unhandled_content schema: {"section": "...", "text": "..."}
            text = data.get("text", "")
            if isinstance(text, str) and text.strip():
                heading_str = heading.strip() if isinstance(heading, str) else ""
                if text.strip() != heading_str:
                    parts.append(text.strip())

        elif isinstance(data, list):
            # Array-type sections (e.g. unhandled_content)
            self._flatten_content(data, parts)

        return "\n".join(parts)

    def _flatten_content(self, content: List, parts: List[str]) -> None:
        """
        Process a content array — each item is a typed block.

        Mirrors json_to_excel._flatten_content: handles paragraph,
        table, subsection, and bare strings. Nothing else.
        """
        if not isinstance(content, list):
            return

        for block in content:
            if not isinstance(block, dict):
                # Bare string (old schema body arrays)
                if isinstance(block, str) and block.strip():
                    parts.append(block.strip())
                continue

            btype = block.get("type", "")

            if btype == "paragraph":
                text = block.get("text", "")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())

            elif btype == "table":
                self._flatten_table(block, parts)

            elif btype == "subsection":
                heading = block.get("heading", "")
                if isinstance(heading, str) and heading.strip():
                    parts.append(heading.strip())
                inner = block.get("content", [])
                if isinstance(inner, list):
                    self._flatten_content(inner, parts)

            else:
                # Unknown block type — try to salvage text
                text = block.get("text", "")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                # Also check for heading (untyped subsection-like block)
                heading = block.get("heading", "")
                if isinstance(heading, str) and heading.strip():
                    if heading.strip() != (text.strip() if isinstance(text, str) else ""):
                        parts.append(heading.strip())
                # Recurse into nested content if present
                inner = block.get("content", [])
                if isinstance(inner, list) and inner:
                    self._flatten_content(inner, parts)

    def _flatten_table(self, table: Dict, parts: List[str]) -> None:
        """Extract text from a table block."""
        caption = table.get("caption", "")
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        if isinstance(caption, str) and caption.strip():
            parts.append(caption.strip())
        if isinstance(headers, list) and headers:
            parts.append(" | ".join(str(h) for h in headers))
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, list):
                    parts.append(" | ".join(str(c) for c in row))

    # ==================================================================
    # Per-page review
    # ==================================================================

    def _review_page(
        self,
        page_data: Dict,
        page_number: int,
        total_pages: int,
        extracted_text: str,
        document_id: str,
    ) -> Dict[str, Any]:
        """
        Review a single page by comparing extracted text against the
        page image using the LLM.
        """
        if not extracted_text.strip():
            extracted_text = "(No text was extracted for this page)"

        images = prepare_images_for_bedrock([page_data])

        prompt = PAGE_REVIEW_PROMPT.format(
            page_number=page_number,
            total_pages=total_pages,
            extracted_text=extracted_text,
        )

        retries = 3
        while retries > 0:
            try:
                response = invoke_multimodal(
                    images=images,
                    prompt=prompt,
                    max_tokens=MODEL_MAX_TOKENS_VALIDATION,
                )
                result = self._parse_page_review(response, page_number)
                return result

            except Exception as e:
                retries -= 1
                logger.warning(
                    f"[{document_id}] Page {page_number} review "
                    f"failed ({retries} retries left): {e}"
                )

        # All retries failed
        return {
            "page_number": page_number,
            "coverage_pct": 0,
            "total_elements_on_page": 0,
            "elements_extracted": 0,
            "elements_missed": 0,
            "has_tables": False,
            "table_accuracy_pct": None,
            "missed_content": [],
            "incorrect_content": [],
            "notes": "Review failed after all retries",
            "review_status": "failed",
        }

    def _parse_page_review(
        self, response: str, page_number: int
    ) -> Dict[str, Any]:
        """Parse the LLM's page review response."""
        cleaned = response.strip()
        for pfx in ("```json", "```"):
            if cleaned.startswith(pfx):
                cleaned = cleaned[len(pfx):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise ValueError(
                f"No JSON in page review response: {cleaned[:200]}"
            )

        data = json.loads(match.group(0))

        # Normalise and validate fields
        result = {
            "page_number": page_number,
            "coverage_pct": self._clamp(
                data.get("coverage_pct", 0), 0, 100
            ),
            "total_elements_on_page": max(
                0, int(data.get("total_elements_on_page", 0))
            ),
            "elements_extracted": max(
                0, int(data.get("elements_extracted", 0))
            ),
            "elements_missed": max(
                0, int(data.get("elements_missed", 0))
            ),
            "has_tables": bool(data.get("has_tables", False)),
            "table_accuracy_pct": data.get("table_accuracy_pct"),
            "missed_content": data.get("missed_content", []),
            "incorrect_content": data.get("incorrect_content", []),
            "notes": str(data.get("notes", "")),
            "review_status": "completed",
        }

        # Validate table_accuracy_pct
        if result["table_accuracy_pct"] is not None:
            try:
                result["table_accuracy_pct"] = self._clamp(
                    float(result["table_accuracy_pct"]), 0, 100
                )
            except (ValueError, TypeError):
                result["table_accuracy_pct"] = None

        return result

    # ==================================================================
    # Statistics computation
    # ==================================================================

    def _compute_statistics(
        self, page_reviews: List[Dict]
    ) -> Dict[str, Any]:
        """Compute aggregate statistics from page reviews."""
        completed = [
            r for r in page_reviews
            if r.get("review_status") == "completed"
        ]

        if not completed:
            return {
                "average_coverage_pct": 0.0,
                "median_coverage_pct": 0.0,
                "min_coverage_pct": 0.0,
                "max_coverage_pct": 0.0,
                "total_pages_reviewed": 0,
                "pages_with_perfect_coverage": 0,
                "pages_with_high_coverage": 0,
                "pages_with_low_coverage": 0,
                "pages_with_no_extraction": 0,
                "pages_with_missed_content": 0,
                "pages_with_errors": 0,
                "total_missed_elements": 0,
                "total_incorrect_elements": 0,
                "total_tables_found": 0,
                "average_table_accuracy_pct": 0.0,
                "quality_grade": "F",
                "coverage_by_page": [],
                "worst_pages": [],
            }

        coverages = [r["coverage_pct"] for r in completed]
        coverages_sorted = sorted(coverages)
        n = len(coverages_sorted)

        # Median
        if n % 2 == 0:
            median = (coverages_sorted[n // 2 - 1] + coverages_sorted[n // 2]) / 2
        else:
            median = coverages_sorted[n // 2]

        avg_coverage = sum(coverages) / len(coverages)

        # Table accuracy
        table_pages = [
            r for r in completed
            if r.get("has_tables") and r.get("table_accuracy_pct") is not None
        ]
        avg_table_acc = (
            sum(r["table_accuracy_pct"] for r in table_pages) / len(table_pages)
            if table_pages else 0.0
        )

        # Count categories
        perfect = sum(1 for c in coverages if c >= 99)
        high = sum(1 for c in coverages if 80 <= c < 99)
        low = sum(1 for c in coverages if 0 < c < 50)
        no_extraction = sum(1 for c in coverages if c == 0)

        missed_count = sum(
            len(r.get("missed_content", [])) for r in completed
        )
        error_count = sum(
            len(r.get("incorrect_content", [])) for r in completed
        )
        pages_with_misses = sum(
            1 for r in completed if r.get("missed_content")
        )
        pages_with_errors = sum(
            1 for r in completed if r.get("incorrect_content")
        )

        # Quality grade
        if avg_coverage >= 95 and error_count == 0:
            grade = "A"
        elif avg_coverage >= 85:
            grade = "B"
        elif avg_coverage >= 70:
            grade = "C"
        elif avg_coverage >= 50:
            grade = "D"
        else:
            grade = "F"

        # Coverage by page (for charting)
        coverage_by_page = [
            {"page": r["page_number"], "coverage_pct": r["coverage_pct"]}
            for r in completed
        ]

        # Worst pages (lowest coverage, top 10)
        worst = sorted(completed, key=lambda r: r["coverage_pct"])[:10]
        worst_pages = [
            {
                "page": r["page_number"],
                "coverage_pct": r["coverage_pct"],
                "missed": len(r.get("missed_content", [])),
                "errors": len(r.get("incorrect_content", [])),
            }
            for r in worst
            if r["coverage_pct"] < 100
        ]

        return {
            "average_coverage_pct": round(avg_coverage, 2),
            "median_coverage_pct": round(median, 2),
            "min_coverage_pct": min(coverages),
            "max_coverage_pct": max(coverages),
            "total_pages_reviewed": len(completed),
            "pages_with_perfect_coverage": perfect,
            "pages_with_high_coverage": high,
            "pages_with_low_coverage": low,
            "pages_with_no_extraction": no_extraction,
            "pages_with_missed_content": pages_with_misses,
            "pages_with_errors": pages_with_errors,
            "total_missed_elements": missed_count,
            "total_incorrect_elements": error_count,
            "total_tables_found": len(table_pages),
            "average_table_accuracy_pct": round(avg_table_acc, 2),
            "quality_grade": grade,
            "coverage_by_page": coverage_by_page,
            "worst_pages": worst_pages,
        }

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _clamp(value: Any, lo: float, hi: float) -> float:
        try:
            v = float(value)
            return max(lo, min(hi, v))
        except (ValueError, TypeError):
            return lo

    def _empty_report(self, document_id: str) -> Dict[str, Any]:
        return {
            "document_id": document_id,
            "total_pages": 0,
            "page_reviews": [],
            "statistics": self._compute_statistics([]),
        }

    # ------------------------------------------------------------------
    # Plain text helpers (for saving reference text)
    # ------------------------------------------------------------------

    def _all_sections_to_plain_text(
        self, section_jsons: List[Dict]
    ) -> str:
        parts = []
        for s in section_jsons:
            name = s.get("section_name", "Unknown")
            data = s.get("data", {})
            parts.append(self._extract_section_text(data, name))
        return "\n\n".join(parts)