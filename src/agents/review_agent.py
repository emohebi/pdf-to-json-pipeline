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

Below is the text that was EXTRACTED from this page by an automated pipeline.
Your task is to carefully compare the extracted text against the actual
page image and assess extraction quality.

=== EXTRACTED TEXT FOR THIS PAGE ===
{extracted_text}
=== END OF EXTRACTED TEXT ===

TASK:
1. Read ALL visible text in the page image carefully.
2. Compare it against the extracted text above.
3. Identify what was correctly captured, what was missed, and what was wrong.

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

These are artefacts of the extraction process, not actual errors.

=== ASSESSMENT CRITERIA ===

COVERAGE: What percentage of the page's MEANINGFUL BODY content was extracted?
- Count paragraphs, clauses, table rows, headings, and other body text blocks
- EXCLUDE all ignored elements listed above (headers, footers, page numbers,
  confidentiality banners, watermarks)
- 100% = all meaningful body content was captured
- 0% = nothing was captured

MISSED CONTENT: List specific BODY text/elements visible on the page but NOT
in the extracted text. Be specific — quote the first few words of each missed
item. Do NOT include any ignored elements here.

INCORRECT CONTENT: List any text in the extraction that is factually WRONG
compared to what is shown on the page. This means:
- Wrong words (e.g. "shall" extracted as "should")
- Wrong numbers (e.g. "$500" extracted as "$5000")
- Genuinely hallucinated text that appears nowhere on the page
Do NOT list case differences, punctuation differences, or formatting
differences as errors (see tolerance rules above).

TABLE ACCURACY: If the page contains tables:
- Were all table rows captured?
- Were headers correct?
- Were cell values accurate?

Return ONLY a JSON object (no markdown fences):
{{
  "page_number": {page_number},
  "coverage_pct": <0-100, percentage of meaningful body content extracted>,
  "total_elements_on_page": <count of distinct body text blocks/paragraphs/rows visible, EXCLUDING headers/footers/page numbers>,
  "elements_extracted": <count of those that were captured in the extraction>,
  "elements_missed": <count of those NOT captured>,
  "has_tables": <true if page contains tables>,
  "table_accuracy_pct": <0-100 if tables exist, null if no tables>,
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
- Do NOT report page headers, footers, page numbers, or confidentiality
  banners as missed content.
- Do NOT report case, punctuation, or whitespace differences as errors.
- Focus ONLY on whether the substantive body content was captured accurately.

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
            section_jsons, total_pages
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
    ) -> Dict[int, str]:
        """
        Build a mapping from page number -> extracted text for that page.

        Uses page_range from section metadata to assign content to pages.
        For multi-page sections, the content is distributed across pages
        based on approximate position (since we don't have exact
        page-level boundaries within a section, we include the full
        section text for each page in its range — the LLM will sort
        out what belongs where by looking at the actual page image).
        """
        page_texts: Dict[int, List[str]] = {
            p: [] for p in range(1, total_pages + 1)
        }

        for section in section_jsons:
            name = section.get("section_name", "Unknown")
            page_range = section.get("page_range", [])
            data = section.get("data", {})

            if not page_range or len(page_range) < 2:
                continue

            start_page, end_page = page_range[0], page_range[1]

            # Extract all text from this section
            section_text = self._extract_section_text(data, name)

            if not section_text.strip():
                continue

            # For sections spanning multiple pages, we split the text
            # roughly across pages. However, the most reliable approach
            # is to send the full section text for each page and let
            # the LLM compare against the image.
            n_pages = end_page - start_page + 1

            if n_pages == 1:
                page_texts[start_page].append(section_text)
            else:
                # Split text into roughly equal chunks per page
                lines = section_text.split("\n")
                lines = [l for l in lines if l.strip()]
                chunk_size = max(1, len(lines) // n_pages)

                for page_idx in range(n_pages):
                    page_num = start_page + page_idx
                    if page_num > total_pages:
                        break

                    chunk_start = page_idx * chunk_size
                    if page_idx == n_pages - 1:
                        # Last page gets the remainder
                        chunk = lines[chunk_start:]
                    else:
                        chunk = lines[chunk_start:chunk_start + chunk_size]

                    if chunk:
                        page_texts[page_num].append("\n".join(chunk))

        # Combine all text for each page
        return {
            page: "\n\n".join(texts)
            for page, texts in page_texts.items()
            if texts
        }

    def _extract_section_text(self, data: Any, section_name: str) -> str:
        """
        Extract readable text from a section's data.

        IMPORTANT: Output only the raw document text — no formatting
        wrappers, no decorators. The LLM compares this text against
        the page image, so any added wrappers would be flagged as
        hallucinated content.
        """
        parts: List[str] = []
        self._walk_content(data, parts, depth=0)
        return "\n".join(parts)

    def _walk_content(
        self, data: Any, parts: List[str], depth: int
    ) -> None:
        """
        Recursively walk content blocks and extract text.

        IMPORTANT: Output only the raw document text without any
        added formatting (no '=== ... ===' wrappers, no '--- ... ---'
        decorators, no '[Table: ...]' prefixes). The LLM compares
        this against the page image, so any added wrappers get
        falsely reported as hallucinated content.
        """
        indent = "  " * depth

        if isinstance(data, str):
            stripped = data.strip()
            if stripped:
                parts.append(f"{indent}{stripped}")

        elif isinstance(data, dict):
            btype = data.get("type", "")

            if btype == "paragraph":
                text = data.get("text", "").strip()
                if text:
                    parts.append(f"{indent}{text}")

            elif btype == "table":
                caption = data.get("caption", "")
                headers = data.get("headers", [])
                rows = data.get("rows", [])

                if caption:
                    parts.append(f"{indent}{caption}")
                if headers:
                    parts.append(
                        f"{indent}"
                        f"{' | '.join(str(h) for h in headers)}"
                    )
                for row in rows:
                    if isinstance(row, list):
                        parts.append(
                            f"{indent}"
                            f"{' | '.join(str(c) for c in row)}"
                        )

            elif btype == "subsection":
                heading = data.get("heading", "").strip()
                if heading:
                    parts.append(f"{indent}{heading}")
                inner = data.get("content", [])
                self._walk_content(inner, parts, depth + 1)

            else:
                # Generic dict — walk values
                for key, val in data.items():
                    if key.startswith("_") or key in (
                        "image", "type", "page_range"
                    ):
                        continue
                    if key == "heading" and isinstance(val, str) and val.strip():
                        parts.append(f"{indent}{val.strip()}")
                    elif key == "content":
                        self._walk_content(val, parts, depth)
                    elif key == "text" and isinstance(val, str) and val.strip():
                        parts.append(f"{indent}{val.strip()}")
                    else:
                        self._walk_content(val, parts, depth)

        elif isinstance(data, list):
            for item in data:
                self._walk_content(item, parts, depth)

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