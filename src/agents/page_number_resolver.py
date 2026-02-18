"""
Page Number Resolver.

Maps printed page numbers (as shown in a Table of Contents)
to absolute PDF page indices (1-based).

Primary strategy: compute a global offset from the TOC itself.
Since we know which absolute page the TOC is on, and the TOC lists
sections with printed page numbers, we can derive:
    absolute_page = printed_page + offset

Fallback strategy: scan PyMuPDF text for strictly standalone page
numbers in footers/headers to build a mapping.
"""
import re
from typing import List, Dict, Optional
from collections import Counter

from src.utils import setup_logger

logger = setup_logger("page_resolver")

# Roman numeral pattern
_ROMAN_RE = re.compile(
    r"^(m{0,3})(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$",
    re.IGNORECASE,
)


def _roman_to_int(s: str) -> Optional[int]:
    s = s.strip().lower()
    if not _ROMAN_RE.match(s):
        return None
    rom_val = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    result = prev = 0
    for ch in reversed(s):
        val = rom_val.get(ch, 0)
        result = result - val if val < prev else result + val
        prev = val
    return result if result > 0 else None


class PageNumberResolver:
    """
    Builds and uses a mapping from printed page numbers to absolute
    PDF page indices.
    """

    def __init__(self, pages_data: List[Dict], toc_abs_start: int = None):
        """
        Args:
            pages_data: Page dicts from pdf_processor.extract_pages().
            toc_abs_start: Absolute page number where the TOC begins
                           (used to anchor the offset calculation).
        """
        self.total_pages = len(pages_data)
        self._offset: Optional[int] = None
        self._printed_to_absolute: Dict[int, int] = {}
        self._build_mapping(pages_data, toc_abs_start)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_printed_to_absolute(self, printed_page: int) -> Optional[int]:
        """Given a printed page number, return the absolute PDF page index."""
        # Try offset first (most reliable)
        if self._offset is not None:
            result = printed_page + self._offset
            if 1 <= result <= self.total_pages:
                return result

        # Try direct mapping
        return self._printed_to_absolute.get(printed_page)

    def resolve_with_fallback(self, printed_page: int) -> int:
        """Resolve printed -> absolute, with fallback to best guess."""
        result = self.resolve_printed_to_absolute(printed_page)
        if result is not None:
            return result

        # If offset exists but result was out of range, clamp it
        if self._offset is not None:
            return max(1, min(printed_page + self._offset, self.total_pages))

        # No offset, no mapping -- assume printed == absolute
        return max(1, min(printed_page, self.total_pages))

    def resolve_toc_entries(
        self,
        toc_entries: List[Dict],
        total_pages: int,
    ) -> List[Dict]:
        """
        Convert TOC entries with printed page numbers into section
        definitions with absolute start_page / end_page.

        Preserves original TOC order. Enforces monotonically increasing
        start pages (flags and fixes violations).
        """
        if not toc_entries:
            return []

        # First pass: resolve all start pages (preserve TOC order)
        resolved = []
        for entry in toc_entries:
            printed = entry.get("printed_page")
            if printed is None:
                continue

            abs_start = self.resolve_with_fallback(printed)

            resolved.append({
                "section_type": entry.get("section_type") or "unhandled_content",
                "section_name": entry.get("section_name", "Unknown"),
                "start_page": abs_start,
                "end_page": None,
                "confidence": 0.90,
                "_source": "toc",
                "_printed_page": printed,
                "_resolution_reliable": True,
            })

        if not resolved:
            return []

        # Second pass: enforce monotonically increasing start pages
        # NOTE: two sections CAN start on the same page (equal is OK),
        # only flag strictly BACKWARD jumps.
        for i in range(1, len(resolved)):
            if resolved[i]["start_page"] < resolved[i - 1]["start_page"]:
                logger.warning(
                    f"Non-monotonic: '{resolved[i]['section_name']}' "
                    f"resolved to page {resolved[i]['start_page']} "
                    f"but previous '{resolved[i - 1]['section_name']}' "
                    f"is at page {resolved[i - 1]['start_page']}. "
                    f"(printed_page={resolved[i]['_printed_page']})"
                )
                resolved[i]["_resolution_reliable"] = False

        # Third pass: fix unreliable runs
        self._fix_non_monotonic(resolved, total_pages)

        # Fourth pass: compute end_page
        for i in range(len(resolved) - 1):
            resolved[i]["end_page"] = resolved[i + 1]["start_page"] - 1
        resolved[-1]["end_page"] = total_pages

        # Safety
        for sec in resolved:
            if sec["end_page"] < sec["start_page"]:
                sec["end_page"] = sec["start_page"]
            sec.pop("_resolution_reliable", None)

        return resolved

    def _fix_non_monotonic(
        self, resolved: List[Dict], total_pages: int
    ) -> None:
        """Fix non-monotonic start pages by interpolating from neighbours."""
        i = 0
        while i < len(resolved):
            if resolved[i]["_resolution_reliable"]:
                i += 1
                continue

            run_start = i
            run_end = i
            while run_end < len(resolved) and not resolved[run_end]["_resolution_reliable"]:
                run_end += 1

            prev_page = resolved[run_start - 1]["start_page"] if run_start > 0 else 0
            next_page = resolved[run_end]["start_page"] if run_end < len(resolved) else total_pages

            run_len = run_end - run_start
            available = next_page - prev_page
            step = max(1, available // (run_len + 1))

            for j in range(run_start, run_end):
                estimated = prev_page + step * (j - run_start + 1)
                estimated = max(prev_page + 1, min(estimated, total_pages))
                logger.info(
                    f"  Estimated '{resolved[j]['section_name']}' "
                    f"start_page: {estimated} (was {resolved[j]['start_page']})"
                )
                resolved[j]["start_page"] = estimated
                resolved[j]["confidence"] = 0.60

            i = run_end

    def get_mapping_stats(self) -> Dict:
        return {
            "total_pages": self.total_pages,
            "offset": self._offset,
            "mapped_pages": len(self._printed_to_absolute),
            "coverage_pct": (
                round(len(self._printed_to_absolute) / self.total_pages * 100, 1)
                if self.total_pages > 0 else 0
            ),
            "sample_mappings": dict(
                list(self._printed_to_absolute.items())[:10]
            ),
        }

    # ------------------------------------------------------------------
    # Mapping builder
    # ------------------------------------------------------------------

    def _build_mapping(
        self, pages_data: List[Dict], toc_abs_start: int = None
    ):
        """
        Build printed -> absolute page mapping.

        Strategy 1 (primary): Extract standalone page numbers from
        footer/header text using STRICT matching (only lines that are
        purely a number, no surrounding text). Compute a consensus
        offset from these reliable matches.

        Strategy 2 (fallback): If we know the TOC absolute page and
        have TOC entries, we can derive the offset externally.
        """
        raw_mappings: List[Dict] = []  # {printed, absolute}

        for page in pages_data:
            abs_idx = page["page_number"]
            text = page.get("text", "")
            if not text.strip():
                continue

            printed = self._extract_printed_page_number_strict(text)
            if printed is not None and printed > 0:
                raw_mappings.append({"printed": printed, "absolute": abs_idx})
                if printed not in self._printed_to_absolute:
                    self._printed_to_absolute[printed] = abs_idx

        # Compute consensus offset from the strict mappings
        if raw_mappings:
            offsets = [m["absolute"] - m["printed"] for m in raw_mappings]
            offset_counts = Counter(offsets)
            best_offset, count = offset_counts.most_common(1)[0]

            # Accept if at least 30% of mappings agree on this offset
            agreement = count / len(raw_mappings)
            if agreement >= 0.3:
                self._offset = best_offset
                logger.info(
                    f"Page offset: {best_offset:+d} "
                    f"(consensus from {count}/{len(raw_mappings)} pages, "
                    f"{agreement:.0%} agreement)"
                )
            else:
                logger.warning(
                    f"No consensus offset found. "
                    f"Best: {best_offset:+d} with only {agreement:.0%} agreement. "
                    f"Top offsets: {offset_counts.most_common(3)}"
                )
        else:
            logger.warning(
                "No page numbers extracted from text. "
                "Will use printed_page = absolute_page (offset 0)."
            )
            self._offset = 0

        logger.info(
            f"Page mapping: {len(self._printed_to_absolute)}/{self.total_pages} "
            f"pages resolved, offset={self._offset}"
        )

    @staticmethod
    def _extract_printed_page_number_strict(text: str) -> Optional[int]:
        """
        Extract the printed page number using STRICT matching.

        Only matches lines where the page number is essentially the
        ONLY content on the line — a standalone number in a footer or
        header, not a clause number embedded in a heading.

        Matches:
          "38"
          " 38 "
          "- 38 -"
          "Page 38"
          "38 of 120"

        Does NOT match:
          "12. Assignment"           (clause number)
          "Section 5.3 Liability"    (section number)
          "Table 12: Revenue"        (table number)
          "submitted on 12 Jan 2024" (date)
        """
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            return None

        candidates = []
        # Footer (last 3 lines) — most common location
        for line in lines[-3:]:
            candidates.append(line)
        # Header (first 3 lines)
        for line in lines[:3]:
            candidates.append(line)

        for candidate in candidates:
            # STRICT: line is ONLY a number (with optional dashes/whitespace)
            # e.g., "38", "- 38 -", "-- 38 --"
            m = re.match(r"^[-\s]*(\d{1,4})[-\s]*$", candidate)
            if m:
                return int(m.group(1))

            # "Page 38" or "page 38" (standalone, nothing else significant)
            m = re.match(r"^page\s+(\d{1,4})\s*$", candidate, re.IGNORECASE)
            if m:
                return int(m.group(1))

            # "38 of 120" style
            m = re.match(r"^(\d{1,4})\s+of\s+\d+\s*$", candidate, re.IGNORECASE)
            if m:
                return int(m.group(1))

            # Standalone Roman numeral (front matter, very short line)
            if len(candidate) <= 8:
                cleaned = candidate.strip("- ")
                roman_val = _roman_to_int(cleaned)
                if roman_val is not None and 1 <= roman_val <= 50:
                    return roman_val

        return None