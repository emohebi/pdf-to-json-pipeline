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

    def __init__(
        self, pages_data: List[Dict],
        toc_entries: List[Dict] = None,
        toc_abs_start: int = None,
        toc_abs_end: int = None,
    ):
        """
        Args:
            pages_data: Page dicts from pdf_processor.extract_pages().
            toc_entries: TOC entries with section_name and printed_page.
            toc_abs_start: Absolute page where the TOC begins.
            toc_abs_end: Absolute page where the TOC ends.
        """
        self.total_pages = len(pages_data)
        self._offset: Optional[int] = None
        self._printed_to_absolute: Dict[int, int] = {}
        self._build_mapping(pages_data, toc_entries, toc_abs_start, toc_abs_end)

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

        # Last section: end_page left as None (unknown).
        # The caller (section_detector) is responsible for determining
        # where the last TOC section actually ends, rather than assuming
        # it runs to the end of the document.
        resolved[-1]["end_page"] = None

        # Safety
        for sec in resolved:
            if sec["end_page"] is not None and sec["end_page"] < sec["start_page"]:
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
        self,
        pages_data: List[Dict],
        toc_entries: List[Dict] = None,
        toc_abs_start: int = None,
        toc_abs_end: int = None,
    ):
        """
        Build printed -> absolute page mapping.

        Strategy 1 (primary): Use TOC entries as anchor points.
            Search for each TOC heading in the extracted page text to
            find its absolute page. Compute offset from verified matches.

        Strategy 2 (fallback): If no TOC entries provided, try strict
            footer number extraction with consensus voting.

        Strategy 3 (last resort): offset = 0.
        """
        # --- Strategy 1: TOC heading search ---
        if toc_entries:
            offset = self._compute_offset_from_toc(
                pages_data, toc_entries, toc_abs_start, toc_abs_end
            )
            if offset is not None:
                self._offset = offset
                logger.info(f"Page offset: {offset:+d} (from TOC heading search)")
                return

        # --- Strategy 2: Strict footer extraction ---
        raw_mappings = []
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

        if raw_mappings:
            offsets = [m["absolute"] - m["printed"] for m in raw_mappings]
            offset_counts = Counter(offsets)
            best_offset, count = offset_counts.most_common(1)[0]
            agreement = count / len(raw_mappings)

            if agreement >= 0.5:  # Raised threshold — need strong consensus
                self._offset = best_offset
                logger.info(
                    f"Page offset: {best_offset:+d} "
                    f"(footer consensus: {count}/{len(raw_mappings)}, "
                    f"{agreement:.0%})"
                )
                return
            else:
                logger.warning(
                    f"Footer extraction unreliable. "
                    f"Best: {best_offset:+d} with {agreement:.0%} agreement. "
                    f"Top: {offset_counts.most_common(3)}"
                )

        # --- Strategy 3: Assume offset 0 ---
        self._offset = 0
        logger.warning(
            "Could not determine page offset. Using offset=0 "
            "(printed page = absolute page)."
        )

    def _compute_offset_from_toc(
        self,
        pages_data: List[Dict],
        toc_entries: List[Dict],
        toc_abs_start: int = None,
        toc_abs_end: int = None,
    ) -> Optional[int]:
        """
        Compute the offset by searching for TOC heading text in pages.

        For each TOC entry, search through extracted page text for
        the section heading. Where found, offset = abs_page - printed_page.
        Use consensus from multiple verified matches.
        """
        verified_offsets = []

        # Use up to 5 entries for verification (first few are most reliable)
        entries_to_check = toc_entries[:5]

        for entry in entries_to_check:
            heading = entry.get("section_name", "").strip()
            printed = entry.get("printed_page")
            if not heading or printed is None:
                continue

            # Normalise the heading for fuzzy matching
            heading_norm = self._normalise_for_search(heading)
            if len(heading_norm) < 3:
                continue

            # Search pages for this heading
            found_page = self._find_heading_in_pages(
                pages_data, heading_norm, printed,
                toc_abs_start, toc_abs_end,
            )

            if found_page is not None:
                offset = found_page - printed
                verified_offsets.append({
                    "heading": heading,
                    "printed": printed,
                    "absolute": found_page,
                    "offset": offset,
                })
                logger.info(
                    f"  TOC anchor: '{heading}' printed={printed} -> "
                    f"abs={found_page} (offset={offset:+d})"
                )

        if not verified_offsets:
            logger.warning("Could not verify any TOC entries against page text")
            return None

        # Take consensus offset
        offsets = [v["offset"] for v in verified_offsets]
        offset_counts = Counter(offsets)
        best_offset, count = offset_counts.most_common(1)[0]

        logger.info(
            f"TOC-verified offset: {best_offset:+d} "
            f"({count}/{len(verified_offsets)} entries agree)"
        )
        return best_offset

    def _find_heading_in_pages(
        self,
        pages_data: List[Dict],
        heading_norm: str,
        printed_page: int,
        toc_abs_start: int = None,
        toc_abs_end: int = None,
    ) -> Optional[int]:
        """
        Search for a normalised heading string in page text.

        Search strategy:
          - Sections follow AFTER the TOC, so anchor the search at
            toc_abs_end + printed_page (not toc_abs_start).
          - Skip TOC pages themselves (they contain all heading text
            as TOC entries and would give false matches).
          - Search outward from the anchor.
        """
        # Pages to skip (TOC pages contain heading text as entries)
        skip_pages = set()
        if toc_abs_start is not None and toc_abs_end is not None:
            for p in range(toc_abs_start, toc_abs_end + 1):
                skip_pages.add(p)

        # Anchor: sections start after the TOC ends
        if toc_abs_end is not None:
            anchor = toc_abs_end + printed_page
        elif toc_abs_start is not None:
            anchor = toc_abs_start + printed_page
        else:
            anchor = printed_page

        # Clamp anchor to valid range
        anchor = max(1, min(anchor, self.total_pages))

        # Build search order: anchor, anchor+1, anchor-1, anchor+2, ...
        search_order = []
        for delta in range(0, 40):
            page = anchor + delta
            if 1 <= page <= self.total_pages and page not in skip_pages:
                search_order.append(page)
            if delta > 0:
                page = anchor - delta
                if 1 <= page <= self.total_pages and page not in skip_pages:
                    search_order.append(page)

        for abs_page in search_order:
            page_data = pages_data[abs_page - 1]
            text = page_data.get("text", "")
            if not text:
                continue

            text_norm = self._normalise_for_search(text)
            if heading_norm in text_norm:
                return abs_page

        return None

    @staticmethod
    def _normalise_for_search(text: str) -> str:
        """Normalise text for fuzzy heading matching."""
        # Lowercase, collapse whitespace, remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _extract_printed_page_number_strict(text: str) -> Optional[int]:
        """
        Extract printed page number — VERY strict matching.

        Only matches lines that are purely a standalone number.
        """
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            return None

        candidates = []
        for line in lines[-3:]:
            candidates.append(line)
        for line in lines[:3]:
            candidates.append(line)

        for candidate in candidates:
            # Line is ONLY a number (with optional dashes/whitespace)
            m = re.match(r"^[-\s]*(\d{1,4})[-\s]*$", candidate)
            if m:
                return int(m.group(1))

            # "Page 38" standalone
            m = re.match(r"^page\s+(\d{1,4})\s*$", candidate, re.IGNORECASE)
            if m:
                return int(m.group(1))

            # "38 of 120" style
            m = re.match(r"^(\d{1,4})\s+of\s+\d+\s*$", candidate, re.IGNORECASE)
            if m:
                return int(m.group(1))

            # Standalone Roman numeral
            if len(candidate) <= 8:
                cleaned = candidate.strip("- ")
                roman_val = _roman_to_int(cleaned)
                if roman_val is not None and 1 <= roman_val <= 50:
                    return roman_val

        return None