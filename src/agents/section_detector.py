"""
Section Detector - Accumulative Window Approach

Scans every page of the document from page 1 sequentially, detecting all
structural elements: cover pages, tables of contents, blank pages, and
content sections with their exact page ranges.

Strategy (mirrors table_detector.py):
  1. Scan page N: "What starts on this page?"
     → Returns a LIST of elements found (cover, toc, blank, sections).
       A single page may contain multiple section headings.
  2. For each section found, grow window page by page to find its end:
     [N], [N, N+1], [N, N+1, N+2]...
  3. At each step the LLM sees the ORIGIN page plus all pages up to the
     check page, so it always has full context.
  4. The boundary check decides continuation based on semantic coherence
     AND identifies any new section found (name + type), eliminating the
     need for a separate identification call.
  5. When the boundary check reports a new section:
     - Record the current section's end page
     - Trace the new section immediately (chain)
  6. After chain breaks, advance past the last section and resume scanning.

No TOC parsing, no page-number offsets, no numbering-pattern matching.
All headings are treated as flat sections (no title vs section hierarchy).
"""

import json
import re
from typing import List, Dict, Optional, Tuple

from config.settings import MAX_IMAGES_PER_BATCH
from config.config_loader import (
    get_section_definitions,
    get_document_type_name,
    build_heading_alias_rules,
    get_merge_rules,
)
from src.tools.llm_provider import invoke_multimodal
from src.tools.bedrock_vision import prepare_images_for_bedrock
from src.utils import setup_logger, StorageManager

logger = setup_logger("section_detector")


# =====================================================================
# Prompts
# =====================================================================

SCAN_PAGE_PROMPT = """\
I am showing you a single page from a {doc_type} document.
This is page {page_num} of {total_pages}.

TASK: What structural elements START on this page?
Examine the ENTIRE page from top to bottom. A single page may contain
content from a previous section at the top, then one or more NEW sections
starting further down.

ELEMENT TYPES:
- "cover_page"     : Title page, cover, or document header page
- "toc_page"       : Table of Contents listing sections with page numbers
- "blank_page"     : Blank or nearly blank page
- "section_start"  : A new document section begins somewhere on this page

WHAT COUNTS AS A SECTION:
A section is a major structural division of the document that introduces
a NEW TOPIC or subject area. Each section covers a self-contained subject.

Any prominent heading that introduces a distinct topic is a section,
whether it is a high-level document part (e.g. "STANDARD TERMS AND
CONDITIONS", "SCHEDULE 1 - SCOPE OF WORK") or a specific topic
(e.g. "DEFINITIONS", "INTERPRETATION", "TERMINATION", "INSURANCE").

HOW TO IDENTIFY A SECTION START:
Look for a prominent heading or title that introduces a DIFFERENT TOPIC
from what came before. The heading usually stands out visually (bold,
larger font, centered, or preceded by whitespace). However, a TOP LEVEL
TITLE that appears above a numbered heading may be less visually bold
than the numbered heading below it — prefer the top-level title as the
section name in that case (see CRITICAL note below).

DO NOT report as section_start:
- Clauses, sub-clauses, or paragraphs that continue the SAME TOPIC
  (even if they have numbers like 1., 1.1, 2.3, (a), (i))
- Bold text for emphasis within paragraphs
- Table headers, figure captions, list items
- Page headers or footers
- Numbered items that elaborate on the same subject
- Heading under or related to a top level section name

CRITICAL:
If you see a TOP LEVEL TITLE above a numbered heading, always pick the
TOP LEVEL TITLE as the section name — even if its font size is smaller
or the same as the numbered heading below it. Position on the page
(higher up, more isolated, visually separate) takes priority over font
size when identifying the section name.

Example:
                [TOP LEVEL TITLE] --> (pick this one as SECTION NAME
                                       even if the font size is smaller
                                       or same as below headings)
    [1  INTRODUCTION] --> (NOT a SECTION, just a sub-heading beneath
                           the top level title above)

{heading_alias_rules}

SECTION TYPES: {section_types}

Return ONLY valid JSON:
{{
  "elements_starting_here": [
    {{
      "page_type": "section_start" | "cover_page" | "toc_page" | "blank_page",
      "section_name": "<heading text if section_start, or null, the most dominant top level text>",
      "section_type": "<from section types list, or null>",
      "confidence": <0.0-1.0>
    }}
  ]
}}

If NOTHING new starts on this page (pure continuation), return:
{{
  "elements_starting_here": []
}}
"""


BOUNDARY_CHECK_PROMPT = """\
I am showing you {n_images} consecutive pages from a {doc_type} document.

The section "{section_name}" starts on the first page shown (page {start_page}).

Your job: Look at ONLY the LAST page shown (page {check_page}) and determine
whether the section "{section_name}" is still continuing there, or whether
a NEW and DIFFERENT section has begun.

== HOW TO DECIDE ==

The section "{section_name}" CONTINUES on page {check_page} if:
- The text on page {check_page} is about the SAME TOPIC as "{section_name}"
- The content is a logical continuation: more clauses, paragraphs, tables,
  or details that elaborate on or belong to "{section_name}"
- Even if the content has sub-numbering (1.1, 2.3, (a), etc.), it still
  belongs to "{section_name}" if it discusses the same subject matter
- The text on page {check_page} is same type as {section_type}

A NEW section starts on page {check_page} if:
- A prominent heading introduces a DIFFERENT TOPIC or subject area
- The subject matter shifts away from "{section_name}" to something
  clearly distinct (e.g. from "Definitions" to "Liability")
- A structural break appears: a new cover page, a new Table of Contents,
  a signature/execution page, a blank separator, or an appendix divider
- A new document part heading appears (e.g. "SCHEDULE 2", "PART B")

== IMPORTANT ==
Do NOT confuse sub-divisions within "{section_name}" with new sections.
Clauses, sub-clauses, and numbered paragraphs that discuss aspects of
the SAME TOPIC are part of "{section_name}", not new sections.

The test is: "Has the SUBJECT MATTER changed?" If the topic is still
"{section_name}", it continues — regardless of numbering or formatting.

== ANSWER ==

Return ONLY valid JSON:
{{
  "has_current_section_content": true | false,
  "new_section_starts": true | false,
  "new_sections": [
    {{
      "section_name": "<heading of the new section>",
      "section_type": "<from types: {section_types}>",
      "page_type": "section_start" | "cover_page" | "toc_page" | "blank_page"
    }}
  ],
  "reason": "<brief explanation of what you see on page {check_page}>"
}}

RULES for "new_sections":
- If new_section_starts is false, return an empty list: "new_sections": []
- If new_section_starts is true, list ALL new sections/elements that begin
  on page {check_page} (there may be more than one if a short section
  starts and ends on the same page followed by another).
"""

TOC_BOUNDARY_CHECK_PROMPT = """\
I am showing you {n_images} consecutive pages from a {doc_type} document.

A Table of Contents (TOC) starts on the first page shown (page {start_page}).

Your job: Look at ONLY the LAST page shown (page {check_page}) and determine
whether the Table of Contents is still continuing there, or whether the TOC
has ended and something DIFFERENT has begun.

== HOW TO DECIDE ==

The TOC CONTINUES on page {check_page} if:
- The page contains a structured listing of section names, headings, or
  chapter titles paired with page numbers or dot leaders
  (e.g. "DEFINITIONS ......... 12")
- The layout matches the TOC format from page {start_page}: indented entries,
  numbered items with corresponding page references, or a columnar list of
  contents
- Even if the style varies slightly (e.g. grouping under a sub-heading like
  "SCHEDULES" or "APPENDICES"), it is still a TOC if it lists document parts
  with page references

The TOC ENDS and something NEW starts on page {check_page} if:
- The page contains actual document content: paragraphs, clauses, definitions,
  legal text, or narrative prose — not just a listing of headings with page
  numbers
- A cover page, blank page, or section heading with body text appears
- The format clearly shifts from a structured listing to document content
- A different TOC appears (a separate Table of Contents for a different
  document part, with its own "TABLE OF CONTENTS" title)

== IMPORTANT ==
A TOC page lists WHERE content is — it does NOT contain the content itself.
If page {check_page} has actual body text, clauses, or detailed content
(not just headings with page numbers), the TOC has ended.

== ANSWER ==

Return ONLY valid JSON:
{{
  "has_current_section_content": true | false,
  "new_section_starts": true | false,
  "new_sections": [
    {{
      "section_name": "<heading or descriptive name of what starts>",
      "section_type": "<from types: {section_types}>",
      "page_type": "section_start" | "cover_page" | "toc_page" | "blank_page"
    }}
  ],
  "reason": "<brief explanation of what you see on page {check_page}>"
}}

RULES:
- "has_current_section_content": true if page {check_page} has TOC entries
- "new_section_starts": true if something OTHER than this TOC begins on
  page {check_page} (including actual document content, a cover page,
  a blank page, or a separate TOC)
- If new_section_starts is true, "new_sections" MUST list what begins.
  If a section heading with body text appears, report it as "section_start".
  If a second separate TOC starts, report it as "toc_page".
- If new_section_starts is false, return "new_sections": []
"""

POST_INTERRUPTION_CHECK_PROMPT = """\
I am showing you 2 pages from a {doc_type} document.

PAGE 1 (page {origin_page}): This is from the section "{section_name}".
PAGE 2 (page {check_page}): This is the page I am asking about.

Between these two pages there was a {interruption_type} (pages {int_start}-{int_end})
which interrupted the document flow.

TASK: Does page {check_page} CONTINUE the section "{section_name}" from
page {origin_page}?

It CONTINUES if:
- The content on page {check_page} is about the SAME TOPIC as "{section_name}"
- It reads as a logical continuation: more clauses, paragraphs, or details
  that belong to "{section_name}"
- The subject matter has NOT changed — it picks up where page {origin_page}
  left off (or continues the same topic after the {interruption_type})

It does NOT continue if:
- A new and DIFFERENT topic or section heading appears
- The content is clearly about a different subject
- A new document part, schedule, or appendix begins

Return ONLY valid JSON:
{{
  "continues_previous_section": true | false,
  "reason": "<brief explanation>"
}}
"""

# =====================================================================
# Main Detector
# =====================================================================

class SectionDetectionAgent:

    def __init__(self):
        self.section_definitions = {
                        "cover page": "Title page, cover, or document header page",
                        "table of content": "Table of Contents listing sections with page numbers",
                        "blank page": "Blank or nearly blank page",
                        "section": "A new document section begins somewhere on this page",
                    } #get_section_definitions()
        self.storage = StorageManager()
        self.max_window = MAX_IMAGES_PER_BATCH

    # ==================================================================
    # Public API
    # ==================================================================

    def detect_sections(
        self, pages_data: List[Dict], document_id: str
    ) -> Optional[List[Dict]]:
        """
        Detect all sections in the document.

        Args:
            pages_data: List of page dicts (one per page, 0-indexed).
                        Each dict is passed to prepare_images_for_bedrock().
            document_id: Identifier for logging / storage.

        Returns:
            List of section dicts sorted by start_page, or None on failure.
            Each dict has: section_type, section_name, start_page, end_page,
            confidence, _source.
        """
        total = len(pages_data)
        logger.info(f"[{document_id}] Detecting sections in {total} pages")

        try:
            sections = self._scan_all_pages(pages_data, document_id)

            if sections:
                sections = self._ensure_full_coverage(
                    sections, total, document_id
                )
                self.storage.save_detection_result(document_id, sections)
                logger.info(
                    f"[{document_id}] Final: {len(sections)} sections"
                )

            return sections

        except Exception as e:
            logger.error(f"[{document_id}] Detection failed: {e}")
            return None

    # ==================================================================
    # Core scanning loop
    # ==================================================================

    def _scan_all_pages(
        self, pages_data: List[Dict], document_id: str
    ) -> List[Dict]:
        """
        Sequential scan from page 1 to the last page.

        Flow per page:
          scan → list of elements starting here
          for each element:
            (cover/toc/blank) → absorb consecutive same-type pages
            (section_start)   → trace → chain via boundary info → advance
          if empty list → continuation, advance +1
        """
        type_map = {
                        "cover_page": ("front_matter", "Cover Page"),
                        "toc_page": ("front_matter", "Table of Contents"),
                        "blank_page": ("unhandled_content", "Blank Page"),
                    }
        total = len(pages_data)
        sections: List[Dict] = []
        current_page = 1
        claimed_pages: set = set()
        last_content_section: Optional[Dict] = None  # track last section before TOC/cover/blank

        while current_page <= total:
            if current_page in claimed_pages:
                current_page += 1
                if current_page > total: break

            logger.info(f"[{document_id}] Scanning page {current_page}/{total}")

            # --- Step 1: scan this single page ---
            scan = self._scan_page(
                pages_data[current_page - 1], current_page, total,
            )
            elements = scan.get("elements_starting_here", [])

            if not elements:
                logger.info(f"[{document_id}]   continuation (skip)")
                claimed_pages.add(current_page)
                current_page += 1
                continue
            claimed_pages.add(current_page)
            # --- Step 2: process each element found on this page ---
            element_page = current_page
            for elem in elements:
                if current_page > element_page:
                    break
                page_type = elem.get("page_type", "section_start")

                # ── COVER / TOC / BLANK ───────────────────────
                if page_type in ("cover_page", "toc_page", "blank_page"):
                    name, stype = type_map[page_type]
                
                elif page_type == "section_start":
                    name = elem.get("section_name") or "Unknown Section"
                    stype = elem.get("section_type") or "unhandled_content"

                logger.info(
                    f"[{document_id}]   Section: '{name}' ({stype})"
                )

                # Trace using accumulative window
                end_page, next_sections, shared_page = self._trace_section(
                    pages_data, current_page, name, stype,
                    total, document_id,
                )
                if not shared_page:
                    claimed_pages.add(end_page)
                sections.append({
                    "section_type": stype,
                    "section_name": name,
                    "start_page": current_page,
                    "end_page": end_page,
                    "confidence": elem.get("confidence", 0.85),
                    "_source": "scan",
                })

                logger.info(
                    f"[{document_id}]   Pages {current_page}-{end_page}"
                )

                # --- Post-interruption continuation check ---
                # If we just finished a TOC/cover/blank and there was
                # a content section before it, check if the page after
                # continues that previous section.
                if (
                    page_type in ("cover_page", "toc_page", "blank_page")
                    and last_content_section is not None
                    and end_page + 1 <= total
                ):
                    prev_sec = last_content_section
                    post_page = end_page + 1

                    logger.info(
                        f"[{document_id}]   Checking if page {post_page} "
                        f"continues '{prev_sec['section_name']}' "
                        f"after {name}"
                    )

                    continues = self._check_post_interruption_continuation(
                        pages_data,
                        prev_sec, post_page,
                        interruption_type=name,
                        int_start=current_page,
                        int_end=end_page,
                    )

                    if continues:
                        logger.info(
                            f"[{document_id}]   YES — resuming "
                            f"'{prev_sec['section_name']}' "
                            f"from page {post_page}"
                        )

                        # Resume tracing from post_page onward
                        # (continuation already confirmed, just find
                        # where this resumed section ends)
                        resume_end, resume_next, resume_shared = (
                            self._trace_section(
                                pages_data, post_page,
                                prev_sec["section_name"],
                                prev_sec["section_type"],
                                total, document_id,
                            )
                        )

                        # Extend the previous section's end page
                        prev_sec["end_page"] = resume_end
                        for p in range(post_page, resume_end + 1):
                            claimed_pages.add(p)

                        logger.info(
                            f"[{document_id}]   Extended "
                            f"'{prev_sec['section_name']}' to "
                            f"page {resume_end}"
                        )

                        end_page = resume_end

                # Track last content section for post-interruption checks
                if page_type == "section_start":
                    last_content_section = sections[-1]
                elif page_type in ("cover_page", "toc_page"):
                    # Don't clear on blank — a blank page between
                    # TOC and continuation shouldn't reset tracking
                    pass

                current_page = end_page

        # Post-process: merge adjacent same-type sections
        sections.sort(key=lambda s: s["start_page"])
        sections = self._merge_adjacent(sections, document_id)

        logger.info(f"[{document_id}] Scan done: {len(sections)} sections")
        for s in sections:
            logger.info(
                f"  [{s['section_type']}] '{s['section_name']}'"
                f" pp {s['start_page']}-{s['end_page']}"
            )
        return sections

    # ==================================================================
    # Single-page scan
    # ==================================================================

    def _scan_page(
        self, page_data: Dict, page_num: int, total_pages: int,
    ) -> Dict:
        """
        Scan a single page — what starts here?

        Returns dict with "elements_starting_here": list of elements.
        Each element has page_type, section_name, section_type, confidence.
        """
        images = prepare_images_for_bedrock([page_data])

        prompt = SCAN_PAGE_PROMPT.format(
            page_num=page_num,
            total_pages=total_pages,
            doc_type=get_document_type_name(),
            section_types=", ".join(self.section_definitions.keys()),
            heading_alias_rules=build_heading_alias_rules(),
        )

        try:
            resp = invoke_multimodal(
                images=images, prompt=prompt, max_tokens=512,
            )
            return self._parse_json(resp)
        except Exception as e:
            logger.warning(f"Scan failed page {page_num}: {e}")
            return {"elements_starting_here": []}

    # ==================================================================
    # Section tracing (accumulative window — core algorithm)
    # ==================================================================

    def _trace_section(
        self,
        pages_data: List[Dict],
        section_start: int,
        section_name: str,
        section_type: str,
        total_pages: int,
        document_id: str,
    ) -> Tuple[int, List[Dict]]:
        """
        Grow window page by page from section_start until the LLM
        detects a semantic/logical break — a different topic starts.

        Window growth:
          [start] → [start, start+1] → [start, start+1, start+2] ...

        The LLM always sees the origin page so it can judge whether
        the check page is still about the same topic.

        If the window exceeds max_window, keep the ORIGIN page plus
        the most recent pages (sliding strategy).

        Returns:
            (end_page, next_sections)
            - end_page: last page belonging to this section
            - next_sections: list of new section dicts discovered on
              the boundary page (each has section_name, section_type,
              page_type, _boundary_page). Empty list if section runs
              to end of document.

        Shared-page handling:
          Both old content AND new heading on check_page:
            end_page = check_page, next sections start on check_page
          Only new heading (no old content) on check_page:
            end_page = check_page - 1, next sections start on check_page
        """
        current_end = section_start
        exclude_pages = []
        while current_end < total_pages:
            check_page = current_end + 1
            window_size = check_page - section_start + 1
            logger.info(f"Window size: pages: {section_start} - {check_page}")
            # Build the accumulative window
            if window_size <= self.max_window:
                window = [
                    pages_data[p - 1]
                    for p in range(section_start, check_page + 1)
                ]
            else:
                # Sliding: origin + most recent pages
                window = [pages_data[section_start - 1]]
                recent_start = check_page - self.max_window + 2
                logger.info(f"Window size capped: origin + pages: {recent_start} - {check_page}")
                for p in range(recent_start, check_page + 1):
                    window.append(pages_data[p - 1])

            result = self._check_boundary(
                window, section_name, section_type, section_start, check_page,
            )

            has_content = result.get("has_current_section_content", True)
            new_starts = result.get("new_section_starts", False)
            new_sections = result.get("new_sections", [])

            # Tag each discovered section with the boundary page
            for ns in new_sections:
                ns["_boundary_page"] = check_page

            if new_starts and has_content:
                # SHARED PAGE: old content + new heading on same page
                logger.info(
                    f"[{document_id}]   Page {check_page}: shared — "
                    f"'{section_name}' content + new section. "
                    f"End = {check_page}."
                )
                return check_page, new_sections, True

            elif new_starts and not has_content:
                # EXCLUSIVE: new section owns this page entirely
                logger.info(
                    f"[{document_id}]   Page {check_page}: new section "
                    f"starts exclusively. '{section_name}' ends at "
                    f"{current_end}."
                )
                return current_end, new_sections, False

            elif has_content:
                # Section continues — grow window
                current_end = check_page

            else:
                # No old content, no new section (e.g. blank page,
                # appendix divider). Treat as boundary.
                logger.info(
                    f"[{document_id}]   Page {check_page}: no content, "
                    f"no new section. '{section_name}' ends at "
                    f"{current_end}."
                )
                # Return empty list — main loop will re-scan check_page
                return current_end, [], False

        # Reached end of document
        return total_pages, [], False

    # ==================================================================
    # Boundary check — semantic continuation + new section identification
    # ==================================================================

    def _check_boundary(
        self,
        window_pages: List[Dict],
        section_name: str,
        section_type: str,
        start_page: int,
        check_page: int,
    ) -> Dict:
        """
        Send the accumulative window to the LLM and ask about the
        LAST page:
          1. Is the content still about the same topic?
          2. Does a different topic begin? If so, WHAT is it?

        Returns full result dict with:
          - has_current_section_content: bool
          - new_section_starts: bool
          - new_sections: list of {section_name, section_type, page_type}
          - reason: str
        """
        images = prepare_images_for_bedrock(window_pages)

        prompt = BOUNDARY_CHECK_PROMPT.format(
            n_images=len(images),
            doc_type=get_document_type_name(),
            section_name=section_name,
            section_type=section_type,
            start_page=start_page,
            check_page=check_page,
            section_types=", ".join(self.section_definitions.keys()),
        )

        if section_type == "Table of Contents":
            prompt = TOC_BOUNDARY_CHECK_PROMPT.format(
                n_images=len(images),
                doc_type=get_document_type_name(),
                section_name=section_name,
                section_type=section_type,
                start_page=start_page,
                check_page=check_page,
                section_types=", ".join(self.section_definitions.keys()),
            )

        try:
            resp = invoke_multimodal(
                images=images, prompt=prompt, max_tokens=256,
            )
            data = self._parse_json(resp)

            logger.debug(
                f"  Page {check_page}: "
                f"content={data.get('has_current_section_content')} "
                f"new={data.get('new_section_starts')} "
                f"({data.get('reason', '')})"
            )
            return data

        except Exception as e:
            logger.warning(
                f"Boundary check failed page {check_page}: {e}"
            )
            return {
                "has_current_section_content": True,
                "new_section_starts": False,
                "new_sections": [],
            }

    # ==================================================================
    # Post-interruption continuation check
    # ==================================================================

    def _check_post_interruption_continuation(
        self,
        pages_data: List[Dict],
        prev_section: Dict,
        post_page: int,
        interruption_type: str,
        int_start: int,
        int_end: int,
    ) -> bool:
        """
        After a TOC/cover/blank interruption, check if the page right
        after it continues the section that was active before the
        interruption.

        Shows the LLM two pages:
          1. The last page of the previous section (origin context)
          2. The page right after the interruption (check page)

        Returns True if the post-interruption page continues the
        previous section.
        """
        origin_page = prev_section["end_page"]
        window = [
            pages_data[origin_page - 1],
            pages_data[post_page - 1],
        ]
        images = prepare_images_for_bedrock(window)

        prompt = POST_INTERRUPTION_CHECK_PROMPT.format(
            doc_type=get_document_type_name(),
            section_name=prev_section["section_name"],
            origin_page=origin_page,
            check_page=post_page,
            interruption_type=interruption_type,
            int_start=int_start,
            int_end=int_end,
        )

        try:
            resp = invoke_multimodal(
                images=images, prompt=prompt, max_tokens=128,
            )
            data = self._parse_json(resp)
            continues = data.get("continues_previous_section", False)

            logger.debug(
                f"  Post-interruption check page {post_page}: "
                f"continues={continues} ({data.get('reason', '')})"
            )
            return continues

        except Exception as e:
            logger.warning(
                f"Post-interruption check failed page {post_page}: {e}"
            )
            return False

    # ==================================================================
    # Merge adjacent same-type sections
    # ==================================================================

    def _merge_adjacent(
        self, sections: List[Dict], document_id: str
    ) -> List[Dict]:
        """Merge consecutive sections with the same type/name."""
        if len(sections) <= 1:
            return sections

        merge_rules = {r["section_type"]: r for r in get_merge_rules()}
        merged = [sections[0]]

        for nxt in sections[1:]:
            cur = merged[-1]
            adjacent = (
                cur["end_page"] is not None
                and nxt["start_page"] <= cur["end_page"] + 1
            )
            if adjacent and self._should_merge(cur, nxt, merge_rules):
                cur["end_page"] = max(
                    cur["end_page"] or 0, nxt["end_page"] or 0
                )
            else:
                merged.append(nxt)

        if len(merged) != len(sections):
            logger.info(
                f"[{document_id}] Merged {len(sections)} -> "
                f"{len(merged)}"
            )
        return merged

    @staticmethod
    def _should_merge(cur, nxt, rules):
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
    # Full coverage — fill gaps
    # ==================================================================

    def _ensure_full_coverage(
        self, sections: List[Dict], total_pages: int, document_id: str
    ) -> List[Dict]:
        """
        Fill any gaps between detected sections so every page is covered.
        """
        if not sections:
            return sections

        sections.sort(key=lambda s: s["start_page"])
        filled = []
        expected = 1

        for sec in sections:
            sp = sec["start_page"]

            if sp < expected:
                sp = expected

            if sp > expected:
                gap = {
                    "section_type": "unhandled_content",
                    "section_name": "Unhandled Content",
                    "start_page": expected,
                    "end_page": sp - 1,
                    "confidence": 0.5,
                    "_source": "gap_fill",
                }
                filled.append(gap)
                logger.info(
                    f"[{document_id}] Gap: pp {expected}-{sp - 1}"
                )

            filled.append(sec)
            ep = sec["end_page"]
            expected = (
                (ep + 1) if ep is not None
                else sec["start_page"] + 1
            )

        if expected <= total_pages:
            last = filled[-1]
            if last["section_type"] in (
                "back_matter", "unhandled_content"
            ):
                last["end_page"] = total_pages
            else:
                filled.append({
                    "section_type": "back_matter",
                    "section_name": "Back Matter",
                    "start_page": expected,
                    "end_page": total_pages,
                    "confidence": 0.7,
                    "_source": "trailing_fill",
                })
                logger.info(
                    f"[{document_id}] Trailing: pp {expected}-"
                    f"{total_pages}"
                )

        return filled

    # ==================================================================
    # JSON parsing
    # ==================================================================

    @staticmethod
    def _parse_json(response: str) -> Dict:
        """Parse JSON from LLM response, handling markdown fences."""
        response = response.strip()

        for pfx in ("```json", "```"):
            if response.startswith(pfx):
                response = response[len(pfx):]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        m = re.search(r"\{[\s\S]*\}", response)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        depth = 0
        start = None
        for i, ch in enumerate(response):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(response[start:i + 1])
                    except json.JSONDecodeError:
                        pass
                    start = None

        logger.error(f"JSON parse failed: {response[:200]}")
        return {}