"""
PDF-to-JSON Pipeline.
All configuration from config.json -- no hardcoded values.

Supports precomputed section detection results via the
`precomputed_sections` parameter to skip the detection stage.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.settings import (
    REVIEW_ENABLED, TERM_MATCHING_ENABLED,
    EFFECTIVE_DATE_ENABLED, UOM_EXTRACTION_ENABLED,
)
from config.schemas_docuporter import get_section_schema
from src.agents import SectionDetectionAgent, SectionExtractionAgent
from src.agents import ValidationAgentDocuPorter, ReviewAgent
from src.agents.term_matcher import TermMatchingAgent
from src.agents.effective_date_extractor import EffectiveDateExtractor
from src.agents.uom_extractor import UOMExtractor
from src.utils import StorageManager, setup_logger
from src.utils.pdf_processor import extract_pages
from src.agents.document_header_extractor import DocumentHeaderExtractor

logger = setup_logger("pipeline")

# Section types that should not be sent through full extraction.
# The header extractor handles the first page; TOC/blank pages have
# no meaningful structured content to extract.
_SKIP_SECTION_TYPES = {"front_matter"}


def process_document(
    pdf_path: str,
    precomputed_sections: Optional[List[Dict]] = None,
    page_range: Optional[Tuple[int, int]] = None,
) -> Optional[Dict]:
    """
    Full pipeline: PDF → detect sections → extract → term match → review → validate.

    Args:
        pdf_path: Path to the PDF file.
        precomputed_sections: If provided, skip section detection and
            use these sections directly. Must be a list of dicts, each
            with at least: section_type, section_name, start_page, end_page.

    Returns:
        Final document JSON, or None on failure.
    """
    pdf_path = Path(pdf_path)
    document_id = pdf_path.stem
    storage = StorageManager()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info(f"Processing: {pdf_path.name}")
    logger.info(f"Document ID: {document_id}")
    if page_range:
        logger.info(f"Page range: {page_range[0]}-{page_range[1]}")
    logger.info("=" * 60)

    try:
        # ── 1. PDF to page images ──────────────────────────────
        logger.info("STAGE 1: Extracting PDF pages...")
        all_pages_data = extract_pages(str(pdf_path))
        total_pdf_pages = len(all_pages_data)
        logger.info(f"  {total_pdf_pages} pages extracted from PDF")

        # Apply page range filter if specified
        if page_range is not None:
            range_start, range_end = page_range
            # Clamp to actual PDF page count
            range_start = max(1, range_start)
            range_end = min(range_end, total_pdf_pages)

            if range_start > total_pdf_pages:
                logger.error(
                    f"[{document_id}] Page range start ({range_start}) "
                    f"exceeds total pages ({total_pdf_pages})"
                )
                return None

            pages_data = all_pages_data[range_start - 1 : range_end]

            # Renumber pages to be 1-based within the range
            for i, page in enumerate(pages_data):
                page["_original_page_number"] = page["page_number"]
                page["page_number"] = i + 1

            logger.info(
                f"  Filtered to pages {range_start}-{range_end} "
                f"({len(pages_data)} pages)"
            )
        else:
            pages_data = all_pages_data

        # ── 2. Detect sections (or use precomputed) ───────────
        if precomputed_sections is not None:
            sections = precomputed_sections
            logger.info(
                f"STAGE 2: Section detection SKIPPED — "
                f"using {len(sections)} precomputed sections"
            )
            # Validate the precomputed sections have required fields
            for i, sec in enumerate(sections):
                for field in (
                    "section_type", "section_name",
                    "start_page", "end_page",
                ):
                    if field not in sec:
                        raise ValueError(
                            f"Precomputed section [{i}] missing "
                            f"required field: {field}"
                        )
            # Save for traceability
            storage.save_detection_result(document_id, sections)
        else:
            logger.info("STAGE 2: Detecting sections...")
            detector = SectionDetectionAgent()
            sections = detector.detect_sections(
                pages_data, document_id
            )
            if not sections:
                logger.error(
                    f"[{document_id}] No sections detected"
                )
                return None
            logger.info(f"  {len(sections)} sections detected")

        # Log section map
        for sec in sections:
            logger.info(
                f"  [{sec['section_type']}] "
                f"'{sec['section_name']}' "
                f"pp {sec['start_page']}-{sec['end_page']}"
            )

        header_extractor = DocumentHeaderExtractor()
        header = header_extractor.extract_header(pages_data[0], document_id)

        # ── 3. Extract sections (sequential, preserving page order) ──
        logger.info("STAGE 3: Extracting sections...")
        section_jsons = _extract_sequential(sections, pages_data, document_id)
        logger.info(f"  {len(section_jsons)} sections extracted")

        # ── 3.5 (optional) Term matching ───────────────────────
        term_matching_report = None
        if TERM_MATCHING_ENABLED:
            logger.info("STAGE 3.5: Term matching...")
            matcher = TermMatchingAgent()
            term_matching_report = matcher.match_terms(
                section_jsons, document_id
            )
        else:
            logger.info("STAGE 3.5: Term matching SKIPPED (disabled)")

        # ── 3.6 (optional) Effective date extraction ──────────
        effective_date_report = None
        if EFFECTIVE_DATE_ENABLED:
            logger.info("STAGE 3.6: Extracting effective date...")
            date_extractor = EffectiveDateExtractor()
            effective_date_report = date_extractor.extract_effective_date(
                section_jsons, document_id,
                document_header=header,
            )
        else:
            logger.info("STAGE 3.6: Effective date extraction SKIPPED (disabled)")

        # ── 3.7 (optional) Unit of measure extraction ─────────
        uom_report = None
        if UOM_EXTRACTION_ENABLED:
            logger.info("STAGE 3.7: Extracting units of measure...")
            uom_extractor = UOMExtractor()
            uom_report = uom_extractor.extract_uom(
                section_jsons, document_id,
            )
        else:
            logger.info("STAGE 3.7: UOM extraction SKIPPED (disabled)")

        # ── 4. Optional review ─────────────────────────────────
        if REVIEW_ENABLED:
            logger.info("STAGE 4: Reviewing...")
            reviewer = ReviewAgent()
            reviewer.review_document(
                section_jsons, document_id, pages_data
            )

        # ── 5. Validate and combine ───────────────────────────
        logger.info("STAGE 5: Validating and combining...")
        validator = ValidationAgentDocuPorter()
        metadata = {
            "source_file": str(pdf_path),
            "total_pages": len(pages_data),
        }

        # Include term matching summary in metadata if available
        if term_matching_report is not None:
            total_terms = len(term_matching_report.get("terms", {}))
            unmatched = len(term_matching_report.get("unmatched_terms", []))
            metadata["term_matching"] = {
                "total_terms": total_terms,
                "matched_terms": total_terms - unmatched,
                "unmatched_terms": term_matching_report.get(
                    "unmatched_terms", []
                ),
            }

        # Include effective date in metadata if available
        if effective_date_report is not None:
            primary = effective_date_report.get(
                "primary_effective_date", {}
            )
            metadata["effective_date"] = {
                "date": primary.get("date", ""),
                "normalised": primary.get("normalised", ""),
                "date_type": primary.get("date_type", ""),
                "confidence": primary.get("confidence", ""),
                "no_date_found": effective_date_report.get(
                    "no_date_found", True
                ),
            }

        # Include UOM extraction summary in metadata if available
        if uom_report is not None:
            uoms = uom_report.get("units_of_measure", [])
            metadata["uom_extraction"] = {
                "total_references": len(uoms),
                "distinct_units": uom_report.get("distinct_units", []),
                "no_uom_found": uom_report.get("no_uom_found", True),
            }

        document_json, metadata = validator.validate_and_combine(
            header, section_jsons, metadata, document_id
        )

        elapsed = time.time() - start_time
        logger.info(
            f"[{document_id}] Pipeline complete "
            f"({elapsed:.1f}s, confidence: "
            f"{metadata.get('confidence_score', 0):.2f})"
        )
        return document_json

    except Exception as e:
        logger.error(f"[{document_id}] Pipeline failed: {e}")
        raise


# ====================================================================
# Section extraction helpers
# ====================================================================

def _should_skip_section(section: Dict) -> bool:
    """
    Return True for section types that do not need LLM extraction.

    front_matter (cover pages, TOC pages) are handled by the document
    header extractor. Sending them through the full extraction agent
    wastes LLM calls and produces noise in the output.
    """
    return section.get("section_type") in _SKIP_SECTION_TYPES


def _extract_sequential(
    sections: List[Dict],
    pages_data: List[Dict],
    document_id: str,
) -> List[Dict]:
    """
    Extract sections one at a time, in ascending page order.

    Sections are explicitly sorted by start_page here so that the
    extraction order is guaranteed correct regardless of what the
    detector returns.  The resulting list is also sorted before
    returning so validate_and_combine always receives page-ordered input.
    """
    # Explicit sort — defence-in-depth, detector should already sort
    # but this guarantees order at the extraction boundary.
    sections_sorted = sorted(
        [s for s in sections if not _should_skip_section(s)],
        key=lambda s: s["start_page"]
    )

    logger.info(f"[{document_id}] Extraction order:")
    for s in sections_sorted:
        logger.info(
            f"  pp {s['start_page']}-{s['end_page']} "
            f"[{s['section_type']}] '{s['section_name']}' "
        )

    results = []
    for i, section in enumerate(sections_sorted):
        next_name = (
            sections_sorted[i + 1]["section_name"]
            if i + 1 < len(sections_sorted) else "END"
        )
        schema = get_section_schema(section["section_type"])
        agent = SectionExtractionAgent(schema)
        section_pages = pages_data[
            section["start_page"] - 1 : section["end_page"]
        ]
        result = agent.extract_section(
            section_pages, section, next_name, document_id
        )
        results.append(result)

    # Sort results by start_page as a final safety net
    results.sort(key=lambda r: r.get("page_range", [0, 0])[0])
    return results