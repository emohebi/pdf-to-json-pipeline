"""
PDF-to-JSON Pipeline.
All configuration from config.json -- no hardcoded values.

Produces output with sections in strict page order.
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
# front_matter (cover pages, TOC) is handled by the header extractor.
_SKIP_SECTION_TYPES = {"front_matter"}


def process_document(
    pdf_path: str,
    precomputed_sections: Optional[List[Dict]] = None,
    page_range: Optional[Tuple[int, int]] = None,
) -> Optional[Dict]:
    """
    Full pipeline: PDF -> detect sections -> extract -> validate.

    Returns final document JSON with sections in strict page order.
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
        # == 1. PDF to page images ==
        logger.info("STAGE 1: Extracting PDF pages...")
        all_pages_data = extract_pages(str(pdf_path))
        total_pdf_pages = len(all_pages_data)
        logger.info(f"  {total_pdf_pages} pages extracted from PDF")

        if page_range is not None:
            range_start, range_end = page_range
            range_start = max(1, range_start)
            range_end = min(range_end, total_pdf_pages)
            if range_start > total_pdf_pages:
                logger.error(
                    f"[{document_id}] Page range start ({range_start}) "
                    f"exceeds total pages ({total_pdf_pages})"
                )
                return None
            pages_data = all_pages_data[range_start - 1 : range_end]
            for i, page in enumerate(pages_data):
                page["_original_page_number"] = page["page_number"]
                page["page_number"] = i + 1
            logger.info(f"  Filtered to pages {range_start}-{range_end} ({len(pages_data)} pages)")
        else:
            pages_data = all_pages_data

        # == 2. Detect sections ==
        if precomputed_sections is not None:
            sections = precomputed_sections
            logger.info(f"STAGE 2: Using {len(sections)} precomputed sections")
            for i, sec in enumerate(sections):
                for field in ("section_type", "section_name", "start_page", "end_page"):
                    if field not in sec:
                        raise ValueError(f"Precomputed section [{i}] missing: {field}")
            storage.save_detection_result(document_id, sections)
        else:
            logger.info("STAGE 2: Detecting sections...")
            detector = SectionDetectionAgent()
            sections = detector.detect_sections(pages_data, document_id)
            if not sections:
                logger.error(f"[{document_id}] No sections detected")
                return None
            logger.info(f"  {len(sections)} sections detected")

        # Log section map in page order
        for sec in sections:
            logger.info(
                f"  [{sec['section_type']}] '{sec['section_name']}' "
                f"pp {sec['start_page']}-{sec['end_page']}"
            )

        # == 2.5 Extract document header from first page ==
        header_extractor = DocumentHeaderExtractor()
        header = header_extractor.extract_header(pages_data[0], document_id)

        # == 3. Extract sections in page order ==
        logger.info("STAGE 3: Extracting sections...")
        section_jsons = _extract_sequential(sections, pages_data, document_id)
        logger.info(f"  {len(section_jsons)} sections extracted")

        # == 3.5 Optional: Term matching ==
        term_matching_report = None
        if TERM_MATCHING_ENABLED:
            logger.info("STAGE 3.5: Term matching...")
            matcher = TermMatchingAgent()
            term_matching_report = matcher.match_terms(section_jsons, document_id)
        else:
            logger.info("STAGE 3.5: Term matching SKIPPED")

        # == 3.6 Optional: Effective date ==
        effective_date_report = None
        if EFFECTIVE_DATE_ENABLED:
            logger.info("STAGE 3.6: Extracting effective date...")
            date_extractor = EffectiveDateExtractor()
            effective_date_report = date_extractor.extract_effective_date(
                section_jsons, document_id, document_header=header,
            )
        else:
            logger.info("STAGE 3.6: Effective date SKIPPED")

        # == 3.7 Optional: UOM extraction ==
        uom_report = None
        if UOM_EXTRACTION_ENABLED:
            logger.info("STAGE 3.7: Extracting units of measure...")
            uom_extractor = UOMExtractor()
            uom_report = uom_extractor.extract_uom(section_jsons, document_id)
        else:
            logger.info("STAGE 3.7: UOM extraction SKIPPED")

        # == 4. Optional: Review ==
        if REVIEW_ENABLED:
            logger.info("STAGE 4: Reviewing...")
            reviewer = ReviewAgent()
            reviewer.review_document(section_jsons, document_id, pages_data)

        # == 5. Validate and combine ==
        logger.info("STAGE 5: Validating and combining...")
        validator = ValidationAgentDocuPorter()
        metadata = {
            "source_file": str(pdf_path),
            "total_pages": len(pages_data),
        }

        if term_matching_report is not None:
            total_terms = len(term_matching_report.get("terms", {}))
            unmatched = len(term_matching_report.get("unmatched_terms", []))
            metadata["term_matching"] = {
                "total_terms": total_terms,
                "matched_terms": total_terms - unmatched,
                "unmatched_terms": term_matching_report.get("unmatched_terms", []),
            }

        if effective_date_report is not None:
            primary = effective_date_report.get("primary_effective_date", {})
            metadata["effective_date"] = {
                "date": primary.get("date", ""),
                "normalised": primary.get("normalised", ""),
                "date_type": primary.get("date_type", ""),
                "confidence": primary.get("confidence", ""),
                "no_date_found": effective_date_report.get("no_date_found", True),
            }

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
            f"({elapsed:.1f}s, confidence: {metadata.get('confidence_score', 0):.2f})"
        )
        return document_json

    except Exception as e:
        logger.error(f"[{document_id}] Pipeline failed: {e}")
        raise


# ====================================================================
# Section extraction helpers
# ====================================================================

def _should_skip_section(section: Dict) -> bool:
    """Return True for section types that don't need LLM extraction."""
    return section.get("section_type") in _SKIP_SECTION_TYPES


def _extract_sequential(
    sections: List[Dict],
    pages_data: List[Dict],
    document_id: str,
) -> List[Dict]:
    """
    Extract sections in strict page order.

    Sections are sorted by start_page. Skipped sections (front_matter)
    are excluded from extraction but their page ranges are preserved
    in the detection result for traceability.
    """
    sections_sorted = sorted(
        [s for s in sections if not _should_skip_section(s)],
        key=lambda s: s["start_page"]
    )

    logger.info(f"[{document_id}] Extraction order ({len(sections_sorted)} sections):")
    for s in sections_sorted:
        logger.info(
            f"  pp {s['start_page']}-{s['end_page']} "
            f"[{s['section_type']}] '{s['section_name']}'"
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

    # Final sort — defense in depth
    results.sort(key=lambda r: r.get("page_range", [0, 0])[0])
    return results
