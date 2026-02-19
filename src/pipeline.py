"""
PDF-to-JSON Pipeline.
All configuration from config.json -- no hardcoded values.

Supports precomputed section detection results via the
`precomputed_sections` parameter to skip the detection stage.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import (
    PARALLEL, MAX_WORKERS, REVIEW_ENABLED,
)
from config.schemas_docuporter import get_section_schema
from src.agents import SectionDetectionAgent, SectionExtractionAgent
from src.agents import ValidationAgentDocuPorter, ReviewAgent
from src.tools.bedrock_vision import prepare_images_for_bedrock
from src.utils import StorageManager, setup_logger
from src.utils.pdf_processor import extract_pages

logger = setup_logger("pipeline")


def process_document(
    pdf_path: str,
    precomputed_sections: Optional[List[Dict]] = None,
) -> Optional[Dict]:
    """
    Full pipeline: PDF → detect sections → extract → review → validate.

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
    logger.info("=" * 60)

    try:
        # ── 1. PDF to page images ──────────────────────────────
        logger.info("STAGE 1: Extracting PDF pages...")
        pages_data = extract_pages(str(pdf_path))
        logger.info(f"  {len(pages_data)} pages extracted")

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

        # ── 3. Extract sections ────────────────────────────────
        if PARALLEL:
            logger.info("STAGE 3: Extracting sections (PARALLEL)...")
            section_jsons = _extract_parallel(
                sections, pages_data, document_id
            )
        else:
            logger.info("STAGE 3: Extracting sections (SEQUENTIAL)...")
            section_jsons = _extract_sequential(
                sections, pages_data, document_id
            )
        logger.info(f"  {len(section_jsons)} sections extracted")

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
        document_json, metadata = validator.validate_and_combine(
            None, section_jsons, metadata, document_id
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

def _extract_sequential(
    sections: List[Dict],
    pages_data: List[Dict],
    document_id: str,
) -> List[Dict]:
    """Extract sections one at a time."""
    results = []
    for i, section in enumerate(sections):
        next_name = (
            sections[i + 1]["section_name"]
            if i + 1 < len(sections) else "END"
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
    return results


def _extract_parallel(
    sections: List[Dict],
    pages_data: List[Dict],
    document_id: str,
) -> List[Dict]:
    """Extract sections in parallel."""
    results = [None] * len(sections)

    def _extract_one(idx, section):
        next_name = (
            sections[idx + 1]["section_name"]
            if idx + 1 < len(sections) else "END"
        )
        schema = get_section_schema(section["section_type"])
        agent = SectionExtractionAgent(schema)
        section_pages = pages_data[
            section["start_page"] - 1 : section["end_page"]
        ]
        return idx, agent.extract_section(
            section_pages, section, next_name, document_id
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_extract_one, i, s): i
            for i, s in enumerate(sections)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return [r for r in results if r is not None]