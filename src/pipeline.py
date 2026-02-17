"""
Pipeline orchestrator - fully config-driven.
No hardcoded section types or domain knowledge.
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import (
    PARALLEL, REVIEW_ENABLED, MAX_WORKERS, PROVIDER_NAME,
)
from config.config_loader import get_document_type_name
from config.schemas_docuporter import get_section_schema
from src.agents.section_detector import SectionDetectionAgent
from src.agents.section_extractor import SectionExtractionAgent
from src.agents.document_header_extractor import DocumentHeaderExtractor
from src.agents.validator_docuporter import ValidationAgentDocuPorter
from src.agents.review_agent import ReviewAgent
from src.utils.pdf_processor import extract_pages
from src.utils import setup_logger

logger = setup_logger("pipeline")


def process_document(pdf_path: str, document_id: str = None) -> Dict:
    """Process a single PDF document through the full pipeline."""
    pdf_path = Path(pdf_path)
    document_id = document_id or pdf_path.stem

    logger.info(f"[{document_id}] Starting pipeline")
    logger.info(f"  Provider: {PROVIDER_NAME}")
    logger.info(f"  Document type: {get_document_type_name()}")
    logger.info(f"  Parallel: {PARALLEL}, Review: {REVIEW_ENABLED}")

    # 1. Extract pages
    pages_data = extract_pages(str(pdf_path))
    logger.info(f"[{document_id}] Extracted {len(pages_data)} pages")

    # 2. Detect sections
    detector = SectionDetectionAgent()
    sections = detector.detect_sections(pages_data, document_id)
    if not sections:
        logger.error(f"[{document_id}] No sections detected")
        return None
    logger.info(f"[{document_id}] Detected {len(sections)} sections")

    # 3. Extract header
    header_extractor = DocumentHeaderExtractor()
    header = header_extractor.extract_header(pages_data[0], document_id)

    # 4. Extract sections
    if PARALLEL:
        section_jsons = _extract_parallel(sections, pages_data, document_id)
    else:
        section_jsons = _extract_sequential(sections, pages_data, document_id)

    # 5. Optional review
    if REVIEW_ENABLED:
        reviewer = ReviewAgent()
        reviewer.review_document(section_jsons, document_id, pages_data)

    # 6. Validate and combine
    validator = ValidationAgentDocuPorter()
    metadata = {"source_file": str(pdf_path), "total_pages": len(pages_data)}
    document_json, metadata = validator.validate_and_combine(
        header, section_jsons, metadata, document_id
    )

    logger.info(f"[{document_id}] Pipeline complete (confidence: {metadata.get('confidence_score', 0):.2f})")
    return document_json


def _extract_sequential(sections, pages_data, document_id) -> List[Dict]:
    results = []
    for i, section in enumerate(sections):
        next_name = sections[i + 1]["section_name"] if i + 1 < len(sections) else "END"
        schema = get_section_schema(section["section_type"])
        agent = SectionExtractionAgent(schema)
        section_pages = pages_data[section["start_page"] - 1 : section["end_page"]]
        result = agent.extract_section(section_pages, section, next_name, document_id)
        results.append(result)
    return results


def _extract_parallel(sections, pages_data, document_id) -> List[Dict]:
    results = [None] * len(sections)

    def _extract_one(idx, section):
        next_name = sections[idx + 1]["section_name"] if idx + 1 < len(sections) else "END"
        schema = get_section_schema(section["section_type"])
        agent = SectionExtractionAgent(schema)
        section_pages = pages_data[section["start_page"] - 1 : section["end_page"]]
        return idx, agent.extract_section(section_pages, section, next_name, document_id)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_extract_one, i, s): i for i, s in enumerate(sections)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return [r for r in results if r is not None]
