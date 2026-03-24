"""
Stage 3: Validation & Aggregation Agent.

Combines extracted section JSONs into the final document.
Preserves strict page order — sections appear in the output
in the same order they appear in the PDF.

Output structure:
{
  "document_id": "...",
  "document_header": {...},
  "sections": [
    {
      "section_name": "DEFINITIONS",
      "section_type": "section",
      "page_range": [3, 8],
      "heading": "DEFINITIONS",
      "heading_level": "1",
      "content": [...]
    },
    ...
  ]
}

The sections array is flat and ordered by page number.
This works correctly for:
  - Single PDFs
  - Concatenated PDFs with multiple cover pages
  - Documents with interleaved front/back matter
"""
from typing import Dict, List, Any, Tuple

from config.settings import CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD
from src.utils import setup_logger, StorageManager

logger = setup_logger("validator_docuporter")


def clean_empty_fields(data: Any) -> Any:
    """Recursively remove empty fields from extracted data."""
    if isinstance(data, dict):
        cleaned = {}
        has_content = False
        for k, v in data.items():
            if isinstance(v, str):
                if v:
                    has_content = True
                cleaned[k] = v
            elif isinstance(v, list):
                cv = clean_empty_fields(v)
                if cv:
                    has_content = True
                cleaned[k] = cv
            elif isinstance(v, dict):
                cv = clean_empty_fields(v)
                if cv:
                    has_content = True
                cleaned[k] = cv
            else:
                cleaned[k] = v
        if "type" in data:
            return cleaned
        if not has_content and set(data.keys()).issubset({"text", "image"}):
            return {}
        return cleaned if has_content else {}
    elif isinstance(data, list):
        result = []
        for item in data:
            ci = clean_empty_fields(item)
            if ci is not None:
                if isinstance(ci, dict) and "type" in ci:
                    result.append(ci)
                elif ci:
                    result.append(ci)
        return result
    return data


class ValidationAgentDocuPorter:
    """Validate and combine section JSONs in strict page order."""

    def __init__(self):
        self.storage = StorageManager()

    def validate_and_combine(
        self,
        document_header: Dict,
        section_jsons: List[Dict],
        document_metadata: Dict,
        document_id: str,
    ) -> Tuple[Dict, Dict]:
        logger.info(f"[{document_id}] Combining {len(section_jsons)} sections")

        # Sort by start page — this is the canonical document order
        section_jsons = sorted(
            section_jsons,
            key=lambda s: s.get("page_range", [0, 0])[0]
        )

        # Build the flat ordered sections array
        sections = []
        section_names = []

        for s in section_jsons:
            name = s.get("section_name", "Unknown")
            stype = s.get("_metadata", {}).get("section_type", "section")
            page_range = s.get("page_range", [0, 0])
            data = s.get("data", {})

            # Clean empty fields
            if isinstance(data, dict):
                data = clean_empty_fields(data)

            # Build the section entry — data fields merged at top level
            entry = {
                "section_name": name,
                "section_type": stype,
                "page_range": page_range,
            }

            # Merge extracted data into the entry
            if isinstance(data, dict):
                entry["heading"] = data.get("heading", name)
                entry["heading_level"] = data.get("heading_level", "")
                entry["content"] = data.get("content", [])
            elif isinstance(data, list):
                # Array-type sections (e.g. unhandled_content)
                entry["heading"] = name
                entry["heading_level"] = ""
                entry["content"] = data
            else:
                entry["heading"] = name
                entry["heading_level"] = ""
                entry["content"] = []

            sections.append(entry)

            if name not in section_names:
                section_names.append(name)

        # Build document
        document_json = {
            "document_id": document_id,
            "document_header": document_header,
            "sections": sections,
        }
        document_json["document_header"]["sections"] = section_names

        # Confidence
        confidences = [
            s.get("_metadata", {}).get("confidence", 0.5)
            for s in section_jsons
        ]
        avg = sum(confidences) / len(confidences) if confidences else 0.0
        needs_review = (
            avg < CONFIDENCE_THRESHOLD
            or any(c < LOW_CONFIDENCE_THRESHOLD for c in confidences)
        )

        document_metadata.update({
            "confidence_score": avg,
            "needs_review": needs_review,
            "section_count": len(section_jsons),
        })

        logger.info(
            f"[{document_id}] Combined (confidence: {avg:.2f}, "
            f"review: {needs_review})"
        )
        self.storage.save_final_json(document_id, document_json)
        return document_json, document_metadata
