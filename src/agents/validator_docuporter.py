"""
Stage 3: Validation & Aggregation Agent - Fully config-driven.
Validates and combines section JSONs into final document.
Uses assembly_order, structure_types, name_mapping from config.json.
No hardcoded section types.
"""
import copy
from typing import Dict, List, Any, Tuple

from config.settings import CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD
from config.config_loader import (
    get_assembly_order,
    get_object_section_types,
    get_array_section_types,
    get_section_name_mapping,
    get_section_schemas,
    get_empty_array_sections,
)
from src.utils import setup_logger, StorageManager

logger = setup_logger("validator_docuporter")


def clean_empty_fields(data: Any) -> Any:
    """Recursively remove empty fields from extracted data."""
    if isinstance(data, dict):
        has_content = False
        cleaned = {}
        for k, v in data.items():
            if isinstance(v, str):
                if v:
                    has_content = True
                cleaned[k] = v
            else:
                cv = clean_empty_fields(v)
                if cv:
                    has_content = True
                cleaned[k] = cv
        if not has_content and set(data.keys()).issubset({"text", "image"}):
            return {}
        return cleaned if has_content else {}
    elif isinstance(data, list):
        return [ci for ci in (clean_empty_fields(i) for i in data) if ci]
    return data


def format_section_for_docuporter(data: Any, section_type: str) -> Any:
    """Format extracted data for final output. Pass-through with cleaning."""
    return clean_empty_fields(data) if data else data


class ValidationAgentDocuPorter:
    """Validate and combine section JSONs. Fully config-driven."""

    def __init__(self):
        self.storage = StorageManager()
        self._object_types = set(get_object_section_types())
        self._array_types = set(get_array_section_types())
        self._name_mapping = get_section_name_mapping()
        self._assembly_order = get_assembly_order()
        self._schemas = get_section_schemas()
        self._empty_array_sections = set(get_empty_array_sections())

    def validate_and_combine(
        self,
        document_header: Dict,
        section_jsons: List[Dict],
        document_metadata: Dict,
        document_id: str,
    ) -> Tuple[Dict, Dict]:
        logger.info(f"[{document_id}] Combining document with {len(section_jsons)} sections")

        document_json = {"document_id": document_id, "document_header": document_header}

        # Populate header sections list
        section_names = []
        for s in section_jsons:
            name = s.get("section_name", "Unknown")
            if name not in section_names:
                section_names.append(name)
        document_json["document_header"]["sections"] = section_names

        # Bucket each section by type
        sections_by_type: Dict[str, Any] = {}
        unhandled_content = []

        for section in section_jsons:
            stype = section.get("_metadata", {}).get("section_type", "unhandled_content")
            sdata = section.get("data", [])

            # Apply name mapping from config
            for alias, canonical in self._name_mapping.items():
                if stype == alias:
                    stype = canonical

            formatted = format_section_for_docuporter(sdata, stype)

            if stype in self._object_types:
                # Object sections: merge dicts
                if stype not in sections_by_type:
                    sections_by_type[stype] = formatted
                elif isinstance(formatted, dict):
                    sections_by_type[stype].update(formatted)
            elif stype in self._array_types:
                # Array sections: extend lists
                sections_by_type.setdefault(stype, [])
                if isinstance(formatted, list):
                    sections_by_type[stype].extend(formatted)
                else:
                    sections_by_type[stype].append(formatted)
            elif stype == "unhandled_content":
                item = {
                    "section": section.get("section_name", "Unknown"),
                    "text": "",
                    "image": "",
                }
                if isinstance(sdata, dict):
                    for v in sdata.values():
                        if isinstance(v, str) and v:
                            item["text"] = v
                            break
                unhandled_content.append(item)
            else:
                # Unknown type: treat as array
                sections_by_type.setdefault(stype, [])
                if isinstance(formatted, list):
                    sections_by_type[stype].extend(formatted)
                elif formatted:
                    sections_by_type[stype].append(formatted)

        # Assemble in config-defined order
        for stype in self._assembly_order:
            if stype in sections_by_type:
                document_json[stype] = sections_by_type[stype]
            elif stype == "unhandled_content":
                document_json["unhandled_content"] = unhandled_content
            else:
                document_json[stype] = self._get_empty_section(stype)

        # Add any sections not in assembly_order
        for stype, sdata in sections_by_type.items():
            if stype not in document_json:
                document_json[stype] = sdata

        # Confidence
        confidences = [s.get("_metadata", {}).get("confidence", 0.5) for s in section_jsons]
        avg = sum(confidences) / len(confidences) if confidences else 0.0
        needs_review = avg < CONFIDENCE_THRESHOLD or any(c < LOW_CONFIDENCE_THRESHOLD for c in confidences)

        document_metadata.update({
            "confidence_score": avg,
            "needs_review": needs_review,
            "section_count": len(section_jsons),
        })

        logger.info(f"[{document_id}] Combined (confidence: {avg:.2f}, review: {needs_review})")
        self.storage.save_final_json(document_id, document_json)
        return document_json, document_metadata

    def _get_empty_section(self, section_type: str) -> Any:
        """Return an empty structure for a section type, based on config."""
        # Check if this section should be an empty array when children empty
        if section_type in self._empty_array_sections:
            return []

        # Use schema to determine structure
        schema = self._schemas.get(section_type)
        if schema is not None:
            if isinstance(schema, list):
                return []
            elif isinstance(schema, dict):
                return {}

        # Fallback based on structure_types
        if section_type in self._object_types:
            return {}
        if section_type in self._array_types:
            return []
        return {}
