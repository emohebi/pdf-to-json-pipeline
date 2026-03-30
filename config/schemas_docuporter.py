"""
Document and section schemas - loaded entirely from config.json.
No hardcoded section types or field definitions.
"""
import copy
from typing import Dict, Any
from config.config_loader import (
    get_section_definitions, get_section_schemas,
    get_document_header_fields, get_section_name_mapping,
    get_empty_array_sections,
)

_defs = get_section_definitions()
_schemas = get_section_schemas()
_header_fields = get_document_header_fields()

SECTION_DEFINITIONS: Dict[str, str] = _defs
SECTION_SCHEMAS: Dict[str, Any] = _schemas
DOCUMENT_HEADER_SCHEMA = {f: {"orig_text": "", "text": ""} for f in _header_fields}
DOCUMENT_HEADER_SCHEMA["sections"] = []

DOCUMENT_SCHEMA: Dict[str, Any] = {"document_id": "", "document_header": copy.deepcopy(DOCUMENT_HEADER_SCHEMA)}
for st in SECTION_DEFINITIONS:
    if st in SECTION_SCHEMAS:
        DOCUMENT_SCHEMA[st] = copy.deepcopy(SECTION_SCHEMAS[st])


def get_section_schema(section_type: str) -> Any:
    mapping = get_section_name_mapping()
    section_type = mapping.get(section_type, section_type)
    return copy.deepcopy(SECTION_SCHEMAS.get(section_type, SECTION_SCHEMAS.get("unhandled_content", [])))

def get_all_section_types() -> list:
    return list(SECTION_DEFINITIONS.keys())

def validate_section_type(section_type: str) -> bool:
    return section_type in SECTION_DEFINITIONS

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

        # For content blocks, keep them if they have type
        if "type" in data:
            return cleaned

        if not has_content and set(data.keys()).issubset({"text", "image", "orig_text", "orig_image"}):
            return {}
        return cleaned if has_content else {}

    elif isinstance(data, list):
        result = []
        for item in data:
            ci = clean_empty_fields(item)
            if ci is not None:
                # Keep content blocks even if some fields are empty
                if isinstance(ci, dict) and "type" in ci:
                    result.append(ci)
                elif ci:
                    result.append(ci)
        return result

    return data
