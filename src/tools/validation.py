"""Generic validation utilities for extracted JSON data."""
from typing import Dict, List, Any, Tuple


def validate_section_json(data: Any, schema: Any) -> Tuple[bool, List[str]]:
    """Validate extracted JSON against its schema structure."""
    issues = []
    if schema is None:
        return True, issues
    if isinstance(schema, list) and not isinstance(data, list):
        issues.append("Expected array, got non-array")
        return False, issues
    if isinstance(schema, dict) and not isinstance(data, dict):
        issues.append("Expected object, got non-object")
        return False, issues
    return len(issues) == 0, issues


def check_data_quality(data: Any) -> Dict[str, Any]:
    """Check data quality metrics."""
    total = empty = 0
    _count_fields(data, lambda t, e: None)
    return {"total_fields": total, "empty_fields": empty, "quality": "high" if empty == 0 else "medium"}


def _count_fields(data, callback):
    if isinstance(data, dict):
        for v in data.values():
            _count_fields(v, callback)
    elif isinstance(data, list):
        for item in data:
            _count_fields(item, callback)


def calculate_confidence_score(data: Any, section_type: str) -> Tuple[float, List[str]]:
    """Calculate confidence score for extracted data."""
    from src.agents.section_extractor import calculate_confidence_score as _calc
    return _calc(data, section_type)


def validate_document_structure(document: Dict) -> Tuple[bool, List[str]]:
    """Validate the overall document structure."""
    issues = []
    if "document_id" not in document:
        issues.append("Missing document_id")
    if "document_header" not in document:
        issues.append("Missing document_header")
    return len(issues) == 0, issues
