"""
Validation tools for JSON schema and data quality checks.
"""
import json
from typing import Dict, Any, List, Tuple
from strands import tool

from config.schemas_docuporter import get_section_schema
from src.utils.logger import setup_logger

logger = setup_logger('validation_tools')


@tool
def validate_section_json(
    section_json: str,
    section_type: str
) -> Dict[str, Any]:
    """
    Validate section JSON against schema.
    
    Args:
        section_json: JSON string to validate
        section_type: Type of section for schema lookup
    
    Returns:
        Dict with validation results
    """
    try:
        parsed = json.loads(section_json)
        schema = get_section_schema(section_type)
        
        # Get required fields
        required_fields = schema.get('required', [])
        missing_fields = [
            f for f in required_fields 
            if f not in parsed
        ]
        
        # Check field types (basic validation)
        type_errors = []
        properties = schema.get('properties', {})
        
        for field, field_schema in properties.items():
            if field in parsed:
                expected_type = field_schema.get('type')
                actual_value = parsed[field]
                
                # Basic type checking
                if not _check_type(actual_value, expected_type):
                    type_errors.append({
                        'field': field,
                        'expected': expected_type,
                        'actual': type(actual_value).__name__
                    })
        
        is_valid = len(missing_fields) == 0 and len(type_errors) == 0
        
        return {
            'valid': is_valid,
            'missing_fields': missing_fields,
            'type_errors': type_errors,
            'data': parsed
        }
        
    except json.JSONDecodeError as e:
        return {
            'valid': False,
            'error': f'JSON parsing error: {str(e)}',
            'data': None
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'data': None
        }


def _check_type(value: Any, expected_type: Any) -> bool:
    """
    Check if value matches expected type.
    
    Args:
        value: Value to check
        expected_type: Expected type (string or list)
    
    Returns:
        True if type matches
    """
    if isinstance(expected_type, list):
        # Multiple types allowed
        return any(_check_single_type(value, t) for t in expected_type)
    else:
        return _check_single_type(value, expected_type)


def _check_single_type(value: Any, expected_type: str) -> bool:
    """Check if value matches a single expected type."""
    type_map = {
        'string': str,
        'integer': int,
        'number': (int, float),
        'boolean': bool,
        'array': list,
        'object': dict,
        'null': type(None)
    }
    
    expected_python_type = type_map.get(expected_type)
    
    if expected_python_type is None:
        return True  # Unknown type, allow
    
    return isinstance(value, expected_python_type)


def calculate_confidence_score(
    section_json: Dict,
    section_type: str
) -> Tuple[float, List[str]]:
    """
    Calculate confidence score for extracted section.
    
    Args:
        section_json: Extracted section data
        section_type: Type of section
    
    Returns:
        Tuple of (confidence_score, issues)
    """
    score = 1.0
    issues = []
    
    schema = get_section_schema(section_type)
    required_fields = schema.get('required', [])
    properties = schema.get('properties', {})
    
    # Check required fields
    missing_count = sum(
        1 for field in required_fields 
        if field not in section_json or not section_json[field]
    )
    
    if missing_count > 0:
        penalty = (missing_count / len(required_fields)) * 0.3
        score -= penalty
        issues.append(
            f"{missing_count} required field(s) missing or empty"
        )
    
    # Check completeness of optional fields
    optional_fields = [
        f for f in properties 
        if f not in required_fields
    ]
    
    filled_optional = sum(
        1 for field in optional_fields
        if field in section_json and section_json[field]
    )
    
    if optional_fields:
        completeness = filled_optional / len(optional_fields)
        if completeness < 0.5:
            score -= 0.1
            issues.append("Low completeness of optional fields")
    
    # Check for placeholder or suspicious content
    for field, value in section_json.items():
        if isinstance(value, str):
            if value.lower() in ['n/a', 'none', 'unknown', '', 'null']:
                score -= 0.05
                issues.append(f"Placeholder value in field: {field}")
            
            # Check for extraction artifacts
            if '[IMAGE' in value or '[TABLE' in value:
                score -= 0.05
                issues.append(f"Extraction artifact in field: {field}")
    
    # Ensure score is between 0 and 1
    score = max(0.0, min(1.0, score))
    
    return score, issues


def validate_document_structure(document_json: Dict) -> Tuple[bool, List[str]]:
    """
    Validate overall document structure.
    
    Args:
        document_json: Complete document JSON
    
    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []
    
    # Check required top-level fields
    required = ['document_id', 'metadata', 'sections']
    for field in required:
        if field not in document_json:
            errors.append(f"Missing required field: {field}")
    
    # Validate sections
    if 'sections' in document_json:
        sections = document_json['sections']
        
        if not isinstance(sections, list):
            errors.append("'sections' must be an array")
        elif len(sections) == 0:
            errors.append("Document has no sections")
        else:
            # Check each section has metadata
            for i, section in enumerate(sections):
                if '_metadata' not in section:
                    errors.append(
                        f"Section {i} missing _metadata field"
                    )
    
    # Validate metadata
    if 'metadata' in document_json:
        metadata = document_json['metadata']
        
        if not isinstance(metadata, dict):
            errors.append("'metadata' must be an object")
        else:
            # Check for key metadata fields
            expected_metadata = [
                'total_pages', 'processing_timestamp'
            ]
            for field in expected_metadata:
                if field not in metadata:
                    errors.append(
                        f"Missing metadata field: {field}"
                    )
    
    is_valid = len(errors) == 0
    
    return is_valid, errors


@tool
def check_data_quality(data: Dict) -> Dict[str, Any]:
    """
    Perform quality checks on extracted data.
    
    Args:
        data: Extracted data to check
    
    Returns:
        Quality check results
    """
    issues = []
    warnings = []
    
    # Check for empty values
    empty_fields = _find_empty_fields(data)
    if empty_fields:
        issues.append(f"Empty fields found: {', '.join(empty_fields)}")
    
    # Check for suspicious patterns
    suspicious = _find_suspicious_content(data)
    if suspicious:
        warnings.extend(suspicious)
    
    # Calculate quality score
    quality_score = 1.0 - (len(issues) * 0.1) - (len(warnings) * 0.05)
    quality_score = max(0.0, min(1.0, quality_score))
    
    return {
        'quality_score': quality_score,
        'issues': issues,
        'warnings': warnings,
        'passed': len(issues) == 0
    }


def _find_empty_fields(data: Dict, prefix: str = '') -> List[str]:
    """Recursively find empty fields in nested dict."""
    empty = []
    
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            empty.extend(_find_empty_fields(value, full_key))
        elif isinstance(value, (list, str)) and len(value) == 0:
            empty.append(full_key)
        elif value is None:
            empty.append(full_key)
    
    return empty


def _find_suspicious_content(data: Dict) -> List[str]:
    """Find suspicious patterns in data."""
    warnings = []
    
    suspicious_patterns = [
        'lorem ipsum', '[insert', 'todo', 'tbd',
        'xxxx', '????', 'error', 'failed'
    ]
    
    def check_value(value: Any, path: str):
        if isinstance(value, str):
            value_lower = value.lower()
            for pattern in suspicious_patterns:
                if pattern in value_lower:
                    warnings.append(
                        f"Suspicious content at {path}: contains '{pattern}'"
                    )
        elif isinstance(value, dict):
            for k, v in value.items():
                check_value(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                check_value(item, f"{path}[{i}]")
    
    for key, value in data.items():
        check_value(value, key)
    
    return warnings
