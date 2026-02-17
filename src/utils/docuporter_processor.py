"""DocuPorter format utilities."""
from typing import Any


def format_section_for_docuporter(data: Any, section_type: str) -> Any:
    """Format extracted data for DocuPorter output."""
    return clean_empty_fields(data) if data else data


def clean_empty_fields(data: Any) -> Any:
    """Recursively remove empty fields."""
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
        if not has_content and set(data.keys()).issubset({"text", "image", "orig_text", "orig_image"}):
            return {}
        return cleaned if has_content else {}
    elif isinstance(data, list):
        return [ci for ci in (clean_empty_fields(i) for i in data) if ci]
    return data
