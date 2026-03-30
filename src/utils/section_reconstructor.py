"""
Shared helper for standalone scripts that need to reconstruct
section data from the final pipeline output JSON.

The final output now uses a flat 'sections' array:
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
"""
from typing import Dict, List, Any


def reconstruct_sections_from_document(document: Dict) -> List[Dict]:
    """
    Convert a final pipeline document JSON into the list-of-dicts
    format that agents expect (section_name, data, _metadata).

    Handles both the new flat format and the old bucketed format.
    """
    # New format: flat sections array
    flat_sections = document.get("sections")
    if isinstance(flat_sections, list) and flat_sections:
        return _from_flat_sections(flat_sections)

    # Old format: sections bucketed by type
    return _from_bucketed_sections(document)


def _from_flat_sections(flat_sections: List[Dict]) -> List[Dict]:
    """Convert new flat sections array to agent format."""
    result = []
    for entry in flat_sections:
        if not isinstance(entry, dict):
            continue
        name = entry.get("section_name", entry.get("heading", "Unknown"))
        stype = entry.get("section_type", "section")

        # The 'data' for agents is the content structure
        data = {
            "heading": entry.get("heading", name),
            "heading_level": entry.get("heading_level", ""),
            "content": entry.get("content", []),
        }

        result.append({
            "section_name": name,
            "data": data,
            "_metadata": {"section_type": stype},
        })
    return result


def _from_bucketed_sections(document: Dict) -> List[Dict]:
    """Convert old bucketed format to agent format (backward compat)."""
    sections: List[Dict] = []
    skip_keys = {"document_id", "document_header"}
    header = document.get("document_header", {})
    header_section_names = header.get("sections", [])

    for key, value in document.items():
        if key in skip_keys:
            continue
        if isinstance(value, list):
            for item in value:
                name = _derive_name(item, key)
                sections.append({
                    "section_name": name,
                    "data": item,
                    "_metadata": {"section_type": key},
                })
        elif isinstance(value, dict) and value:
            name = _derive_name(value, key)
            sections.append({
                "section_name": name,
                "data": value,
                "_metadata": {"section_type": key},
            })

    if header_section_names and sections:
        name_order = {n: i for i, n in enumerate(header_section_names)}
        sections.sort(key=lambda s: name_order.get(s["section_name"], 9999))

    return sections


def _derive_name(data: Any, fallback_type: str) -> str:
    if isinstance(data, dict):
        for key in ("heading", "section_name", "section", "caption"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return fallback_type.replace("_", " ").title()
