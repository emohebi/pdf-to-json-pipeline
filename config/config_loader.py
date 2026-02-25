"""
Configuration loader - reads config.json and provides typed access.
Prompt arrays are joined with newlines. Template variables use {var} syntax.
ALL domain knowledge lives in config.json -- this module is fully generic.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

_config_cache: Optional[Dict[str, Any]] = None
_config_path: Optional[str] = None


def load_config(config_path: str = None) -> Dict[str, Any]:
    global _config_cache, _config_path
    if config_path:
        _config_path = config_path
    if _config_cache is not None and (config_path is None or config_path == _config_path):
        return _config_cache

    search = []
    if config_path:
        search.append(Path(config_path))
    env_path = os.getenv("PIPELINE_CONFIG")
    if env_path:
        search.append(Path(env_path))
    base_dir = Path(__file__).resolve().parent.parent
    search += [base_dir / "config.json", Path.cwd() / "config.json"]

    for p in search:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                _config_cache = json.load(f)
            _config_path = str(p)
            return _config_cache
    raise FileNotFoundError(f"config.json not found. Searched: {[str(p) for p in search]}")


def reload_config(config_path: str = None) -> Dict[str, Any]:
    global _config_cache
    _config_cache = None
    return load_config(config_path)


# ---------------------------------------------------------------------------
# Top-level section accessors
# ---------------------------------------------------------------------------

def get_input_config() -> Dict[str, Any]:
    return load_config().get("INPUT", {})

def get_task_config() -> Dict[str, Any]:
    return load_config().get("TASK", {})

def get_output_config() -> Dict[str, Any]:
    return load_config().get("OUTPUT", {})

def get_provider_name() -> str:
    return get_task_config().get("provider", "aws_bedrock")

def get_provider_config() -> Dict[str, Any]:
    t = get_task_config()
    return t.get(t.get("provider", "aws_bedrock"), {})

def get_model_params() -> Dict[str, Any]:
    return get_task_config().get("model_params", {})

def get_processing_config() -> Dict[str, Any]:
    return get_task_config().get("processing", {})

def get_confidence_config() -> Dict[str, float]:
    return get_task_config().get("confidence", {"threshold": 0.85, "low_threshold": 0.70})


# ---------------------------------------------------------------------------
# Document type
# ---------------------------------------------------------------------------

def get_document_type_config() -> Dict[str, str]:
    return get_task_config().get("document_type", {})

def get_document_type_name() -> str:
    return get_document_type_config().get("name", "Document")

def get_document_type_description() -> str:
    return get_document_type_config().get("description", "")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def get_sections_config() -> Dict[str, Any]:
    return get_task_config().get("sections", {})

def get_section_definitions() -> Dict[str, str]:
    return get_sections_config().get("definitions", {})

def get_heading_aliases() -> Dict[str, str]:
    """Return heading text -> section_type mapping for detection prompt."""
    return get_sections_config().get("heading_aliases", {})

def get_section_schemas() -> Dict[str, Any]:
    return get_sections_config().get("schemas", {})

def get_assembly_order() -> List[str]:
    return get_sections_config().get("assembly_order", list(get_section_definitions().keys()))

def get_object_section_types() -> List[str]:
    return get_sections_config().get("structure_types", {}).get("object", [])

def get_array_section_types() -> List[str]:
    return get_sections_config().get("structure_types", {}).get("array", [])

def get_section_name_mapping() -> Dict[str, str]:
    return get_sections_config().get("name_mapping", {})

def get_merge_rules() -> List[Dict]:
    return get_sections_config().get("merge_rules", [])

def get_empty_array_sections() -> List[str]:
    return get_sections_config().get("empty_array_when_all_children_empty", [])


# ---------------------------------------------------------------------------
# Document header
# ---------------------------------------------------------------------------

def get_document_header_fields() -> list:
    return get_task_config().get("document_header", {}).get("fields", [])

def get_header_prompt() -> str:
    return join_prompt(get_task_config().get("document_header", {}).get("prompt", ""))


# ---------------------------------------------------------------------------
# Document classification (config-driven)
# ---------------------------------------------------------------------------

def get_document_classification_config() -> Dict[str, Any]:
    return get_task_config().get("document_classification", {})


# ---------------------------------------------------------------------------
# Term matching (config-driven, optional)
# ---------------------------------------------------------------------------

def get_term_matching_config() -> Dict[str, Any]:
    """Return the TASK.term_matching config block."""
    return get_task_config().get("term_matching", {})

def is_term_matching_enabled() -> bool:
    """Check if term matching is enabled in config."""
    cfg = get_term_matching_config()
    return cfg.get("enabled", False) and bool(cfg.get("terms"))


# ---------------------------------------------------------------------------
# Effective date extraction (config-driven, optional)
# ---------------------------------------------------------------------------

def get_effective_date_config() -> Dict[str, Any]:
    """Return the TASK.effective_date config block."""
    return get_task_config().get("effective_date", {})

def is_effective_date_enabled() -> bool:
    """Check if effective date extraction is enabled in config."""
    return get_effective_date_config().get("enabled", False)


# ---------------------------------------------------------------------------
# UOM extraction (config-driven, optional)
# ---------------------------------------------------------------------------

def get_uom_extraction_config() -> Dict[str, Any]:
    """Return the TASK.uom_extraction config block."""
    return get_task_config().get("uom_extraction", {})

def is_uom_extraction_enabled() -> bool:
    """Check if UOM extraction is enabled in config."""
    return get_uom_extraction_config().get("enabled", False)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def get_prompts_config() -> Dict[str, Any]:
    return get_task_config().get("prompts", {})

def get_post_processing_config() -> Dict[str, Any]:
    return get_task_config().get("post_processing", {})


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def join_prompt(prompt_or_lines: Union[str, List[str]]) -> str:
    """Join a prompt that may be a list of lines or already a string."""
    if isinstance(prompt_or_lines, list):
        return "\n".join(prompt_or_lines)
    return prompt_or_lines


def get_prompt(path: str, default: str = "") -> str:
    """
    Get a prompt by dot-separated path within TASK.prompts.
    E.g. 'extraction.system_preamble' or 'review_template'.
    Arrays are auto-joined with newlines.
    """
    prompts = get_prompts_config()
    parts = path.split(".")
    node = prompts
    for p in parts:
        if isinstance(node, dict) and p in node:
            node = node[p]
        else:
            return default
    return join_prompt(node)


def render_prompt(template_text: str, **kwargs) -> str:
    """
    Render a prompt template by substituting {var} placeholders.
    Uses str.format_map with a defaulting dict so missing keys don't crash.
    """
    class SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    return template_text.format_map(SafeDict(**kwargs))


# ---------------------------------------------------------------------------
# Detection prompt helpers
# ---------------------------------------------------------------------------

def get_detection_prompt_template() -> str:
    return get_prompt("detection.template")


def build_heading_alias_rules() -> str:
    """Build prompt text for heading aliases from config."""
    aliases = get_heading_aliases()
    if not aliases:
        return ""
    # Group by target section type
    grouped: Dict[str, List[str]] = {}
    for heading, section_type in aliases.items():
        grouped.setdefault(section_type, []).append(heading)

    rules = []
    for section_type, headings in grouped.items():
        heading_list = "', '".join(headings)
        rules.append(f"- Exception: '{heading_list}' headings should be reported as belonging to the '{section_type}' section")
    return "\n".join(rules)


# ---------------------------------------------------------------------------
# Extraction prompt helpers
# ---------------------------------------------------------------------------

def get_extraction_preamble() -> str:
    return get_prompt("extraction.system_preamble")


def get_extraction_general_rules() -> str:
    return get_prompt("extraction.general_rules")


def get_section_extraction_prompt(section_type: str) -> str:
    """
    Return the extraction prompt template for a section type.
    Checks section_overrides first, falls back to default_template.
    """
    ext = get_prompts_config().get("extraction", {})
    overrides = ext.get("section_overrides", {})
    if section_type in overrides:
        return join_prompt(overrides[section_type])
    return join_prompt(ext.get("default_template", ""))


# ---------------------------------------------------------------------------
# Image prompt helpers
# ---------------------------------------------------------------------------

def get_image_context_hint(section_type: str) -> str:
    hints = get_prompts_config().get("image_context_hints", {})
    return hints.get(section_type, hints.get("_default", "Describe this image briefly."))

def get_image_description_prompt() -> str:
    return get_prompt("image_description_prompt")
