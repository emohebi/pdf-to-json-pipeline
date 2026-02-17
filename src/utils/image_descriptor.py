"""Image description utilities for section images."""
from typing import List, Dict
from config.config_loader import get_image_context_hint
from src.tools.llm_provider import invoke_vision
from src.utils.logger import setup_logger

logger = setup_logger("image_descriptor")


def describe_section_images(
    section_type: str,
    images: List[Dict],
    max_description_len: int = 30,
) -> List[Dict]:
    """Generate descriptions for images in a section."""
    context_hint = get_image_context_hint(section_type)
    described = []
    for img in images:
        try:
            import base64
            img_b64 = base64.b64encode(img["image_bytes"]).decode("utf-8")
            prompt = f"{context_hint}\nProvide a SPECIFIC description (max {max_description_len} words).\nReturn ONLY the description:"
            description = invoke_vision(image_data=img_b64, prompt=prompt, max_tokens=256)
            img["description"] = description.strip()
        except Exception as e:
            logger.warning(f"Failed to describe image: {e}")
            img["description"] = "Image"
        described.append(img)
    return described


def format_image_mapping_for_prompt(image_mappings: List[Dict]) -> str:
    """Format image mappings into prompt text."""
    if not image_mappings:
        return ""
    lines = ["\n\nIMAGES IN THIS SECTION:", "-" * 60]
    for img in image_mappings:
        idx = img.get("sorted_index", img.get("index", 0))
        lines.append(f"[{idx}] Page {img['page']}, {img['grid']} (Y:{img['y_percent']:.0f}%)")
        lines.append(f"    Description: {img['description'][:50]}...")
        lines.append(f"    PATH: {img['path']}")
        lines.append("")
    lines += ["-" * 60, "Match images by PAGE and POSITION, then copy exact PATH."]
    return "\n".join(lines)
