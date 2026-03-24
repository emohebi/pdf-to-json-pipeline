"""
Batch merge logic for the ordered content-block schema.

Each section is:
    {
      "heading": "...",
      "heading_level": "...",
      "content": [
        {"type": "paragraph", "text": "..."},
        {"type": "table", "caption": "...", "headers": [...], "rows": [...]},
        {"type": "subsection", "heading": "...", "heading_level": "...", "content": [...]}
      ]
    }

Because content is a single ordered array, merging batches is simple:
  1. Normalize each batch (fix rogue fields, flatten bad nesting).
  2. For each continuation batch, append or stitch its content blocks
     onto the merged result in order.
  3. Deduplicate as a safety net.
"""
import re
import copy
import logging
from typing import Dict, List, Any, Union, Optional

logger = logging.getLogger(__name__)


# ======================================================================
# Public API
# ======================================================================

def merge_batch_results(
    results: List[Union[Dict, List]],
    trailing_contexts: List[Dict[str, Any]],
    document_id: str,
) -> Union[Dict, List]:
    """Merge results from multiple extraction batches into one."""
    if not results:
        return {}

    all_lists = all(isinstance(r, list) for r in results)
    all_dicts = all(isinstance(r, dict) for r in results)

    if all_lists:
        merged: list = []
        for r in results:
            merged.extend(r)
        return merged

    if all_dicts:
        normalized = [normalize_batch_result(r) for r in results]
        merged_dict = _merge_section_batches(normalized, trailing_contexts)
        _deduplicate_content(merged_dict)
        logger.info(f"[{document_id}] Merged {len(results)} batches")
        return merged_dict

    # Mixed
    merged_list: list = []
    for r in results:
        if isinstance(r, list):
            merged_list.extend(r)
        else:
            merged_list.append(r)
    return merged_list


# ======================================================================
# Normalization
# ======================================================================

def normalize_batch_result(result: Any) -> Any:
    """
    Clean up a batch result before merging.

    Fixes:
      - Rogue 'text'/'section'/'body' fields at wrong level
      - Incorrectly nested sibling subsections in content array
      - Ensures content is a list
    """
    if isinstance(result, list):
        return [_normalize_block(item) for item in result]
    if isinstance(result, dict):
        return _normalize_section(result)
    return result


def _normalize_section(d: Dict) -> Dict:
    """Normalize a top-level section dict."""
    out = {}
    out["heading"] = d.get("heading", "")
    out["heading_level"] = d.get("heading_level", "")

    content = d.get("content", [])
    if not isinstance(content, list):
        content = [content] if content else []

    # If the LLM used old schema fields, convert them to content blocks
    content = _migrate_old_schema(d, content)

    # Normalize each block and flatten bad nesting
    normalized = [_normalize_block(b) for b in content if b]
    out["content"] = _flatten_nested_siblings(normalized)
    return out


def _migrate_old_schema(d: Dict, existing_content: List) -> List:
    """
    If the LLM returned old-schema fields (body, subsections, tables),
    convert them into ordered content blocks.

    This handles the case where the LLM ignores the new schema and
    falls back to the old one.
    """
    blocks = list(existing_content)

    # Convert 'body' array to paragraph blocks
    body = d.get("body")
    if isinstance(body, list) and body:
        for line in body:
            if isinstance(line, str) and line.strip():
                blocks.append({"type": "paragraph", "text": line})

    # Convert 'subsections' to subsection blocks
    subs = d.get("subsections")
    if isinstance(subs, list) and subs:
        for sub in subs:
            if isinstance(sub, dict):
                block = {"type": "subsection"}
                block["heading"] = sub.get("heading", "")
                block["heading_level"] = sub.get("heading_level", "")
                # Recursively normalize the subsection's content
                inner = sub.get("content", [])
                if not isinstance(inner, list):
                    inner = []
                inner = _migrate_old_schema(sub, inner)
                block["content"] = [_normalize_block(b) for b in inner if b]
                blocks.append(block)

    # Convert top-level 'tables' to table blocks
    tables = d.get("tables")
    if isinstance(tables, list) and tables:
        for tbl in tables:
            if isinstance(tbl, dict):
                blocks.append({
                    "type": "table",
                    "caption": tbl.get("caption", ""),
                    "headers": tbl.get("headers", []),
                    "rows": tbl.get("rows", []),
                })

    # Handle rogue 'text' field
    text_val = d.get("text")
    if isinstance(text_val, str) and text_val.strip():
        # Prepend as first paragraph if not already in blocks
        blocks.insert(0, {"type": "paragraph", "text": text_val})

    return blocks


def _normalize_block(block: Any) -> Any:
    """Normalize a single content block."""
    if not isinstance(block, dict):
        # Bare string → paragraph
        if isinstance(block, str) and block.strip():
            return {"type": "paragraph", "text": block}
        return None

    btype = block.get("type", "")

    # Handle rogue 'section'/'text' fields (old unhandled_content schema)
    if "section" in block and "text" in block and not btype:
        text = block.get("text", "")
        if isinstance(text, str) and text.strip():
            return {"type": "paragraph", "text": text}
        return None

    if btype == "paragraph":
        text = block.get("text", "")
        if isinstance(text, str) and text.strip():
            return {"type": "paragraph", "text": text}
        return None

    if btype == "table":
        return {
            "type": "table",
            "caption": block.get("caption", ""),
            "headers": block.get("headers", []),
            "rows": block.get("rows", []),
        }

    if btype == "subsection":
        content = block.get("content", [])
        if not isinstance(content, list):
            content = [content] if content else []
        content = _migrate_old_schema(block, content)
        normalized = [_normalize_block(b) for b in content if b]
        normalized = [b for b in normalized if b]  # remove Nones
        return {
            "type": "subsection",
            "heading": block.get("heading", ""),
            "heading_level": block.get("heading_level", ""),
            "content": _flatten_nested_siblings(normalized),
        }

    # Unknown type — try to salvage as paragraph
    text = block.get("text", "")
    if isinstance(text, str) and text.strip():
        return {"type": "paragraph", "text": text}
    return None


# ======================================================================
# Flattening
# ======================================================================

def _flatten_nested_siblings(content: List[Dict]) -> List[Dict]:
    """
    Flatten incorrectly nested sibling subsections in a content array.

    If subsection A contains subsection B as its only/first content,
    and B has the same heading_level or numbering depth as A, then B
    should be a sibling not a child.
    """
    if not content:
        return []

    result: List[Dict] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "subsection":
            result.append(block)
            continue

        inner = block.get("content", [])
        if not inner:
            result.append(block)
            continue

        # Check if any inner subsection should be a sibling
        should_flatten = False
        parent_level = block.get("heading_level", "")
        parent_depth = _numbering_depth(block.get("heading", ""))

        for inner_block in inner:
            if not isinstance(inner_block, dict) or inner_block.get("type") != "subsection":
                continue
            child_heading = inner_block.get("heading", "")
            if not child_heading or not child_heading.strip():
                continue
            child_level = inner_block.get("heading_level", "")
            child_depth = _numbering_depth(child_heading)
            if parent_level and child_level and parent_level == child_level:
                should_flatten = True
                break
            if parent_depth > 0 and child_depth > 0 and parent_depth == child_depth:
                should_flatten = True
                break

        if should_flatten:
            # Keep parent's non-subsection content, then flatten children
            parent_content = [b for b in inner if not isinstance(b, dict) or b.get("type") != "subsection"]
            child_subs = [b for b in inner if isinstance(b, dict) and b.get("type") == "subsection"]

            parent_copy = dict(block)
            parent_copy["content"] = parent_content
            result.append(parent_copy)
            result.extend(_flatten_nested_siblings(child_subs))
        else:
            # Legitimate nesting — recurse
            block_copy = dict(block)
            block_copy["content"] = _flatten_nested_siblings(inner)
            result.append(block_copy)

    return result


def _numbering_depth(heading: str) -> int:
    if not heading:
        return 0
    heading = heading.strip()
    m = re.match(r'^(\d+(?:\.\d+)*)\b', heading)
    if m:
        return m.group(1).count('.') + 1
    if re.match(r'^\([a-z0-9ivxlc]+\)', heading, re.IGNORECASE):
        return 1
    return 0


# ======================================================================
# Core merge
# ======================================================================

def _merge_section_batches(
    batches: List[Dict],
    trailing_contexts: List[Dict[str, Any]],
) -> Dict:
    """
    Merge multiple batch dicts into one section.

    With the ordered content schema, merging is straightforward:
    - Keep heading/heading_level from first batch
    - For each continuation batch, append its content blocks in order,
      stitching the first block into the trailing subsection if it's
      a continuation.
    """
    if not batches:
        return {}
    if len(batches) == 1:
        return batches[0]

    merged = copy.deepcopy(batches[0])

    for i in range(1, len(batches)):
        overlay = batches[i]
        prev_ctx = trailing_contexts[i - 1] if i - 1 < len(trailing_contexts) else {}
        _apply_overlay(merged, overlay, prev_ctx)

    return merged


def _apply_overlay(merged: Dict, overlay: Dict, prev_ctx: Dict) -> None:
    """
    Append a continuation batch's content blocks onto the merged result.

    When there's a trailing subsection, leading content in the overlay
    (paragraphs, tables, or a continuation subsection) gets routed into
    that subsection. Everything after goes to the top-level content.
    """
    trailing_sub = prev_ctx.get("trailing_subsection", "")
    overlay_content = overlay.get("content", [])
    if not isinstance(overlay_content, list):
        return

    merged_content = merged.setdefault("content", [])
    start_idx = 0

    if trailing_sub and overlay_content:
        last_sub = _find_last_subsection_block(merged_content)

        if last_sub is not None:
            # Walk through overlay blocks and route leading content
            # into the trailing subsection until we hit a NEW subsection
            for j in range(len(overlay_content)):
                blk = overlay_content[j]
                if not isinstance(blk, dict):
                    break

                btype = blk.get("type", "")

                if btype in ("paragraph", "table"):
                    # Continuation content — route into trailing sub
                    last_sub.setdefault("content", []).append(copy.deepcopy(blk))
                    start_idx = j + 1

                elif btype == "subsection":
                    if _is_continuation(blk, trailing_sub):
                        # Continuation of the same subsection — stitch
                        _stitch_subsection(last_sub, blk)
                        start_idx = j + 1
                    else:
                        # New different subsection — stop routing
                        break
                else:
                    break

    # Append remaining blocks to top-level content
    for block in overlay_content[start_idx:]:
        merged_content.append(copy.deepcopy(block))


# ======================================================================
# Helpers
# ======================================================================

def _is_continuation(block: Dict, trailing_sub: str) -> bool:
    """Check if a subsection block is a continuation of trailing_sub."""
    heading = block.get("heading", "")
    if not isinstance(heading, str):
        return False
    h = heading.strip()
    if not h:
        return True
    if trailing_sub and h == trailing_sub.strip():
        return True
    return False


def _find_last_subsection_block(content: List[Dict]) -> Optional[Dict]:
    """Find the last subsection block in a content array."""
    for i in range(len(content) - 1, -1, -1):
        if isinstance(content[i], dict) and content[i].get("type") == "subsection":
            return content[i]
    return None


def _stitch_subsection(target: Dict, continuation: Dict) -> None:
    """Merge continuation subsection's content into target."""
    cont_content = continuation.get("content", [])
    if isinstance(cont_content, list) and cont_content:
        target.setdefault("content", []).extend(copy.deepcopy(cont_content))


# ======================================================================
# Deduplication
# ======================================================================

def _deduplicate_content(d: Any) -> None:
    """
    Remove duplicate paragraph blocks from content arrays.
    Preserves first occurrence order. Only deduplicates paragraphs
    (tables and subsections are kept as-is).
    """
    if not isinstance(d, dict):
        return

    content = d.get("content")
    if isinstance(content, list):
        seen_texts = set()
        deduped = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "paragraph":
                text = block.get("text", "")
                if text in seen_texts:
                    continue
                seen_texts.add(text)
            elif isinstance(block, dict) and block.get("type") == "subsection":
                _deduplicate_content(block)
            deduped.append(block)
        d["content"] = deduped


# ======================================================================
# Trailing context
# ======================================================================

def get_trailing_context(batch_result: Union[Dict, List]) -> Dict[str, Any]:
    """
    Extract trailing context: which subsection was last in the batch.
    """
    ctx: Dict[str, Any] = {"trailing_subsection": ""}

    if not isinstance(batch_result, dict):
        return ctx

    content = batch_result.get("content", [])
    if not isinstance(content, list) or not content:
        return ctx

    # Walk backward to find the last subsection
    for i in range(len(content) - 1, -1, -1):
        block = content[i]
        if isinstance(block, dict) and block.get("type") == "subsection":
            heading = block.get("heading", "")
            if isinstance(heading, str) and heading.strip():
                ctx["trailing_subsection"] = heading.strip()
            break

    return ctx
