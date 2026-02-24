#!/usr/bin/env python3
"""
Run term matching as a standalone step against an already-extracted
document JSON (the final output from the pipeline).

Reads the assembled document, reconstructs section representations,
then feeds them to TermMatchingAgent.

Usage:
    python scripts/term_matcher_run.py --input ./output/20260220/final/my_doc.json
    python scripts/term_matcher_run.py --input ./output.json --terms "Liability" "Insurance" "Termination"
    python scripts/term_matcher_run.py --input ./output.json --output ./term_report.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ====================================================================
# Reconstruct section_jsons from the final assembled document
# ====================================================================

def _reconstruct_sections_from_document(document: Dict) -> List[Dict]:
    """
    Convert a final pipeline document JSON back into the list-of-dicts
    format that TermMatchingAgent.match_terms() expects.

    The final document has sections bucketed by type under keys like
    'section', 'unhandled_content', etc.  Each array item is the raw
    extracted data (heading, body, subsections ...).

    We wrap each item back into::

        {
            "section_name": <derived from the data>,
            "data": <the raw extracted dict/list>,
            "_metadata": {"section_type": <key>}
        }
    """
    sections: List[Dict] = []

    # Use the header's section list for ordering hints (optional)
    header = document.get("document_header", {})
    header_section_names = header.get("sections", [])

    # Keys to skip — these are not section content
    skip_keys = {"document_id", "document_header"}

    for key, value in document.items():
        if key in skip_keys:
            continue

        if isinstance(value, list):
            for item in value:
                name = _derive_section_name(item, key)
                sections.append({
                    "section_name": name,
                    "data": item,
                    "_metadata": {"section_type": key},
                })
        elif isinstance(value, dict) and value:
            name = _derive_section_name(value, key)
            sections.append({
                "section_name": name,
                "data": value,
                "_metadata": {"section_type": key},
            })

    # If header has an ordered section list, sort by that order
    if header_section_names and sections:
        name_order = {n: i for i, n in enumerate(header_section_names)}
        sections.sort(
            key=lambda s: name_order.get(s["section_name"], 9999)
        )

    return sections


def _derive_section_name(data: Any, fallback_type: str) -> str:
    """
    Derive a human-readable section name from extracted data.

    Checks common keys: heading, section_name, section, caption,
    list_title — falls back to the section type key.
    """
    if isinstance(data, dict):
        for key in ("heading", "section_name", "section", "caption", "list_title"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return fallback_type.replace("_", " ").title()


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run term matching against an extracted document JSON"
    )
    # parser.add_argument(
    #     "--input", "-i", required=True,
    #     help="Path to the final output JSON from the pipeline",
    # )
    parser.add_argument(
        "--output", "-o",
        help="Path to write the term matching report. "
             "If omitted, saves to the default intermediate directory.",
    )
    parser.add_argument(
        "--terms", nargs="+",
        help="Override terms from CLI instead of using config.json "
             "(space-separated list)",
    )
    parser.add_argument(
        "--provider",
        choices=["aws_bedrock", "azure_openai"],
        help="Override LLM provider",
    )
    args = parser.parse_args()

    # --- Provider override (must happen before config imports) ---
    import os
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    from config.config_loader import (
        load_config, get_term_matching_config, get_document_type_name,
    )
    from config import settings
    from src.agents.term_matcher import TermMatchingAgent

    # --- Load the document JSON ---
    args.input = "./output/20260220_223355/final/BHP_JBB_EPCM_Contract_Final_R0_301122_Calibre_Signed.json"
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    try:
        with open(input_path, encoding="utf-8") as f:
            document = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {input_path}: {e}")
        sys.exit(1)

    if not isinstance(document, dict):
        print("ERROR: Expected a JSON object (the final document), "
              f"got {type(document).__name__}")
        sys.exit(1)

    document_id = document.get(
        "document_id", input_path.stem
    )

    # --- Reconstruct sections ---
    section_jsons = _reconstruct_sections_from_document(document)
    if not section_jsons:
        print("ERROR: No sections found in the document JSON")
        sys.exit(1)

    print(f"Document: {document_id}")
    print(f"Sections found: {len(section_jsons)}")
    for s in section_jsons:
        stype = s.get("_metadata", {}).get("section_type", "?")
        print(f"  [{stype}] {s['section_name']}")

    # --- CLI term override ---
    if args.terms:
        # Temporarily patch the config so the agent picks up CLI terms
        cfg = get_term_matching_config()
        cfg["terms"] = args.terms
        cfg["enabled"] = True
        settings.TERM_MATCHING_ENABLED = True
        print(f"\nTerms (CLI override): {args.terms}")
    else:
        cfg = get_term_matching_config()
        terms = cfg.get("terms", [])
        if not terms:
            print("\nERROR: No terms configured. Either set "
                  "TASK.term_matching.terms in config.json or "
                  "pass --terms from the CLI.")
            sys.exit(1)
        print(f"\nTerms (from config): {terms}")

    print(f"Provider: {settings.PROVIDER_NAME}")
    print()

    # --- Run term matching ---
    matcher = TermMatchingAgent()
    report = matcher.match_terms(section_jsons, document_id)

    if report is None:
        print("ERROR: Term matching returned no results")
        sys.exit(1)

    # --- Save / print results ---
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to: {output_path}")
    else:
        print("Report saved to default intermediate directory")

    # --- Summary ---
    terms_result = report.get("terms", {})
    unmatched = report.get("unmatched_terms", [])

    print(f"\n{'=' * 60}")
    print("TERM MATCHING SUMMARY")
    print(f"{'=' * 60}")
    for term, info in terms_result.items():
        related = info.get("related_sections", [])
        if related:
            print(f"\n  {term}  ({len(related)} section(s))")
            for rs in related:
                rel = rs.get("relevance", "?")
                name = rs.get("section_name", "?")
                reason = rs.get("reason", "")
                print(f"    [{rel:>6}] {name}")
                if reason:
                    print(f"             {reason}")
        else:
            print(f"\n  {term}  (no matches)")

    if unmatched:
        print(f"\nUnmatched terms: {', '.join(unmatched)}")

    print(f"\n{'=' * 60}")
    matched_count = len(terms_result) - len(unmatched)
    print(
        f"Matched: {matched_count}/{len(terms_result)} terms  |  "
        f"Sections analysed: {len(section_jsons)}"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
