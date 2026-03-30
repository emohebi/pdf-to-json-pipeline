#!/usr/bin/env python3
"""
Run effective date extraction as a standalone step against an
already-extracted document JSON (the final output from the pipeline).

Reads the assembled document, reconstructs section representations
and header info, then feeds them to EffectiveDateExtractor.

Usage:
    python scripts/effective_date_run.py --input ./output/final/my_doc.json
    python scripts/effective_date_run.py --input ./output.json --output ./dates.json
    python scripts/effective_date_run.py --input ./output.json --provider azure_openai
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ====================================================================
# Reconstruct sections from the final assembled document
# (shared logic with term_matcher_run.py)
# ====================================================================

def _reconstruct_sections_from_document(document: Dict) -> List[Dict]:
    """
    Convert a final pipeline document JSON back into the list-of-dicts
    format that the agents expect.
    """
    sections: List[Dict] = []
    skip_keys = {"document_id", "document_header"}
    header = document.get("document_header", {})
    header_section_names = header.get("sections", [])

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

    if header_section_names and sections:
        name_order = {n: i for i, n in enumerate(header_section_names)}
        sections.sort(
            key=lambda s: name_order.get(s["section_name"], 9999)
        )

    return sections


def _derive_section_name(data: Any, fallback_type: str) -> str:
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
        description="Extract effective date from an extracted document JSON"
    )
    # parser.add_argument(
    #     "--input", "-i", required=True,
    #     help="Path to the final output JSON from the pipeline",
    # )
    parser.add_argument(
        "--output", "-o",
        help="Path to write the effective date report. "
             "If omitted, saves to the default intermediate directory.",
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

    from config.config_loader import load_config
    from config import settings
    from src.agents.effective_date_extractor import EffectiveDateExtractor
    args.input = "./output/9100000695 Monadelphous Extension (2 years) + Rate Review Amending Deed Mono Signed & Executed/final/9100000695_Monadelphous_Extension_2_years_Rate_Review_Amending_Deed_Mono_Signed.json"
    # --- Load the document JSON ---
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

    document_id = document.get("document_id", input_path.stem)

    # --- Reconstruct sections ---
    section_jsons = _reconstruct_sections_from_document(document)
    if not section_jsons:
        print("ERROR: No sections found in the document JSON")
        sys.exit(1)

    # --- Extract header ---
    document_header = document.get("document_header", {})

    print(f"Document: {document_id}")
    print(f"Sections found: {len(section_jsons)}")
    for s in section_jsons:
        stype = s.get("_metadata", {}).get("section_type", "?")
        print(f"  [{stype}] {s['section_name']}")

    # Show header date if present
    header_date = document_header.get("date", {})
    if isinstance(header_date, dict):
        header_date = header_date.get("text", "")
    if header_date:
        print(f"\nHeader date: {header_date}")

    print(f"Provider: {settings.PROVIDER_NAME}")
    print()

    # --- Force-enable for standalone run ---
    settings.EFFECTIVE_DATE_ENABLED = True

    # --- Run extraction ---
    extractor = EffectiveDateExtractor()
    report = extractor.extract_effective_date(
        section_jsons, document_id,
        document_header=document_header,
    )

    if report is None:
        print("ERROR: Effective date extraction returned no results")
        sys.exit(1)

    # --- Save if custom output path ---
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to: {output_path}")
    else:
        print("Report saved to default intermediate directory")

    # --- Print summary ---
    print(f"\n{'=' * 60}")
    print("EFFECTIVE DATE EXTRACTION RESULTS")
    print(f"{'=' * 60}")

    primary = report.get("primary_effective_date", {})
    no_date = report.get("no_date_found", True)

    if no_date or not primary.get("date"):
        print("\n  No effective date could be identified.")
    else:
        print(f"\n  PRIMARY EFFECTIVE DATE:")
        print(f"    Date:       {primary.get('date', '?')}")
        normalised = primary.get("normalised", "")
        if normalised:
            print(f"    Normalised: {normalised}")
        print(f"    Type:       {primary.get('date_type', '?')}")
        print(f"    Source:     {primary.get('source_section', '?')}")
        print(f"    Confidence: {primary.get('confidence', '?')}")
        reason = primary.get("reason", "")
        if reason:
            print(f"    Reason:     {reason}")

    all_dates = report.get("all_dates_found", [])
    if all_dates and len(all_dates) > 1:
        print(f"\n  ALL CANDIDATE DATES ({len(all_dates)}):")
        for i, d in enumerate(all_dates, 1):
            conf = d.get("confidence", "?")
            dtype = d.get("date_type", "?")
            src = d.get("source_section", "?")
            date_str = d.get("date", "?")
            normalised = d.get("normalised", "")
            reason = d.get("reason", "")

            label = f"{date_str}"
            if normalised:
                label += f" ({normalised})"

            print(f"    {i}. [{conf:>6}] {label}")
            print(f"              Type: {dtype} | Section: {src}")
            if reason:
                print(f"              {reason}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
