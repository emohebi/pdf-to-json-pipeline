#!/usr/bin/env python3
"""
Run unit of measure extraction as a standalone step against an
already-extracted document JSON (the final output from the pipeline).

Reads the assembled document, reconstructs section representations,
then feeds them to UOMExtractor.

Usage:
    python scripts/uom_extraction_run.py --input ./output/final/my_doc.json
    python scripts/uom_extraction_run.py --input ./output.json --output ./uom_report.json
    python scripts/uom_extraction_run.py --input ./output.json --provider azure_openai
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
        description="Extract units of measure from an extracted document JSON"
    )
    # parser.add_argument(
    #     "--input", "-i", required=True,
    #     help="Path to the final output JSON from the pipeline",
    # )
    parser.add_argument(
        "--output", "-o",
        help="Path to write the UOM report. "
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
    from src.agents.uom_extractor import UOMExtractor
    args.input = "./output/20260224_234604/final/Amended_and_Restated_GPSFA_KPMG_Fully_Executed_151221_2.json"
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

    print(f"Document: {document_id}")
    print(f"Sections found: {len(section_jsons)}")
    for s in section_jsons:
        stype = s.get("_metadata", {}).get("section_type", "?")
        print(f"  [{stype}] {s['section_name']}")

    print(f"Provider: {settings.PROVIDER_NAME}")
    print()

    # --- Force-enable for standalone run ---
    settings.UOM_EXTRACTION_ENABLED = True

    # --- Run extraction ---
    extractor = UOMExtractor()
    report = extractor.extract_uom(section_jsons, document_id)

    if report is None:
        print("ERROR: UOM extraction returned no results")
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
    print("UNIT OF MEASURE EXTRACTION RESULTS")
    print(f"{'=' * 60}")

    no_uom = report.get("no_uom_found", True)
    uoms = report.get("units_of_measure", [])
    distinct = report.get("distinct_units", [])

    if no_uom or not uoms:
        print("\n  No units of measure found in this document.")
    else:
        print(f"\n  DISTINCT UNITS: {', '.join(distinct)}")
        print(f"  TOTAL REFERENCES: {len(uoms)}")
        print()

        # Group by normalised unit for display
        grouped: Dict[str, List[Dict]] = {}
        for u in uoms:
            norm = u.get("normalised_unit", u.get("unit", "?")).lower()
            grouped.setdefault(norm, []).append(u)

        for norm_unit, entries in grouped.items():
            print(f"  {norm_unit.upper()} ({len(entries)} reference(s))")
            for e in entries:
                conf = e.get("confidence", "?")
                src = e.get("source_section", "?")
                applies = e.get("applies_to", "")
                verbatim = e.get("verbatim_text", "")

                print(f"    [{conf:>6}] Section: {src}")
                if applies:
                    print(f"             Applies to: {applies}")
                if verbatim:
                    print(f"             Verbatim: \"{verbatim}\"")
            print()

    print(f"{'=' * 60}")
    print(
        f"Distinct units: {len(distinct)}  |  "
        f"Total references: {len(uoms)}  |  "
        f"Sections analysed: {len(section_jsons)}"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
