#!/usr/bin/env python3
"""
Run CPI price adjustment as a standalone step against an
already-extracted document JSON (the final output from the pipeline),
a pricing table (Excel), and ABS CPI data (Excel).

Reads the assembled document, reconstructs section representations,
then feeds them to CPIAdjustmentAgent.

Usage:
    python scripts/cpi_adjustment_run.py \
        --input ./output/final/my_contract.json \
        --pricing ./pricing_table.xlsx \
        --cpi ./640101__2_.xlsx

    python scripts/cpi_adjustment_run.py \
        --input ./output/final/my_contract.json \
        --pricing ./pricing_table.xlsx \
        --cpi ./640101__2_.xlsx \
        --output ./adjusted_pricing.xlsx \
        --provider azure_openai
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ====================================================================
# Reconstruct section_jsons from the final assembled document
# (identical to term_matcher_run.py)
# ====================================================================

def _reconstruct_sections_from_document(document: Dict) -> List[Dict]:
    sections: List[Dict] = []
    header = document.get("document_header", {})
    header_section_names = header.get("sections", [])
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

    if header_section_names and sections:
        name_order = {n: i for i, n in enumerate(header_section_names)}
        sections.sort(key=lambda s: name_order.get(s["section_name"], 9999))

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
        description="Run CPI price adjustment against an extracted document JSON"
    )
    # parser.add_argument(
    #     "--input", "-i", required=True,
    #     help="Path to the final output JSON from the pipeline",
    # )
    # parser.add_argument(
    #     "--pricing", "-p", required=True,
    #     help="Path to the pricing table Excel file",
    # )
    # parser.add_argument(
    #     "--cpi", "-c", required=True,
    #     help="Path to the ABS CPI Excel file (6401.0)",
    # )
    parser.add_argument(
        "--output", "-o",
        help="Output Excel path (default: <pricing>_adjusted.xlsx)",
    )
    parser.add_argument(
        "--provider",
        choices=["aws_bedrock", "azure_openai"],
        help="Override LLM provider",
    )
    args = parser.parse_args()
    args.provider = "azure_openai"
    args.input = r"output\1._BMA_Dyno_5_site_contract_[Executed by both parties]\final\1._BMA_Dyno_5_site_contract_Executed_by_both_parties.json"
    args.pricing = r"output\1._BMA_Dyno_5_site_contract_[Executed by both parties]\final\1._BMA_Dyno_5_site_contract_Executed_by_both_parties_pricing.xlsx"
    args.cpi = r"C:\Users\hasssa\OneDrive - BHP\Documents\Code\Concat - 9100000695\640101 (2).xlsx"
    
    # --- Provider override (must happen before config imports) ---
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    from config.config_loader import load_config
    from config import settings
    from src.agents.cpi_adjustment import (
        CPIAdjustmentAgent, load_cpi_data, get_cpi_series_names,
    )

    # --- Output path ---
    if args.output is None:
        pricing_path = Path(args.pricing)
        args.output = str(
            pricing_path.parent / f"{pricing_path.stem}_adjusted.xlsx"
        )

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
        print(
            f"ERROR: Expected a JSON object (the final document), "
            f"got {type(document).__name__}"
        )
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

    # --- Load ABS CPI data ---
    print(f"\nLoading ABS CPI data from {args.cpi}...")
    cpi_df = load_cpi_data(args.cpi)
    cpi_series = get_cpi_series_names(cpi_df)
    print(
        f"  {len(cpi_df)} quarterly records, {len(cpi_series)} index series"
    )
    print(
        f"  Range: {cpi_df['date'].min().strftime('%Y-%m')} "
        f"to {cpi_df['date'].max().strftime('%Y-%m')}"
    )

    # --- Load pricing table ---
    print(f"\nLoading pricing table from {args.pricing}...")
    pricing_df = pd.read_excel(args.pricing, dtype=str)
    print(f"  {len(pricing_df)} rows loaded")

    print(f"\nProvider: {settings.PROVIDER_NAME}")
    print()

    # --- Run CPI adjustment ---
    agent = CPIAdjustmentAgent()
    report = agent.adjust_prices(
        section_jsons=section_jsons,
        pricing_df=pricing_df,
        cpi_df=cpi_df,
        cpi_series_names=cpi_series,
        document_id=document_id,
    )

    # --- Save adjusted pricing table ---
    result_df = report.pop("result_df", pricing_df)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result_df.to_excel(args.output, index=False)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("CPI ADJUSTMENT SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Document:        {document_id}")
    print(f"  Clauses found:   {report['clauses_found']}")
    print(f"  Formulas parsed: {report['formulas_parsed']}")
    print(f"  Original rows:   {report['original_rows']}")
    print(f"  Adjusted rows:   {report['adjusted_rows']}")
    print(f"  Total rows:      {report['total_rows']}")
    print(f"  Output:          {args.output}")

    if report.get("formulas"):
        print(f"\n  Formulas applied:")
        for i, f in enumerate(report["formulas"], 1):
            print(f"    {i}. {f.get('formula_type')} | {f.get('cpi_series')}")
            print(f"       Applies to: {f.get('applies_to')}")
            print(f"       {f.get('formula')}")
            if f.get("notes"):
                print(f"       Notes: {f['notes']}")

    if report.get("clauses"):
        print(f"\n  Source clauses:")
        for c in report["clauses"]:
            print(f"    [{c['section_type']}] {c['section_name']}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
