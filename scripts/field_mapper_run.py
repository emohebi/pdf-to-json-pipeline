#!/usr/bin/env python3
"""
Run field mapping against one or more extracted document JSONs.

Accepts either a single JSON file or a directory containing JSON files.
For each document, finds all tables, classifies each as invoice/timesheet/skip,
and maps source fields to target database columns using Azure OpenAI
structured outputs.

Usage:
    # Single file
    python scripts/field_mapper_run.py --input ./output/final/my_doc.json

    # Directory of JSON files
    python scripts/field_mapper_run.py --input ./output/final/

    # With custom output directory
    python scripts/field_mapper_run.py --input ./output/final/ \\
        --output ./mapping_reports/

    # Override provider
    python scripts/field_mapper_run.py --input ./output/final/ \\
        --provider azure_openai
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _print_proposal_summary(report: Dict, verbose: bool = True) -> None:
    """Print a formatted summary for a single document's mapping report."""
    document_id = report.get("document_id", "?")
    proposals = report.get("proposals", [])

    print(f"  Document:       {document_id}")
    print(f"  Tables found:   {report.get('tables_found', 0)}")
    print(f"  Tables mapped:  {report.get('tables_mapped', 0)}")

    if report.get("error"):
        print(f"  Error:          {report['error']}")

    if not verbose or not proposals:
        print()
        return

    print()

    for i, proposal in enumerate(proposals, 1):
        source = proposal.get("_source", {})
        target_table = proposal.get("target_table", "?")
        mappings = proposal.get("mappings", [])
        unmapped = proposal.get("unmapped_target_fields", [])
        reasoning = proposal.get("classification_reasoning", "")

        table_id = source.get("table_id", f"table_{i}")
        section_name = source.get("section_name", "?")
        col_count = source.get("column_count", 0)
        row_count = source.get("row_count", 0)

        print(f"  TABLE {i}: {table_id}")
        print(f"    Section:        {section_name}")
        print(f"    Classification: {target_table}")
        print(f"    Reasoning:      {reasoning[:100]}")
        print(f"    Columns:        {col_count}")
        print(f"    Rows:           {row_count}")
        print()

        if target_table == "SKIP":
            print(f"    (Skipped)")
            print()
            continue

        mapped = [m for m in mappings if m.get("target") != "UNMAPPED"]
        unmapped_src = [m for m in mappings if m.get("target") == "UNMAPPED"]

        if mapped:
            print(f"    MAPPED FIELDS ({len(mapped)}):")
            for m in mapped:
                conf = m.get("confidence", 0)
                hint = m.get("transform_hint", "none")
                hint_str = f" [{hint}]" if hint != "none" else ""
                print(
                    f"      {m['source']:30s} -> {m['target']:25s} "
                    f"({conf:.0%}){hint_str}"
                )
            print()

        if unmapped_src:
            print(f"    UNMAPPED SOURCE FIELDS ({len(unmapped_src)}):")
            for m in unmapped_src:
                print(
                    f"      {m['source']:30s}  "
                    f"({m.get('reasoning', '')[:60]})"
                )
            print()

        if unmapped:
            print(f"    UNMAPPED TARGET FIELDS ({len(unmapped)}):")
            for field in unmapped:
                print(f"      {field}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Classify and map table fields from extracted "
                    "document JSON(s) using LLM structured outputs",
    )
    # parser.add_argument(
    #     "--input", "-i", required=True,
    #     help="Path to a single JSON file or a directory containing "
    #          "JSON files to process",
    # )
    parser.add_argument(
        "--output", "-o",
        help="Output path. For a single file: path to the report JSON. "
             "For a directory: directory where reports are saved "
             "(one per input file, named <stem>_field_mapping.json). "
             "If omitted, saves to the pipeline's final directory.",
    )
    parser.add_argument(
        "--provider",
        choices=["aws_bedrock", "azure_openai"],
        help="Override LLM provider (structured outputs require "
             "azure_openai)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Print only the final summary, not per-table details",
    )
    args = parser.parse_args()
    args.input = r"C:\Users\hasssa\OneDrive - BHP\Documents\Code\extract\output\9100075152"
    # --- Provider override ---
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    from config.config_loader import load_config
    from config import settings
    from src.agents.field_mapper import FieldMappingAgent

    # --- Resolve input files ---
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input path not found: {input_path}")
        sys.exit(1)

    if input_path.is_file():
        json_files = [input_path]
    elif input_path.is_dir():
        json_files = sorted(input_path.glob("**/*_extracted.json"))
        if not json_files:
            print(f"ERROR: No .json files found in {input_path}")
            sys.exit(1)
    else:
        print(f"ERROR: {input_path} is not a file or directory")
        sys.exit(1)

    print(f"Provider: {settings.PROVIDER_NAME}")
    print(f"Input:    {input_path}")
    print(f"Files:    {len(json_files)}")
    print()

    # --- Resolve output directory ---
    if args.output:
        output_base = Path(args.output)
        # If single file and output looks like a file path, use it directly
        if len(json_files) == 1 and output_base.suffix == ".json":
            output_dir = None  # will use output_base as the file path
        else:
            output_dir = output_base
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None  # will use pipeline's FINAL_DIR

    # --- Process each file ---
    agent = FieldMappingAgent()
    all_reports: List[Dict] = []
    # Accumulate rows across all files for combined Excel output
    combined_invoice_dfs: List = []
    combined_timesheet_dfs: List = []
    total_start = time.time()

    for fi, json_file in enumerate(json_files, 1):
        print(f"{'=' * 70}")
        print(f"FILE {fi}/{len(json_files)}: {json_file.name}")
        print(f"{'=' * 70}")

        # Load
        try:
            with open(json_file, encoding="utf-8") as f:
                document = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  ERROR: Invalid JSON: {e}")
            print()
            all_reports.append({
                "document_id": json_file.stem,
                "file": str(json_file),
                "error": f"Invalid JSON: {e}",
                "tables_found": 0,
                "tables_mapped": 0,
                "proposals": [],
            })
            continue

        if not isinstance(document, dict):
            print(f"  ERROR: Expected JSON object, got {type(document).__name__}")
            print()
            all_reports.append({
                "document_id": json_file.stem,
                "file": str(json_file),
                "error": "Not a JSON object",
                "tables_found": 0,
                "tables_mapped": 0,
                "proposals": [],
            })
            continue

        document_id = document.get("document_id", json_file.stem)
        file_start = time.time()

        # Map
        report = agent.map_document(str(json_file), document_id)
        report["file"] = str(json_file)

        elapsed = time.time() - file_start
        report["elapsed_seconds"] = round(elapsed, 1)

        # Print summary
        _print_proposal_summary(report, verbose=not args.quiet)

        # Save individual mapping report (JSON)
        if output_dir is not None:
            out_path = output_dir / f"{document_id}_field_mapping.json"
        elif args.output and len(json_files) == 1:
            out_path = Path(args.output)
        else:
            out_path = settings.FINAL_DIR / f"{document_id}_field_mapping.json"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"  Mapping report: {out_path}")

        # Materialise target tables and write per-file Excel
        try:
            import pandas as pd

            target_dfs = agent.materialise_tables(
                document, report, document_id
            )

            if target_dfs:
                excel_path = out_path.with_suffix(".xlsx")
                agent.write_excel(target_dfs, excel_path, document_id)
                print(f"  Excel output:   {excel_path}")

                for tbl_name, df in target_dfs.items():
                    print(f"    {tbl_name}: {len(df)} rows, {len(df.columns)} columns")

                # Accumulate for combined output
                if "TblInvoice" in target_dfs:
                    combined_invoice_dfs.append(target_dfs["TblInvoice"])
                if "TblTimesheets" in target_dfs:
                    combined_timesheet_dfs.append(target_dfs["TblTimesheets"])
            else:
                print(f"  Excel output:   (no tables to materialise)")

        except ImportError:
            print(f"  WARNING: pandas/openpyxl not installed — skipping Excel")
        except Exception as e:
            print(f"  WARNING: Excel materialisation failed: {e}")

        print(f"  Time:           {elapsed:.1f}s")
        print()

        all_reports.append(report)

    # --- Combined summary ---
    total_elapsed = time.time() - total_start

    total_tables = sum(r.get("tables_found", 0) for r in all_reports)
    total_mapped = sum(r.get("tables_mapped", 0) for r in all_reports)
    total_errors = sum(1 for r in all_reports if r.get("error"))

    # Count classifications across all reports
    classification_counts = {"TblInvoice": 0, "TblTimesheets": 0, "SKIP": 0}
    for r in all_reports:
        for p in r.get("proposals", []):
            tt = p.get("target_table", "")
            if tt in classification_counts:
                classification_counts[tt] += 1

    print(f"{'=' * 70}")
    print("COMBINED SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Files processed:  {len(all_reports)}")
    print(f"  Files with errors:{total_errors}")
    print(f"  Total tables:     {total_tables}")
    print(f"  Total mapped:     {total_mapped}")
    print(f"  Classifications:")
    print(f"    TblInvoice:     {classification_counts['TblInvoice']}")
    print(f"    TblTimesheets:  {classification_counts['TblTimesheets']}")
    print(f"    SKIP:           {classification_counts['SKIP']}")
    print(f"  Total time:       {total_elapsed:.1f}s")

    if len(all_reports) > 1:
        print(f"  Avg per file:     {total_elapsed / len(all_reports):.1f}s")

    print(f"{'=' * 70}")

    # Save combined report if processing a directory
    if len(json_files) > 1:
        if output_dir is not None:
            combined_path = output_dir / "_combined_mapping_report.json"
        else:
            combined_path = (
                settings.FINAL_DIR / "_combined_mapping_report.json"
            )

        combined = {
            "files_processed": len(all_reports),
            "total_tables": total_tables,
            "total_mapped": total_mapped,
            "total_errors": total_errors,
            "classifications": classification_counts,
            "elapsed_seconds": round(total_elapsed, 1),
            "reports": all_reports,
        }

        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"Combined report: {combined_path}")

    # Write combined Excel with all rows across all files
    try:
        import pandas as pd

        combined_dfs = {}
        if combined_invoice_dfs:
            combined_dfs["TblInvoice"] = pd.concat(
                combined_invoice_dfs, ignore_index=True
            )
        if combined_timesheet_dfs:
            combined_dfs["TblTimesheets"] = pd.concat(
                combined_timesheet_dfs, ignore_index=True
            )

        if combined_dfs:
            if output_dir is not None:
                combined_excel = output_dir / "_combined_mapped_data.xlsx"
            else:
                combined_excel = (
                    settings.FINAL_DIR / "_combined_mapped_data.xlsx"
                )

            agent.write_excel(combined_dfs, combined_excel)

            print(f"\nCombined Excel:  {combined_excel}")
            for tbl_name, df in combined_dfs.items():
                print(f"  {tbl_name}: {len(df)} total rows from {len(json_files)} file(s)")

    except ImportError:
        pass
    except Exception as e:
        print(f"WARNING: Combined Excel failed: {e}")


if __name__ == "__main__":
    main()