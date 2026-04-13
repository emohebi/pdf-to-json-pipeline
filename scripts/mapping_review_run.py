#!/usr/bin/env python3
"""
Review field mapping accuracy against the mapper's output files.

Takes the original extracted JSON(s) and the mapper's output directory
containing both mapping reports (*_field_mapping.json) and materialised
Excel files (*_field_mapping.xlsx). Reviews classification, mapping,
and value correctness using a second LLM call as QA.

Usage:
    # Single file — mapping outputs in same directory
    python scripts/mapping_review_run.py \\
        --input ./extracted/my_doc.json \\
        --mapping ./output/

    # Directory of source JSONs + directory of mapper outputs
    python scripts/mapping_review_run.py \\
        --input ./extracted/ \\
        --mapping ./output/

    # With custom output directory
    python scripts/mapping_review_run.py \\
        --input ./extracted/ \\
        --mapping ./output/ \\
        --output ./review_reports/

    # Quiet mode
    python scripts/mapping_review_run.py \\
        --input ./extracted/ \\
        --mapping ./output/ -q
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _find_mapping_files(
    document_id: str,
    source_stem: str,
    mapping_dir: Path,
) -> tuple:
    """
    Find the mapping report JSON and Excel file for a document.

    The mapper outputs:
      {document_id}_field_mapping.json
      {document_id}_field_mapping.xlsx

    We try document_id first, then source file stem, then
    source stem with _extracted stripped.

    Returns (json_path_or_None, excel_path_or_None).
    """
    candidates = [document_id, source_stem]
    # Also try stripping common suffixes
    for suffix in ("_extracted", "_output", "_final"):
        if source_stem.endswith(suffix):
            candidates.append(source_stem[: -len(suffix)])

    for stem in candidates:
        json_path = mapping_dir / f"{stem}_field_mapping.json"
        excel_path = mapping_dir / f"{stem}_field_mapping.xlsx"

        if json_path.exists():
            return json_path, excel_path if excel_path.exists() else None

    # Fallback: search for any _field_mapping.json containing this doc id
    for f in mapping_dir.glob("*_field_mapping.json"):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("document_id") == document_id:
                xlsx = f.with_suffix(".xlsx")
                return f, xlsx if xlsx.exists() else None
        except Exception:
            continue

    return None, None


def _load_excel_as_dfs(
    excel_path: Path,
) -> Dict[str, Any]:
    """
    Load the materialised Excel file into a dict of DataFrames,
    keyed by sheet name (TblInvoice, TblTimesheets).
    """
    try:
        import pandas as pd
        xls = pd.ExcelFile(excel_path)
        result = {}
        for sheet_name in xls.sheet_names:
            if sheet_name in ("No Data",):
                continue
            df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
            df = df.fillna("")
            result[sheet_name] = df
        return result
    except Exception as e:
        print(f"    WARNING: Failed to load Excel {excel_path}: {e}")
        return {}


def _print_review_summary(
    review: Dict, verbose: bool = True
) -> None:
    """Print a formatted review summary."""
    table_reviews = review.get("table_reviews", [])
    stats = review.get("statistics", {})

    if review.get("error"):
        print(f"  Error: {review['error']}")
        return

    print(f"  Tables reviewed:  {stats.get('tables_reviewed', 0)}")
    print(f"  Average accuracy: {stats.get('average_accuracy_pct', 0):.1f}%")
    print()

    if not verbose or not table_reviews:
        return

    for i, tr in enumerate(table_reviews, 1):
        table_id = tr.get("table_id", f"table_{i}")

        if "error" in tr:
            print(f"  TABLE {i}: {table_id}")
            print(f"    Error: {tr['error']}")
            print()
            continue

        accuracy = tr.get("overall_accuracy_pct", 0)
        summary = tr.get("summary", "")

        cls_v = tr.get("classification_verdict", {})
        cls_verdict = cls_v.get("verdict", "?")
        cls_actual = cls_v.get("actual_classification", "?")
        cls_expected = cls_v.get("expected_classification", "?")

        print(f"  TABLE {i}: {table_id}")
        print(f"    Accuracy:        {accuracy:.1f}%")
        print(f"    Summary:         {summary[:100]}")
        print()

        cls_icon = "OK" if cls_verdict == "correct" else "WRONG"
        if cls_verdict == "correct":
            print(f"    Classification:  [{cls_icon}] {cls_actual}")
        else:
            print(
                f"    Classification:  [{cls_icon}] {cls_actual} "
                f"-> should be {cls_expected}"
            )
            print(
                f"                     "
                f"{cls_v.get('reasoning', '')[:80]}"
            )
        print()

        mapping_vs = tr.get("mapping_verdicts", [])
        correct = [m for m in mapping_vs if m.get("verdict") == "correct"]
        incorrect = [
            m for m in mapping_vs if m.get("verdict") == "incorrect"
        ]
        missing = [
            m for m in mapping_vs if m.get("verdict") == "missing"
        ]
        unnecessary = [
            m for m in mapping_vs if m.get("verdict") == "unnecessary"
        ]

        print(
            f"    Mappings: {len(correct)} correct, "
            f"{len(incorrect)} incorrect, "
            f"{len(missing)} missing, "
            f"{len(unnecessary)} unnecessary"
        )

        if incorrect:
            print(f"    INCORRECT MAPPINGS:")
            for m in incorrect:
                print(
                    f"      {m['source']:25s} -> "
                    f"{m['mapped_target']:20s} "
                    f"(should be: {m['expected_target']})"
                )
                if m.get("reasoning"):
                    print(f"        {m['reasoning'][:70]}")

        if missing:
            print(f"    MISSING MAPPINGS (should have been mapped):")
            for m in missing:
                print(
                    f"      {m['source']:25s} -> UNMAPPED "
                    f"(should be: {m['expected_target']})"
                )

        if unnecessary:
            print(f"    UNNECESSARY MAPPINGS (should be UNMAPPED):")
            for m in unnecessary:
                print(
                    f"      {m['source']:25s} -> "
                    f"{m['mapped_target']}"
                )

        value_vs = tr.get("value_verdicts", [])
        val_incorrect = [
            v for v in value_vs if v.get("verdict") == "incorrect"
        ]
        if val_incorrect:
            print()
            print(f"    INCORRECT VALUES ({len(val_incorrect)}):")
            for v in val_incorrect:
                print(
                    f"      {v['target_field']:25s}: "
                    f"got '{v['output_value']}' "
                    f"expected '{v['expected_value']}'"
                )
                if v.get("reasoning"):
                    print(f"        {v['reasoning'][:70]}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Review field mapping accuracy by checking the "
                    "mapper's JSON reports and Excel outputs",
    )
    # parser.add_argument(
    #     "--input", "-i", required=True,
    #     help="Path to extracted JSON file or directory of JSON files "
    #          "(the original source documents)",
    # )
    # parser.add_argument(
    #     "--mapping", "-m", required=True,
    #     help="Directory containing the mapper's outputs: "
    #          "*_field_mapping.json and *_field_mapping.xlsx files",
    # )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for review report(s). "
             "If omitted, saves to the pipeline's final directory.",
    )
    parser.add_argument(
        "--provider",
        choices=["aws_bedrock", "azure_openai"],
        help="Override LLM provider",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Print only summary, not per-table details",
    )
    args = parser.parse_args()
    args.input = r"input\JSONs"
    args.mapping = r"output\20260408_173948\final"
    args.output = r"output\20260408_173948\review"
    # --- Provider override ---
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    from config.config_loader import load_config
    from config import settings
    from src.agents.mapping_reviewer import MappingReviewAgent

    # --- Resolve input files ---
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input path not found: {input_path}")
        sys.exit(1)

    if input_path.is_file():
        json_files = [input_path]
    elif input_path.is_dir():
        json_files = sorted(
            f for f in input_path.glob("*.json")
            if "_field_mapping" not in f.stem
            and "_mapping_review" not in f.stem
            and "_combined" not in f.stem
        )
        if not json_files:
            print(f"ERROR: No source .json files found in {input_path}")
            sys.exit(1)
    else:
        print(f"ERROR: {input_path} is not a file or directory")
        sys.exit(1)

    # --- Resolve mapping directory ---
    mapping_dir = Path(args.mapping)
    if not mapping_dir.exists():
        print(f"ERROR: Mapping directory not found: {mapping_dir}")
        sys.exit(1)
    if not mapping_dir.is_dir():
        print(f"ERROR: --mapping must be a directory: {mapping_dir}")
        sys.exit(1)

    # --- Output directory ---
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

    print(f"Provider:    {settings.PROVIDER_NAME}")
    print(f"Input:       {input_path}")
    print(f"Mapping dir: {mapping_dir}")
    print(f"Files:       {len(json_files)}")
    print()

    # --- Process ---
    reviewer = MappingReviewAgent()
    all_reviews: List[Dict] = []
    total_start = time.time()

    for fi, json_file in enumerate(json_files, 1):
        print(f"{'=' * 70}")
        print(f"FILE {fi}/{len(json_files)}: {json_file.name}")
        print(f"{'=' * 70}")

        # Load source document
        try:
            with open(json_file, encoding="utf-8") as f:
                document = json.load(f)
        except Exception as e:
            print(f"  ERROR loading document: {e}")
            print()
            all_reviews.append({
                "document_id": json_file.stem, "error": str(e),
            })
            continue

        document_id = document.get("document_id", json_file.stem)

        # Find mapping outputs in the mapping directory
        report_path, excel_path = _find_mapping_files(
            document_id, json_file.stem, mapping_dir,
        )

        if report_path is None:
            print(
                f"  WARNING: No mapping report found for "
                f"'{document_id}' in {mapping_dir}"
            )
            print(f"  Run field_mapper_run.py first.")
            print()
            all_reviews.append({
                "document_id": document_id,
                "error": "No mapping report found",
            })
            continue

        print(f"  Mapping report: {report_path.name}")
        if excel_path:
            print(f"  Mapping Excel:  {excel_path.name}")
        else:
            print(f"  Mapping Excel:  (not found — value checks will be limited)")

        # Load mapping report
        try:
            with open(report_path, encoding="utf-8") as f:
                mapping_report = json.load(f)
        except Exception as e:
            print(f"  ERROR loading mapping report: {e}")
            print()
            all_reviews.append({
                "document_id": document_id, "error": str(e),
            })
            continue

        # Load Excel as DataFrames for value checking
        materialised = None
        if excel_path:
            materialised = _load_excel_as_dfs(excel_path)
            if materialised:
                for tbl_name, df in materialised.items():
                    print(
                        f"  Loaded {tbl_name}: "
                        f"{len(df)} rows, {len(df.columns)} columns"
                    )
            else:
                print(f"  WARNING: Excel loaded but no data sheets found")

        # Swap description with orig_description so the reviewer
        # validates the raw extracted value, not the normalised one
        if materialised:
            for df in materialised.values():
                if (
                    "orig_description" in df.columns
                    and "description" in df.columns
                ):
                    df["description"] = df["orig_description"]

        # Run review
        file_start = time.time()
        review = reviewer.review_mapping(
            document=document,
            mapping_report=mapping_report,
            materialised_data=materialised,
            document_id=document_id,
        )
        elapsed = time.time() - file_start

        print()
        _print_review_summary(review, verbose=not args.quiet)

        # Save review report as Excel
        if output_dir is not None:
            review_path = (
                output_dir / f"{document_id}_mapping_review.xlsx"
            )
        else:
            review_path = (
                settings.FINAL_DIR
                / f"{document_id}_mapping_review.xlsx"
            )

        try:
            reviewer.write_review_excel(review, review_path)
            print(f"  Review report: {review_path}")
        except Exception as e:
            # Fall back to JSON if Excel writing fails
            json_fallback = review_path.with_suffix(".json")
            with open(json_fallback, "w", encoding="utf-8") as f:
                json.dump(review, f, indent=2, ensure_ascii=False)
            print(f"  Review report: {json_fallback} (Excel failed: {e})")

        print(f"  Review report: {review_path}")
        print(f"  Time:          {elapsed:.1f}s")
        print()

        all_reviews.append(review)

    # --- Combined summary ---
    total_elapsed = time.time() - total_start

    completed = [r for r in all_reviews if "error" not in r]
    accuracies = [
        r.get("statistics", {}).get("average_accuracy_pct", 0)
        for r in completed
    ]

    print(f"{'=' * 70}")
    print("COMBINED REVIEW SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Files reviewed:     {len(all_reviews)}")
    print(f"  Files with errors:  {len(all_reviews) - len(completed)}")

    if accuracies:
        avg_acc = sum(accuracies) / len(accuracies)
        print(f"  Average accuracy:   {avg_acc:.1f}%")
        print(f"  Min accuracy:       {min(accuracies):.1f}%")
        print(f"  Max accuracy:       {max(accuracies):.1f}%")

    # Aggregate stats
    total_correct = total_incorrect = total_missing = total_unnecessary = 0
    total_val_correct = total_val_incorrect = 0
    for r in completed:
        stats = r.get("statistics", {})
        total_correct += stats.get("mappings_correct", 0)
        total_incorrect += stats.get("mappings_incorrect", 0)
        total_missing += stats.get("mappings_missing", 0)
        total_unnecessary += stats.get("mappings_unnecessary", 0)
        total_val_correct += stats.get("values_correct", 0)
        total_val_incorrect += stats.get("values_incorrect", 0)

    total_mappings = (
        total_correct + total_incorrect
        + total_missing + total_unnecessary
    )
    if total_mappings > 0:
        print()
        print(f"  MAPPING ACCURACY ({total_mappings} total):")
        print(
            f"    Correct:      {total_correct:>4d} "
            f"({total_correct / total_mappings * 100:.1f}%)"
        )
        print(
            f"    Incorrect:    {total_incorrect:>4d} "
            f"({total_incorrect / total_mappings * 100:.1f}%)"
        )
        print(
            f"    Missing:      {total_missing:>4d} "
            f"({total_missing / total_mappings * 100:.1f}%)"
        )
        print(
            f"    Unnecessary:  {total_unnecessary:>4d} "
            f"({total_unnecessary / total_mappings * 100:.1f}%)"
        )

    total_vals = total_val_correct + total_val_incorrect
    if total_vals > 0:
        print()
        print(f"  VALUE ACCURACY ({total_vals} checked):")
        print(
            f"    Correct:      {total_val_correct:>4d} "
            f"({total_val_correct / total_vals * 100:.1f}%)"
        )
        print(
            f"    Incorrect:    {total_val_incorrect:>4d} "
            f"({total_val_incorrect / total_vals * 100:.1f}%)"
        )

    print(f"\n  Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Save combined review as Excel
    if len(json_files) > 1:
        if output_dir is not None:
            combined_path = (
                output_dir / "_combined_mapping_review.xlsx"
            )
        else:
            combined_path = (
                settings.FINAL_DIR / "_combined_mapping_review.xlsx"
            )

        combined_report = {
            "document_id": "COMBINED",
            "tables_reviewed": sum(
                r.get("tables_reviewed", 0) for r in completed
            ),
            "table_reviews": [
                {**tr, "_document_id": r.get("document_id", "")}
                for r in completed
                for tr in r.get("table_reviews", [])
            ],
            "statistics": {
                "tables_reviewed": sum(
                    r.get("statistics", {}).get("tables_reviewed", 0)
                    for r in completed
                ),
                "average_accuracy_pct": (
                    round(sum(accuracies) / len(accuracies), 1)
                    if accuracies else 0.0
                ),
                "classification_correct": sum(
                    r.get("statistics", {}).get(
                        "classification_correct", 0
                    ) for r in completed
                ),
                "classification_incorrect": sum(
                    r.get("statistics", {}).get(
                        "classification_incorrect", 0
                    ) for r in completed
                ),
                "mappings_correct": total_correct,
                "mappings_incorrect": total_incorrect,
                "mappings_missing": total_missing,
                "mappings_unnecessary": total_unnecessary,
                "values_correct": total_val_correct,
                "values_incorrect": total_val_incorrect,
            },
        }

        try:
            reviewer.write_review_excel(combined_report, combined_path)
            print(f"Combined review: {combined_path}")
        except Exception as e:
            print(f"WARNING: Combined Excel failed: {e}")


if __name__ == "__main__":
    main()