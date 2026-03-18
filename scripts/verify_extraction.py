#!/usr/bin/env python3
"""
Run extraction verification as a standalone step.

Takes an extraction file (CSV, Excel, or JSON) with a list of
information items and their page numbers, plus a PDF file, and
verifies each item against the relevant PDF page(s).

Output:
  - Excel report  : one row per item, colour-coded verdict + reasoning
  - JSON summary  : overall accuracy statistics

Usage:
    python scripts/verify_extraction.py \\
        --pdf ./input/my_doc.pdf \\
        --extraction ./input/items.csv \\
        --output ./output/verification

    python scripts/verify_extraction.py \\
        --pdf ./input/my_doc.pdf \\
        --extraction ./input/items.xlsx \\
        --info-field "Information" \\
        --page-field "Page No" \\
        --provider azure_openai

Extraction file formats
-----------------------
CSV / Excel:   Must have at least two columns:
               - information column  (configurable, default: "information")
               - page_number column  (configurable, default: "page_number")
               Any additional columns are carried through to the output.

JSON:          Array of objects, e.g.:
               [
                 {"information": "...", "page_number": 5},
                 ...
               ]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Extraction file reader
# ---------------------------------------------------------------------------

def read_extraction_file(
    path: Path,
    info_field: str,
    page_field: str,
) -> list:
    """
    Read CSV, Excel, or JSON extraction file into a list of dicts.

    Performs light normalisation:
      - Strips whitespace from string values.
      - Ensures the info_field and page_field keys are present
        (raises if either is missing from the file).
    """
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                "JSON extraction file must be an array of objects."
            )
        items = data

    elif suffix == ".csv":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for CSV input: pip install pandas")
        df = pd.read_csv(path, dtype=str).fillna("")
        items = df.to_dict(orient="records")

    elif suffix in (".xlsx", ".xls", ".xlsm"):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for Excel input: pip install pandas")
        df = pd.read_excel(path, dtype=str).fillna("")
        items = df.to_dict(orient="records")

    else:
        raise ValueError(
            f"Unsupported extraction file format: {suffix}. "
            "Use .csv, .xlsx, .xls, .xlsm, or .json"
        )

    # Strip whitespace
    for item in items:
        for k, v in item.items():
            if isinstance(v, str):
                item[k] = v.strip()

    # Validate required columns
    if items:
        first = items[0]
        missing_cols = []
        if info_field not in first:
            missing_cols.append(f"'{info_field}' (information)")
        if page_field not in first:
            missing_cols.append(f"'{page_field}' (page number)")
        if missing_cols:
            available = list(first.keys())
            raise ValueError(
                f"Required column(s) not found: {', '.join(missing_cols)}.\n"
                f"Available columns: {available}\n"
                f"Use --info-field and --page-field to specify column names."
            )

    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify extracted information against a PDF document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # parser.add_argument(
    #     "--pdf", "-p", required=True,
    #     help="Path to the source PDF file",
    # )
    # parser.add_argument(
    #     "--extraction", "-e", required=True,
    #     help="Path to the extraction file (CSV, Excel, or JSON)",
    # )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Output directory or base path for reports. "
            "Defaults to the pipeline's configured output directory. "
            "Two files are written: <base>.xlsx and <base>_accuracy.json"
        ),
    )
    parser.add_argument(
        "--info-field",
        default=None,
        help="Column name for the information text (overrides config)",
    )
    parser.add_argument(
        "--page-field",
        default=None,
        help="Column name for the page number (overrides config)",
    )
    parser.add_argument(
        "--context-pages",
        type=int,
        default=None,
        help="Extra pages ±N to include around the target page (overrides config)",
    )
    parser.add_argument(
        "--provider",
        choices=["aws_bedrock", "azure_openai"],
        help="Override LLM provider",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help=(
            "Limit PDF extraction to a page range, e.g. '1-50'. "
            "Page numbers in the extraction file must still be "
            "absolute (1-based)."
        ),
    )
    args = parser.parse_args()
    args.provider = "azure_openai"
    sheet_name = "price_list_results - batch 1"
    file_name_col = "source_document"
    args.extraction = "input/price_list_results - comments.xlsx"

    # Provider override must happen before any config imports
    import os
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    from config.config_loader import get_task_config
    from config import settings
    from src.agents.extraction_verifier import ExtractionVerifier, get_verification_config
    from src.agents.verification_reporter import write_excel_report, write_json_summary
    from src.utils.pdf_processor import extract_pages
    import pandas as pd

    # ── Resolve field names (CLI > config > defaults) ────────────────
    vcfg = get_verification_config()
    info_field = args.info_field or vcfg.get("information_field", "information")
    page_field = args.page_field or vcfg.get("page_field", "page_number")

    if args.context_pages is not None:
        vcfg["context_pages"] = args.context_pages

    # ── Read extraction items ─────────────────────────────────────────
    extraction_path = Path(args.extraction)
    if not extraction_path.exists():
        print(f"ERROR: Extraction file not found: {extraction_path}")
        sys.exit(1)

    print(f"Reading extraction file: {extraction_path}")
    try:
        df = pd.read_excel(extraction_path, dtype=str, sheet_name=sheet_name).fillna("")
        files = df[file_name_col].unique().tolist()
        print(f"Number of Unique PDF files: {len(files)}")
    except (ValueError, ImportError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    for file_name in files:
        items = df[df[file_name_col] == file_name].to_dict(orient="records")
        print(f"  {len(items)} item(s) found")
        print(f"  Information field : '{info_field}'")
        print(f"  Page number field : '{page_field}'")

        # ── Validate paths ────────────────────────────────────────────────
        args.pdf = f"input/{file_name}"
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"WARNING: PDF not found: {pdf_path} .. skipping ")
            continue

        # ── Determine output paths ────────────────────────────────────────
        # if args.output:
        #     out_base = Path(args.output)
        #     # If user gave a directory, generate filename from PDF stem
        #     if out_base.suffix == "":
        #         out_base.mkdir(parents=True, exist_ok=True)
        #         stem = pdf_path.stem
        #         excel_out  = out_base / f"{stem}_verification.xlsx"
        #         json_out   = out_base / f"{stem}_accuracy.json"
        #     else:
        #         excel_out  = out_base.with_suffix(".xlsx")
        #         json_out   = out_base.with_name(out_base.stem + "_accuracy.json")
        # else:
        # Default to pipeline's FINAL_DIR
        final_dir = settings.FINAL_DIR
        stem = pdf_path.stem
        excel_out = final_dir / f"{stem}_verification.xlsx"
        json_out  = final_dir / f"{stem}_accuracy.json"

        # ── Extract PDF pages ─────────────────────────────────────────────
        print(f"\nExtracting PDF pages: {pdf_path}")
        pages_data = extract_pages(str(pdf_path))
        total_pages = len(pages_data)
        print(f"  {total_pages} pages extracted")

        # Optionally restrict to a page range (for large PDFs)
        if args.pages:
            parts = args.pages.split("-")
            start = max(1, int(parts[0]))
            end   = min(total_pages, int(parts[-1]))
            pages_data = pages_data[start - 1 : end]
            # Re-number so indices stay consistent with the extraction file
            for i, p in enumerate(pages_data):
                p["page_number"] = start + i
            print(f"  Filtered to pages {start}-{end} ({len(pages_data)} pages)")

        print(f"\nProvider: {settings.PROVIDER_NAME}")
        print(f"Output Excel : {excel_out}")
        print(f"Output JSON  : {json_out}")
        print()

        # ── Run verification ──────────────────────────────────────────────
        document_id = pdf_path.stem
        verifier = ExtractionVerifier()
        report = verifier.verify(items, pages_data, document_id)

        # ── Write outputs ─────────────────────────────────────────────────
        write_excel_report(report, excel_out)
        write_json_summary(report, json_out)

        # ── Print summary ─────────────────────────────────────────────────
        acc = report.get("accuracy", {})
        print(f"\n{'=' * 60}")
        print("VERIFICATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Total items     : {acc.get('total', 0)}")
        print(f"  Verifiable      : {acc.get('verifiable', 0)}")
        print(f"  ✔  CORRECT      : {acc.get('correct', 0)}")
        print(f"  ✘  INCORRECT    : {acc.get('incorrect', 0)}")
        print(f"  ?  MISSING      : {acc.get('missing', 0)}")
        print(f"  ~  UNVERIFIABLE : {acc.get('unverifiable', 0)}")
        print(f"  Accuracy        : {acc.get('accuracy_pct', 0.0):.1f}%")
        print(f"{'=' * 60}")
        print(f"\nExcel report : {excel_out}")
        print(f"JSON summary : {json_out}")


if __name__ == "__main__":
    main()
