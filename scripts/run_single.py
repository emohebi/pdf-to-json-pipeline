#!/usr/bin/env python3
"""
Process a single PDF (or all PDFs in the input directory) to JSON.
All configuration from config.json -- no hardcoded values.

Usage:
    python scripts/run_single.py --pdf ./document.pdf
    python scripts/run_single.py --pdf ./document.pdf --sections-json ./sections.json
    python scripts/run_single.py --pdf ./document.pdf --pages 5-20
    python scripts/run_single.py --provider azure_openai
    python scripts/run_single.py --config ./my_config.json
    python scripts/run_single.py --review
    python scripts/run_single.py --term-match
"""
import argparse
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_page_range(page_range_str: str):
    """
    Parse a page range string like '5-20' into a (start, end) tuple (1-based, inclusive).
    Also supports single page like '5' (returns (5, 5)).
    """
    if not page_range_str:
        return None
    page_range_str = page_range_str.strip()
    if '-' in page_range_str:
        parts = page_range_str.split('-', 1)
        start = int(parts[0].strip())
        end = int(parts[1].strip())
    else:
        start = end = int(page_range_str)
    if start < 1 or end < start:
        raise ValueError(f"Invalid page range: {page_range_str} (must be start-end with start >= 1)")
    return (start, end)


def main():
    parser = argparse.ArgumentParser(
        description="PDF to JSON extraction pipeline"
    )
    # parser.add_argument(
    #     "--config", help="Path to config.json"
    # )
    parser.add_argument(
        "--pdf", help="Path to a single PDF file"
    )
    parser.add_argument(
        "--sections-json",
        help="Path to a section detection JSON file. "
             "If provided, section detection is skipped and this "
             "file is used directly as the section map.",
    )
    parser.add_argument(
        "--pages",
        help="Page range to extract, e.g. '5-20' or '10'. "
             "1-based, inclusive. If omitted, all pages are processed.",
    )
    parser.add_argument(
        "--provider",
        choices=["aws_bedrock", "azure_openai"],
        help="Override LLM provider",
    )
    parser.add_argument(
        "--review", action="store_true", help="Enable review stage"
    )
    parser.add_argument(
        "--no-review", action="store_true", help="Disable review stage"
    )
    parser.add_argument(
        "--term-match", action="store_true",
        help="Enable term matching stage (overrides config)"
    )
    parser.add_argument(
        "--no-term-match", action="store_true",
        help="Disable term matching stage (overrides config)"
    )
    parser.add_argument(
        "--effective-date", action="store_true",
        help="Enable effective date extraction (overrides config)"
    )
    parser.add_argument(
        "--no-effective-date", action="store_true",
        help="Disable effective date extraction (overrides config)"
    )
    args = parser.parse_args()
    args.config = None
    args.sections_json = None
    args.pdf = "./input/DYNO NOBEL (9100075152)_DoAA and Contract-signed.pdf"
    # args.sections_json = "./output/20260301_154125/intermediate/detection/Amended_and_Restated_GPSFA_KPMG_Fully_Executed_151221_2_detection.json"
    # args.pages = "1-131"
    # Parse page range
    page_range = None
    if args.pages:
        try:
            page_range = parse_page_range(args.pages)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    # Apply overrides before loading config
    if args.config:
        os.environ["PIPELINE_CONFIG"] = args.config
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    from config.config_loader import (
        load_config, get_input_config, get_document_type_name,
    )
    from config import settings

    # if args.review:
    #     settings.REVIEW_ENABLED = True
    # elif args.no_review:
    #     settings.REVIEW_ENABLED = False

    # if args.term_match:
    #     settings.TERM_MATCHING_ENABLED = True
    # elif args.no_term_match:
    #     settings.TERM_MATCHING_ENABLED = False

    # if args.effective_date:
    #     settings.EFFECTIVE_DATE_ENABLED = True
    # elif args.no_effective_date:
    #     settings.EFFECTIVE_DATE_ENABLED = False

    from src.pipeline import process_document

    print("Pipeline Configuration:")
    print(f"  Provider: {settings.PROVIDER_NAME}")
    print(f"  Document type: {get_document_type_name()}")
    print(f"  Review: {settings.REVIEW_ENABLED}")
    print(f"  Term Matching: {settings.TERM_MATCHING_ENABLED}")
    print(f"  Effective Date: {settings.EFFECTIVE_DATE_ENABLED}")
    print(f"  UOM: {settings.UOM_EXTRACTION_ENABLED}")
    print(f"  Output: {settings.OUTPUT_DIR}")
    if page_range:
        print(f"  Page range: {page_range[0]}-{page_range[1]}")
    if args.sections_json:
        print(f"  Sections JSON: {args.sections_json} (detection SKIPPED)")
    print()

    # --- Load pre-computed sections if provided ---
    precomputed_sections = None
    if args.sections_json:
        sections_path = Path(args.sections_json)
        if not sections_path.exists():
            print(f"ERROR: Sections JSON not found: {sections_path}")
            sys.exit(1)
        try:
            with open(sections_path, encoding='utf-8') as f:
                precomputed_sections = json.load(f)
            if not isinstance(precomputed_sections, list):
                print(
                    "ERROR: Sections JSON must be a list of section dicts"
                )
                sys.exit(1)
            print(
                f"Loaded {len(precomputed_sections)} sections from "
                f"{sections_path.name}"
            )
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in {sections_path}: {e}")
            sys.exit(1)

    # --- Process PDF(s) ---
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"ERROR: PDF not found: {pdf_path}")
            sys.exit(1)
        print(f"Processing: {pdf_path}")
        result = process_document(
            str(pdf_path),
            precomputed_sections=precomputed_sections,
            page_range=page_range,
        )
        if result:
            print(f"SUCCESS: {pdf_path.name}")
        else:
            print(f"FAILED: {pdf_path.name}")
            sys.exit(1)
    else:
        input_cfg = get_input_config()
        pdf_dir = Path(input_cfg.get("pdf_directory", "./input"))
        extensions = input_cfg.get(
            "supported_extensions", [".pdf"]
        )

        if not pdf_dir.exists():
            print(f"ERROR: Input directory not found: {pdf_dir}")
            sys.exit(1)

        pdfs = [
            f for f in pdf_dir.iterdir()
            if f.suffix.lower() in extensions
        ]
        if not pdfs:
            print(f"No PDF files found in {pdf_dir}")
            sys.exit(0)

        print(f"Found {len(pdfs)} PDF(s) in {pdf_dir}")
        for pdf_path in sorted(pdfs):
            print(f"\nProcessing: {pdf_path.name}")
            try:
                result = process_document(
                    str(pdf_path),
                    precomputed_sections=precomputed_sections,
                    page_range=page_range,
                )
                status = "SUCCESS" if result else "FAILED"
                print(f"  {status}: {pdf_path.name}")
            except Exception as e:
                print(f"  ERROR: {pdf_path.name}: {e}")


if __name__ == "__main__":
    main()