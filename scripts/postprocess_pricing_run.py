#!/usr/bin/env python3
"""
Run pricing table post-processing as a standalone step.

Reads all .xlsx pricing files from a directory, concatenates them
(adding a FILE_NAME column for traceability), then uses the LLM to:
  1. Clean descriptions (strip UOM/rate/currency noise)
  2. Reconcile remaining duplicates (abbreviations, synonyms, etc.)
  3. Reconcile ORDER_UOM values

Then computes VALID_TO dates and optionally performs weekly expansion
and cross-contract duplication.

Usage:
    python scripts/postprocess_pricing_run.py \\
        --input ./Concat-9100000695/

    python scripts/postprocess_pricing_run.py \\
        --input ./Concat-9100000695/ \\
        --output ./processed.xlsx \\
        --mode weekly \\
        --provider azure_openai

    python scripts/postprocess_pricing_run.py \\
        --input ./pricing_dir/ \\
        --no-cross-contract \\
        --save-mapping
"""
import argparse
import json
import os
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATE_FMT = "%d/%m/%Y"


# ====================================================================
# Date chaining logic
# ====================================================================

def compute_valid_to(
    df: pd.DataFrame,
    description_col: str = "ITEM_DESCRIPTION",
    valid_from_col: str = "VALID_FROM",
    valid_to_col: str = "VALID_TO",
    date_fmt: str = DATE_FMT,
) -> pd.DataFrame:
    """
    Compute VALID_TO dates by chaining within each ITEM_DESCRIPTION group.

    For each group (sorted by VALID_FROM asc):
      row[i].VALID_TO = row[i+1].VALID_FROM - 1 day  (if empty)
      last row.VALID_TO = today  (if empty)
    """
    df = df.copy()

    df["_vf_dt"] = pd.to_datetime(
        df[valid_from_col], format=date_fmt,
        dayfirst=True, errors="coerce",
    )

    if valid_to_col in df.columns:
        df["_vt_dt"] = pd.to_datetime(
            df[valid_to_col], format=date_fmt,
            dayfirst=True, errors="coerce",
        )
    else:
        df["_vt_dt"] = pd.NaT
        df[valid_to_col] = ""

    df.sort_values(
        [description_col, "_vf_dt"], ascending=[True, True], inplace=True,
    )
    df.reset_index(drop=True, inplace=True)

    for item, group in df.groupby(description_col):
        idxs = group.index.tolist()
        for i in range(len(idxs) - 1):
            cur, nxt = idxs[i], idxs[i + 1]
            if pd.isna(df.loc[cur, "_vt_dt"]):
                next_from = df.loc[nxt, "_vf_dt"]
                if pd.notna(next_from):
                    df.loc[cur, "_vt_dt"] = next_from - timedelta(days=1)

    today = pd.Timestamp.today().normalize()
    df.loc[pd.isna(df["_vt_dt"]), "_vt_dt"] = today

    df[valid_to_col] = df["_vt_dt"].dt.strftime(date_fmt)
    df.drop(columns=["_vf_dt", "_vt_dt"], inplace=True)
    return df


# ====================================================================
# Weekly expansion
# ====================================================================

def expand_weekly(
    df: pd.DataFrame,
    valid_from_col: str = "VALID_FROM",
    valid_to_col: str = "VALID_TO",
    date_fmt: str = DATE_FMT,
) -> pd.DataFrame:
    """Expand each row into weekly rows within [VALID_FROM, VALID_TO]."""
    expanded = []
    for _, row in df.iterrows():
        start = pd.to_datetime(
            row[valid_from_col], format=date_fmt,
            dayfirst=True, errors="coerce",
        )
        end = pd.to_datetime(
            row[valid_to_col], format=date_fmt,
            dayfirst=True, errors="coerce",
        )

        if pd.isna(start):
            expanded.append(row.to_dict())
            continue
        if pd.isna(end):
            r = row.to_dict()
            r[valid_from_col] = start.strftime(date_fmt)
            r[valid_to_col] = ""
            expanded.append(r)
            continue

        week_start = start
        while week_start <= end:
            week_end = min(week_start + timedelta(days=6), end)
            r = row.to_dict()
            r[valid_from_col] = week_start.strftime(date_fmt)
            r[valid_to_col] = week_end.strftime(date_fmt)
            expanded.append(r)
            week_start += timedelta(days=7)

    return pd.DataFrame(expanded)


# ====================================================================
# Cross-contract duplication
# ====================================================================

def duplicate_across_contracts(
    df: pd.DataFrame, contract_col: str = "CONTRACT_ID",
) -> pd.DataFrame:
    """Duplicate all rows across each CONTRACT_ID."""
    unique = df[contract_col].unique()
    if len(unique) <= 1:
        return df

    parts = []
    for cid in unique:
        block = df.copy()
        block[contract_col] = cid
        parts.append(block)

    result = pd.concat(parts, ignore_index=True)
    result.sort_values(
        [contract_col, "ITEM_DESCRIPTION", "VALID_FROM"], inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    return result


# ====================================================================
# Minimal sanitisation (invisible chars only — no semantic changes)
# ====================================================================

def sanitise_text(text: str) -> str:
    """Replace invisible/special Unicode chars with ASCII equivalents."""
    if not text or not text.strip():
        return text
    s = text.strip()
    s = s.replace('\u00a0', ' ')
    s = s.replace('\u200b', '')
    s = s.replace('\u200c', '')
    s = s.replace('\u200d', '')
    s = s.replace('\ufeff', '')
    s = s.replace('\u2013', '-')
    s = s.replace('\u2014', '-')
    s = s.replace('\u2012', '-')
    s = s.replace('\u2015', '-')
    s = s.replace('\u2018', "'")
    s = s.replace('\u2019', "'")
    s = s.replace('\u201c', '"')
    s = s.replace('\u201d', '"')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-process pricing table: LLM-clean descriptions, "
                    "reconcile duplicates, compute VALID_TO, optionally "
                    "expand weekly and duplicate across contracts."
    )
    # parser.add_argument(
    #     "--input", "-i", required=True,
    #     help="Path to the directory containing .xlsx pricing files.",
    # )
    parser.add_argument(
        "--output", "-o",
        help="Output Excel path. Default: <input_dir>/all_processed.xlsx",
    )
    parser.add_argument(
        "--mode", choices=["non-weekly", "weekly"], default="non-weekly",
        help="Output mode (default: non-weekly)",
    )
    parser.add_argument(
        "--no-cross-contract", action="store_true",
        help="Disable cross-contract duplication",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM entirely (no cleaning, no reconciliation)",
    )
    parser.add_argument(
        "--provider", choices=["aws_bedrock", "azure_openai"],
        help="Override LLM provider",
    )
    parser.add_argument(
        "--save-mapping", action="store_true",
        help="Save the reconciliation mapping to a JSON file",
    )
    args = parser.parse_args()
    args.input = r"C:\Users\hasssa\OneDrive - BHP\Documents\Concat - 9100000695"
    args.output = Path(args.input) / "results_2" / "all_processed.xlsx"
    # --- Provider override ---
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    from config.config_loader import load_config
    from config import settings

    # --- Resolve paths ---
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"ERROR: Input path not found: {input_dir}")
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"ERROR: Not a directory: {input_dir}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "_weekly" if args.mode == "weekly" else "_processed"
        output_path = input_dir / f"all{suffix}.xlsx"

    # --- Load and concatenate .xlsx files ---
    xlsx_files = sorted(input_dir.glob("*.xlsx"))
    if not xlsx_files:
        print(f"ERROR: No .xlsx files found in {input_dir}")
        sys.exit(1)

    print(f"Loading .xlsx files from: {input_dir}")
    all_dfs = []
    for f in xlsx_files:
        file_df = pd.read_excel(f, dtype=str).fillna("")
        file_df["FILE_NAME"] = f.stem.replace("_pricing", "") + ".pdf"
        all_dfs.append(file_df)
        print(f"  {f.name}: {len(file_df)} rows")

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total: {len(df)} rows from {len(xlsx_files)} file(s)")

    # Validate required columns
    required = ["ITEM_DESCRIPTION", "VALID_FROM", "CONTRACT_ID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"  Available: {list(df.columns)}")
        sys.exit(1)

    # --- Step 0: Sanitise invisible characters (regex only) ---
    print(f"\nStep 0: Sanitising invisible characters...")
    df["ITEM_DESCRIPTION_RAW"] = df["ITEM_DESCRIPTION"].copy()
    df["ITEM_DESCRIPTION"] = df["ITEM_DESCRIPTION"].apply(sanitise_text)
    sanitised_count = (
        df["ITEM_DESCRIPTION"] != df["ITEM_DESCRIPTION_RAW"]
    ).sum()
    print(f"  {sanitised_count} values had invisible chars fixed")

    # --- Step 1: LLM cleaning + reconciliation ---
    desc_mapping: Dict[str, str] = {}
    uom_mapping: Dict[str, str] = {}

    if not args.no_llm:
        from src.agents.description_reconciler import DescriptionReconciler

        print(f"\nStep 1: LLM-powered cleaning & reconciliation...")
        print(f"  Provider: {settings.PROVIDER_NAME}")

        reconciler = DescriptionReconciler()
        uom_values = (
            df["ORDER_UOM"].tolist() if "ORDER_UOM" in df.columns
            else None
        )

        result = reconciler.reconcile(
            descriptions=df["ITEM_DESCRIPTION"].tolist(),
            contract_ids=df["CONTRACT_ID"].tolist(),
            uom_values=uom_values,
            document_id=input_dir.name,
        )

        desc_mapping = result.get("item_descriptions", {})
        uom_mapping = result.get("order_uom", {})
        stats = result.get("stats", {})

        print(f"\n  Results:")
        print(f"    Distinct original:         {stats.get('distinct_original', '?')}")
        print(f"    After LLM cleaning:        {stats.get('distinct_after_cleaning', '?')}")
        print(f"    After case/punct norm:     {stats.get('distinct_after_normalisation', '?')}")
        print(f"    After duplicate merging:   {stats.get('distinct_reconciled', '?')}")
        print(f"    Cleaning mappings:         {stats.get('cleaning_mappings', '?')}")
        print(f"    Duplicate mappings:        {stats.get('duplicate_mappings', '?')}")
        print(f"    Verification mappings:     {stats.get('verification_mappings', '?')}")
        print(f"    Total mappings:            {stats.get('total_mappings', '?')}")
        print(f"    UOM mappings:              {len(uom_mapping)}")

        if desc_mapping:
            print(f"\n  Description mappings:")
            for variant, canonical in sorted(desc_mapping.items()):
                print(f"    '{variant}' -> '{canonical}'")

        if uom_mapping:
            print(f"\n  UOM mappings:")
            for variant, canonical in sorted(uom_mapping.items()):
                print(f"    '{variant}' -> '{canonical}'")

        if args.save_mapping:
            mapping_path = output_path.with_name(
                f"{output_path.stem}_mapping.json"
            )
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n  Mapping saved: {mapping_path}")

    else:
        print(f"\nStep 1: LLM SKIPPED (--no-llm)")

    # --- Step 2: Apply mappings ---
    print(f"\nStep 2: Applying mappings...")

    if desc_mapping:
        original_descs = df["ITEM_DESCRIPTION"].copy()
        df["ITEM_DESCRIPTION"] = df["ITEM_DESCRIPTION"].map(
            lambda x: desc_mapping.get(x.strip(), x.strip())
        )
        changed = (df["ITEM_DESCRIPTION"] != original_descs).sum()
        print(f"  {changed} description values remapped")

    if uom_mapping and "ORDER_UOM" in df.columns:
        original_uom = df["ORDER_UOM"].copy()
        df["ORDER_UOM"] = df["ORDER_UOM"].map(
            lambda x: uom_mapping.get(x.strip(), x.strip())
        )
        changed = (df["ORDER_UOM"] != original_uom).sum()
        print(f"  {changed} UOM values remapped")

    # --- Step 3: Compute VALID_TO ---
    print(f"\nStep 3: Computing VALID_TO dates...")
    distinct = df["ITEM_DESCRIPTION"].nunique()
    df = compute_valid_to(df)
    print(f"  {distinct} distinct item descriptions")
    print(f"  {len(df)} rows after date chaining")

    # --- Step 4: Mode-specific processing ---
    if args.mode == "weekly":
        print(f"\nStep 4: Expanding to weekly rows...")
        df = expand_weekly(df)
        print(f"  {len(df)} rows after weekly expansion")
    else:
        print(f"\nStep 4: Non-weekly snapshot mode")

    # --- Step 5: Cross-contract duplication ---
    if not args.no_cross_contract:
        print(f"\nStep 5: Cross-contract duplication...")
        n_contracts = df["CONTRACT_ID"].nunique()
        rows_before = len(df)
        df = duplicate_across_contracts(df)
        print(f"  {n_contracts} contracts x {rows_before} rows = {len(df)}")
    else:
        print(f"\nStep 5: Cross-contract duplication SKIPPED")

    # --- Step 6: Save ---
    print(f"\nStep 6: Saving...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(str(output_path), index=False)

    print(f"\n{'=' * 60}")
    print("POST-PROCESSING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Input:          {input_dir} ({len(xlsx_files)} files)")
    print(f"  Output:         {output_path}")
    print(f"  Mode:           {args.mode}")
    print(f"  LLM:            {'yes' if not args.no_llm else 'no'}")
    print(f"  Cross-contract: {'yes' if not args.no_cross_contract else 'no'}")
    print(f"  Desc mappings:  {len(desc_mapping)}")
    print(f"  UOM mappings:   {len(uom_mapping)}")
    print(f"  Final rows:     {len(df)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()