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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ANSI escapes
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"


class ProgressDisplay:
    """Fixed-block ANSI display with per-worker progress bars."""

    def __init__(self, num_workers: int, total_files: int):
        self.num_workers = num_workers
        self.total = total_files
        self._worker_lines: list[str] = [
            f"  {DIM}[w{i + 1}] idle{RESET}" for i in range(num_workers)
        ]
        self._ok = 0
        self._fail = 0
        self._done = 0
        self._lock = threading.Lock()
        self._started = False

    def start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
            sys.stdout.write("\n" * (1 + self.num_workers))
            self._redraw()

    def finish(self):
        with self._lock:
            if self._started:
                sys.stdout.write("\n")
                sys.stdout.flush()

    def mark_started(self, worker_id: int, filename: str):
        self._update(
            worker_id,
            f"  {self._tag(worker_id)} {self._bar(0)}   0% "
            f"{DIM}Loading{RESET}  {filename}",
        )

    def mark_mapping(self, worker_id: int, filename: str):
        self._update(
            worker_id,
            f"  {self._tag(worker_id)} {self._bar(40)}  40% "
            f"{CYAN}Mapping{RESET}  {filename}",
        )

    def mark_materialising(self, worker_id: int, filename: str):
        self._update(
            worker_id,
            f"  {self._tag(worker_id)} {self._bar(75)}  75% "
            f"{YELLOW}Materialising{RESET}  {filename}",
        )

    def mark_ok(self, worker_id: int, filename: str, tables: int, elapsed: float):
        with self._lock:
            self._done += 1
            self._ok += 1
            self._worker_lines[worker_id - 1] = (
                f"  {self._tag(worker_id)} {self._bar(100)} "
                f"{GREEN}OK{RESET} ({tables}t, {elapsed:.1f}s)  {filename}"
            )
            if self._started:
                self._redraw()

    def mark_failed(self, worker_id: int, filename: str, error: str):
        with self._lock:
            self._done += 1
            self._fail += 1
            short_err = (error[:50] + "...") if len(error) > 50 else error
            self._worker_lines[worker_id - 1] = (
                f"  {self._tag(worker_id)} {self._bar(0)} "
                f"{RED}FAIL{RESET}  {filename}  {DIM}{short_err}{RESET}"
            )
            if self._started:
                self._redraw()

    def mark_idle(self, worker_id: int):
        self._update(
            worker_id,
            f"  {DIM}[w{worker_id}] finished{RESET}",
        )

    def _tag(self, worker_id: int) -> str:
        return f"[w{worker_id}|{self._done}/{self.total}]"

    @staticmethod
    def _bar(pct: int) -> str:
        filled = pct // 5
        return "\u2588" * filled + "\u2591" * (20 - filled)

    def _update(self, worker_id: int, text: str):
        with self._lock:
            self._worker_lines[worker_id - 1] = text
            if self._started:
                self._redraw()

    def _redraw(self):
        n = 1 + self.num_workers
        sys.stdout.write(f"\033[{n}A")
        pct = int(self._done / self.total * 100) if self.total else 100
        sys.stdout.write(
            f"\033[2K  {BOLD}Progress: "
            f"{self._done}/{self.total} ({pct}%){RESET}"
            f"  {GREEN}{self._ok} ok{RESET}  {RED}{self._fail} failed{RESET}\n"
        )
        for line in self._worker_lines:
            sys.stdout.write(f"\033[2K{line}\n")
        sys.stdout.flush()


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
    #     help="Path to a single JSON file, a .txt file list, or a "
    #          "directory containing *_extracted.json files to process",
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
    parser.add_argument(
        '--workers', '-w', type=int, default=1,
        help='Number of parallel workers for processing files '
             '(default: 1). Set to 1 to disable parallelism.',
    )
    parser.add_argument(
        '--dyno', action='store_true',
        help='Use the Dyno Nobel field mapper variant.',
    )
    args = parser.parse_args()
    args.input = r"C:\Users\hasssa\OneDrive - BHP\Documents\Code\extract\output\9100000695"
    # --- Provider override ---
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    from config.config_loader import load_config
    from config import settings
    if args.dyno:
        from src.agents.field_mapper_dyno import FieldMappingAgent
    else:
        from src.agents.field_mapper import FieldMappingAgent

    # --- Resolve input files ---
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input path not found: {input_path}")
        sys.exit(1)

    if input_path.is_file() and input_path.suffix == ".txt":
        # Read file list from a text file (one path per line)
        json_files = [
            Path(line.strip()) for line in input_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        missing = [f for f in json_files if not f.exists()]
        if missing:
            print(f"WARNING: {len(missing)} file(s) in list not found")
            json_files = [f for f in json_files if f.exists()]
        if not json_files:
            print(f"ERROR: No valid files found in {input_path}")
            sys.exit(1)
    elif input_path.is_file():
        json_files = [input_path]
    elif input_path.is_dir():
        json_files = sorted(input_path.rglob("*_extracted.json"))
        if not json_files:
            print(f"ERROR: No *_extracted.json files found in {input_path}")
            sys.exit(1)
    else:
        print(f"ERROR: {input_path} is not a file or directory")
        sys.exit(1)

    print(f"Provider: {settings.PROVIDER_NAME}")
    print(f"Input:    {input_path}")
    print(f"Files:    {len(json_files)}")
    print(f"Workers:  {args.workers}")
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

    def _process_single_file(json_file: Path) -> Dict:
        """Process one JSON file: load, map, materialise, save."""
        # Load
        try:
            with open(json_file, encoding="utf-8") as f:
                document = json.load(f)
        except json.JSONDecodeError as e:
            return {
                "document_id": json_file.stem,
                "file": str(json_file),
                "error": f"Invalid JSON: {e}",
                "tables_found": 0,
                "tables_mapped": 0,
                "proposals": [],
                "_target_dfs": {},
            }

        if not isinstance(document, dict):
            return {
                "document_id": json_file.stem,
                "file": str(json_file),
                "error": "Not a JSON object",
                "tables_found": 0,
                "tables_mapped": 0,
                "proposals": [],
                "_target_dfs": {},
            }

        document_id = document.get("document_id", json_file.stem)
        file_start = time.time()

        # Map
        report = agent.map_document(str(json_file), document_id)
        report["file"] = str(json_file)

        elapsed = time.time() - file_start
        report["elapsed_seconds"] = round(elapsed, 1)

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

        # Materialise target tables and write per-file Excel
        target_dfs = {}
        try:
            import pandas as pd

            target_dfs = agent.materialise_tables(
                document, report, document_id
            )

            if target_dfs:
                excel_path = out_path.with_suffix(".xlsx")
                agent.write_excel(target_dfs, excel_path, document_id)

        except ImportError:
            pass
        except Exception:
            pass

        report["_out_path"] = str(out_path)
        report["_target_dfs"] = target_dfs
        return report

    # --- Parallel or sequential execution ---
    effective_workers = min(args.workers, len(json_files))

    if effective_workers > 1 and len(json_files) > 1:
        display = ProgressDisplay(effective_workers, len(json_files))
        display.start()

        # Track which worker ID each thread gets
        _worker_ids: Dict[int, int] = {}  # future_id -> worker_id
        _next_worker = [1]  # mutable counter
        _worker_lock = threading.Lock()

        def _get_worker_id() -> int:
            tid = threading.get_ident()
            with _worker_lock:
                if tid not in _worker_ids:
                    _worker_ids[tid] = _next_worker[0]
                    _next_worker[0] = (_next_worker[0] % effective_workers) + 1
                return _worker_ids[tid]

        def _process_with_display(json_file: Path) -> Dict:
            wid = _get_worker_id()
            fname = json_file.name
            display.mark_started(wid, fname)

            # Load
            try:
                with open(json_file, encoding="utf-8") as f:
                    document = json.load(f)
            except json.JSONDecodeError as e:
                display.mark_failed(wid, fname, f"Invalid JSON: {e}")
                return {
                    "document_id": json_file.stem,
                    "file": str(json_file),
                    "error": f"Invalid JSON: {e}",
                    "tables_found": 0, "tables_mapped": 0,
                    "proposals": [], "_target_dfs": {},
                }

            if not isinstance(document, dict):
                display.mark_failed(wid, fname, "Not a JSON object")
                return {
                    "document_id": json_file.stem,
                    "file": str(json_file),
                    "error": "Not a JSON object",
                    "tables_found": 0, "tables_mapped": 0,
                    "proposals": [], "_target_dfs": {},
                }

            document_id = document.get("document_id", json_file.stem)
            file_start = time.time()

            # Map
            display.mark_mapping(wid, fname)
            report = agent.map_document(str(json_file), document_id)
            report["file"] = str(json_file)

            elapsed = time.time() - file_start
            report["elapsed_seconds"] = round(elapsed, 1)

            # Save mapping report
            if output_dir is not None:
                out_path = output_dir / f"{document_id}_field_mapping.json"
            else:
                out_path = settings.FINAL_DIR / f"{document_id}_field_mapping.json"

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            # Materialise
            display.mark_materialising(wid, fname)
            target_dfs = {}
            try:
                import pandas as pd
                target_dfs = agent.materialise_tables(
                    document, report, document_id
                )
                if target_dfs:
                    excel_path = out_path.with_suffix(".xlsx")
                    agent.write_excel(target_dfs, excel_path, document_id)
            except (ImportError, Exception):
                pass

            tables_mapped = report.get("tables_mapped", 0)
            if report.get("error"):
                display.mark_failed(wid, fname, report["error"])
            else:
                display.mark_ok(wid, fname, tables_mapped, elapsed)

            report["_out_path"] = str(out_path)
            report["_target_dfs"] = target_dfs
            return report

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_file = {
                executor.submit(_process_with_display, jf): jf
                for jf in json_files
            }
            for future in as_completed(future_to_file):
                jf = future_to_file[future]
                try:
                    report = future.result()
                except Exception as e:
                    report = {
                        "document_id": jf.stem,
                        "file": str(jf),
                        "error": str(e),
                        "tables_found": 0,
                        "tables_mapped": 0,
                        "proposals": [],
                        "_target_dfs": {},
                    }

                # Collect target DataFrames
                target_dfs = report.pop("_target_dfs", {})
                if "TblInvoice" in target_dfs:
                    combined_invoice_dfs.append(target_dfs["TblInvoice"])
                if "TblTimesheets" in target_dfs:
                    combined_timesheet_dfs.append(target_dfs["TblTimesheets"])

                report.pop("_out_path", None)
                all_reports.append(report)

        display.finish()
        print()
    else:
        for fi, json_file in enumerate(json_files, 1):
            print(f"{'=' * 70}")
            print(f"FILE {fi}/{len(json_files)}: {json_file.name}")
            print(f"{'=' * 70}")

            report = _process_single_file(json_file)

            if report.get("error"):
                print(f"  ERROR: {report['error']}")
                print()

            # Collect target DataFrames
            target_dfs = report.pop("_target_dfs", {})
            if "TblInvoice" in target_dfs:
                combined_invoice_dfs.append(target_dfs["TblInvoice"])
            if "TblTimesheets" in target_dfs:
                combined_timesheet_dfs.append(target_dfs["TblTimesheets"])

            # Print summary
            _print_proposal_summary(report, verbose=not args.quiet)
            out_path = report.pop("_out_path", "")
            if out_path:
                print(f"  Mapping report: {out_path}")

            if target_dfs:
                for tbl_name, df in target_dfs.items():
                    print(f"    {tbl_name}: {len(df)} rows, {len(df.columns)} columns")

            print(f"  Time:           {report.get('elapsed_seconds', 0):.1f}s")
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