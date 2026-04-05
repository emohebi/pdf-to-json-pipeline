"""
Optional Stage: CPI Price Adjustment Agent - config-driven.

Takes extracted contract sections, identifies CPI/escalation clauses,
interprets them via the LLM, then applies the formula against ABS CPI
data to generate adjusted price rows in a pricing table.

Controlled by TASK.cpi_adjustment config block.
"""
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from src.tools.llm_provider import invoke_text
from src.utils import setup_logger, StorageManager

logger = setup_logger("cpi_adjustment")

DATE_FMT = "%d/%m/%Y"

# ── Default prompt (overridable via config.json TASK.prompts.cpi_adjustment) ──
_DEFAULT_PROMPT = """You are a contract pricing analyst specialising in Australian contracts.

I will give you a clause from a contract that describes a price adjustment mechanism (typically CPI-based).

Your task is to extract the adjustment formula as a structured JSON object.

CLAUSE TEXT:
{clause_text}

AVAILABLE CPI SERIES (from ABS 6401.0):
{cpi_series}

Return ONLY a JSON object (no markdown, no explanation) with these fields:
{{
    "formula_type": "cpi_ratio | cpi_percentage | fixed_percentage | other",
    "description": "<brief description of how the adjustment works>",
    "cpi_series": "<exact column name from the available CPI series list that best matches the clause, e.g. the city or Australia-wide>",
    "base_period": "<the base/reference period if mentioned, e.g. 'contract start date' or 'Q3 2020', or 'valid_from' if not specified>",
    "adjustment_frequency": "quarterly | annually | other",
    "formula": "<the mathematical formula, e.g. 'new_price = old_price * (CPI_current / CPI_base)'>",
    "applies_to": "all | <specific item descriptions if mentioned>",
    "notes": "<any caveats, caps, floors, or conditions mentioned>"
}}

RULES:
- formula_type "cpi_ratio" means: new_price = old_price * (CPI_new / CPI_old)
- formula_type "cpi_percentage" means: new_price = old_price * (1 + percentage_change)
- If the clause mentions a specific city (e.g. Sydney, Perth), pick that city's CPI series
- If no city is specified, use the Australia-wide series
- If no base period is specified, assume "valid_from" (the item's start date)
- If adjustment frequency is not specified, assume "annually"
"""


class CPIAdjustmentAgent:
    """
    Identify CPI adjustment clauses in contract sections, interpret
    them via the LLM, and generate adjusted pricing rows.
    """

    # Keywords that signal a price adjustment clause
    ESCALATION_KEYWORDS = [
        "cpi", "consumer price index", "escalat", "price adjust",
        "rate adjust", "cost adjust", "inflation",
        "rise and fall", "variation of price",
    ]

    def __init__(self):
        self.storage = StorageManager()
        self._load_config()

    def _load_config(self):
        """Load CPI adjustment config. Graceful fallback if not configured."""
        try:
            from config.config_loader import get_task_config, get_prompt
            cfg = get_task_config().get("cpi_adjustment", {})
            self._max_snippet = cfg.get("max_snippet_chars", 30000)
            self._max_tokens = cfg.get("max_tokens", 8192)
            # Try to load custom prompt template from config
            try:
                self._prompt_template = get_prompt("cpi_adjustment")
            except Exception:
                self._prompt_template = _DEFAULT_PROMPT
        except Exception:
            self._max_snippet = 30000
            self._max_tokens = 8192
            self._prompt_template = _DEFAULT_PROMPT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adjust_prices(
        self,
        section_jsons: List[Dict],
        pricing_df: pd.DataFrame,
        cpi_df: pd.DataFrame,
        cpi_series_names: List[str],
        document_id: str,
    ) -> Dict[str, Any]:
        """
        Full CPI adjustment pipeline.

        Args:
            section_jsons:    Reconstructed section dicts from pipeline JSON.
            pricing_df:       Pricing table DataFrame (read with dtype=str).
            cpi_df:           Parsed ABS CPI DataFrame from load_cpi_data().
            cpi_series_names: List of available CPI index series column names.
            document_id:      Identifier for logging / storage.

        Returns:
            Report dict::

                {
                    "document_id": "...",
                    "clauses_found": 3,
                    "formulas_parsed": 2,
                    "original_rows": 100,
                    "adjusted_rows": 50,
                    "total_rows": 150,
                    "formulas": [...],
                    "result_df": <DataFrame with appended rows>
                }
        """
        logger.info(
            f"[{document_id}] CPI adjustment: scanning "
            f"{len(section_jsons)} sections"
        )

        # Step 1: Find adjustment clauses
        clauses = self._find_adjustment_clauses(section_jsons)
        logger.info(
            f"[{document_id}] Found {len(clauses)} clause(s) "
            f"mentioning CPI/escalation"
        )

        if not clauses:
            return self._build_report(
                document_id, clauses=[], formulas=[],
                original_rows=len(pricing_df),
                result_df=pricing_df,
            )

        # Step 2: Interpret each clause with LLM
        formulas = []
        for i, clause in enumerate(clauses):
            logger.info(
                f"[{document_id}]   Clause {i + 1}: "
                f"{clause['section_name']}"
            )
            formula = self._interpret_clause(
                clause["clause_text"], cpi_series_names
            )
            if formula:
                periods = formula.get("periods", [])
                logger.info(
                    f"[{document_id}]     type={formula.get('formula_type')}, "
                    f"series={formula.get('cpi_series')}, "
                    f"applies_to={formula.get('applies_to')}, "
                    f"periods={len(periods)}"
                )
                for pi, p in enumerate(periods):
                    logger.info(
                        f"[{document_id}]       Period {pi+1}: "
                        f"{p.get('period_start')} to {p.get('period_end')}, "
                        f"Po={p.get('base_price')}, CPIo={p.get('base_cpi')}, "
                        f"freq={p.get('adjustment_frequency')}, "
                        f"formula={p.get('formula', '')[:60]}"
                    )
                formulas.append(formula)
            else:
                logger.warning(
                    f"[{document_id}]     Could not interpret clause"
                )

        if not formulas:
            return self._build_report(
                document_id, clauses=clauses, formulas=[],
                original_rows=len(pricing_df),
                result_df=pricing_df,
            )

        # Step 3: Apply formulas to pricing table
        result_df = self._apply_adjustments(
            pricing_df, formulas, cpi_df, document_id
        )

        report = self._build_report(
            document_id, clauses=clauses, formulas=formulas,
            original_rows=len(pricing_df), result_df=result_df,
        )

        # Persist report (without the DataFrame)
        save_report = {k: v for k, v in report.items() if k != "result_df"}
        try:
            self.storage.save_cpi_adjustment_result(document_id, save_report)
        except AttributeError:
            logger.warning(
                f"[{document_id}] StorageManager.save_cpi_adjustment_result() "
                f"not found. Add CPI_ADJUSTMENT_DIR to settings.py and "
                f"save_cpi_adjustment_result() to StorageManager."
            )

        return report

    # ------------------------------------------------------------------
    # Clause detection
    # ------------------------------------------------------------------

    def _find_adjustment_clauses(
        self, section_jsons: List[Dict]
    ) -> List[Dict[str, str]]:
        clauses = []
        for section in section_jsons:
            name = section.get("section_name", "")
            stype = section.get("_metadata", {}).get("section_type", "")
            data = section.get("data", {})
            full_text = self._extract_text_recursive(data)
            text_lower = full_text.lower()

            if any(kw in text_lower for kw in self.ESCALATION_KEYWORDS):
                # Extract focused CPI-relevant portions
                focused = self._extract_cpi_blocks(full_text)
                logger.info(
                    f"Section '{name}': {len(full_text)} total chars, "
                    f"{len(focused)} chars after CPI filtering"
                )
                clauses.append({
                    "section_name": name,
                    "section_type": stype,
                    "clause_text": focused[: self._max_snippet],
                })
        return clauses

    def _extract_cpi_blocks(self, text: str) -> str:
        """
        Extract contiguous blocks of text around CPI/escalation keywords.
        Uses a character-based window to grab surrounding context,
        ensuring formula components (Po, CPIo, dates) are captured
        even when split across many short lines.
        """
        lines = text.split("\n")
        lines = [l for l in lines if l.strip()]

        # Find all lines that contain CPI-related keywords
        keyword_lines = set()
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in self.ESCALATION_KEYWORDS):
                keyword_lines.add(i)
            # Also flag lines with formula indicators
            if any(indicator in line for indicator in [
                "Po =", "Po=", "CPIo", "CPIln", "CPIn", "Pn =", "Pn=",
                "base price", "Base Price", "Review Milestone",
                "calendar quarter", "Cat No.", "6401",
            ]):
                keyword_lines.add(i)

        if not keyword_lines:
            return ""

        # For each keyword line, grab a generous window around it
        # (30 lines before, 50 lines after to capture full formula blocks)
        relevant_indices = set()
        for idx in keyword_lines:
            for j in range(max(0, idx - 30), min(len(lines), idx + 50)):
                relevant_indices.add(j)

        # Build contiguous blocks
        sorted_indices = sorted(relevant_indices)
        result_parts = []
        for i in sorted_indices:
            result_parts.append(lines[i])

        return "\n".join(result_parts)

    # ------------------------------------------------------------------
    # LLM interpretation
    # ------------------------------------------------------------------

    def _interpret_clause(
        self,
        clause_text: str,
        cpi_series_names: List[str],
    ) -> Optional[Dict]:
        cpi_list = "\n".join(f"- {s}" for s in cpi_series_names)

        prompt = (
            "You are a contract pricing expert specialising in Australian contracts.\n\n"
            "TASK: Read the contract clause text below carefully line by line. "
            "Extract ALL CPI-based price adjustment periods and their formulas.\n\n"
            "Contracts often define MULTIPLE adjustment periods, for example:\n"
            "  Period A: '1 January 2023 to 1 April 2024' with formula Pn = Po x CPIn/CPIo, "
            "Po = $841/t, CPIo = 130.2\n"
            "  Period B: '1 July 2024 to 1 January 2026' with a different weighted formula\n\n"
            "Look for these specific values in the text:\n"
            "- Period date ranges (e.g., 'between 1 January 2023 to 1 April 2024')\n"
            "- Base price Po (e.g., 'Po = $841/t')\n"
            "- Base CPI CPIo (e.g., 'CPIo = 130.2')\n"
            "- Formula (e.g., 'Pn = Po x CPIn / CPIo')\n"
            "- CPI series reference (e.g., 'Cat No. 6401.0, Table 1, Brisbane index; A2325816R')\n\n"
            "If the clause does not contain a price adjustment or CPI mechanism, return:\n"
            '{"formula_type": "none", "description": "No adjustment formula found"}\n\n'
            f"CONTRACT CLAUSE TEXT:\n"
            f'"""\n{clause_text}\n"""\n\n'
            f"AVAILABLE CPI SERIES (from ABS 6401.0):\n{cpi_list}\n\n"
            "Return ONLY a JSON object with these fields:\n"
            "{\n"
            '  "formula_type": "cpi_ratio" or "cpi_percentage" or "weighted" or "none",\n'
            '  "description": "brief description",\n'
            '  "cpi_series": "matching column name from the available list above",\n'
            '  "cpi_series_id": "the ABS series ID if mentioned (e.g. A2325816R)",\n'
            '  "applies_to": "all" or "specific item descriptions",\n'
            '  "periods": [\n'
            '    {\n'
            '      "period_start": "DD/MM/YYYY",\n'
            '      "period_end": "DD/MM/YYYY",\n'
            '      "base_price": <number from the clause, e.g. 841>,\n'
            '      "base_cpi": <number from the clause, e.g. 130.2>,\n'
            '      "base_cpi_description": "e.g. CPI Brisbane Sep 2022 quarter",\n'
            '      "formula": "the exact formula text, e.g. Pn = Po x (CPIn / CPIo)",\n'
            '      "adjustment_frequency": "quarterly" or "annually" or "semi-annually",\n'
            '      "notes": "any weights, caps, floors, or conditions for this period"\n'
            '    }\n'
            '  ]\n'
            "}\n\n"
            "CRITICAL RULES:\n"
            "- You MUST extract the EXACT numeric base_price (Po) and base_cpi (CPIo) from the text\n"
            "- You MUST extract the EXACT period start and end dates in DD/MM/YYYY format\n"
            "- If the clause defines multiple periods (e.g., Period A, Period B), "
            "include EACH as a separate entry in the periods array\n"
            "- Read EVERY paragraph — the formula, Po, CPIo, and dates may each be in "
            "separate paragraphs\n"
            "- Match the CPI series to the closest column name from the available list\n\n"
            "IMPORTANT: Respond with ONLY the JSON object. No explanation, "
            "no markdown, no code blocks. Start with { and end with }."
        )

        logger.info(f"Sending clause to LLM ({len(clause_text)} chars of clause text)")
        logger.info(f"First 200 chars: {clause_text[:200]}")
        logger.info(f"Last 200 chars: {clause_text[-200:]}")

        try:
            response = invoke_text(prompt, max_tokens=self._max_tokens)
            logger.info(f"Raw LLM response (first 1500 chars):\n{response[:1500]}")

            response = response.strip()

            # Strip markdown code fences
            if response.startswith("```"):
                response = re.sub(r"^```(?:json)?\s*", "", response)
                response = re.sub(r"\s*```$", "", response)

            # Try to find JSON object in the response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                response = json_match.group(0)

            parsed = json.loads(response)
            logger.info(f"Parsed LLM response keys: {list(parsed.keys())}")

            # Detect error responses from the LLM
            if set(parsed.keys()).issubset({"error", "message", "reason"}):
                error_msg = parsed.get("error") or parsed.get("message") or str(parsed)
                logger.warning(f"LLM returned an error instead of formula: {error_msg}")
                return None

            # If the LLM wrapped the result inside a key, unwrap it
            if len(parsed) == 1:
                only_val = list(parsed.values())[0]
                if isinstance(only_val, dict) and any(
                    k in only_val for k in ("formula_type", "type", "cpi_series")
                ):
                    parsed = only_val

            # Normalise: the LLM sometimes uses slightly different keys
            normalised = {
                "formula_type": (
                    parsed.get("formula_type")
                    or parsed.get("type")
                    or parsed.get("adjustment_type")
                    or "cpi_ratio"
                ),
                "description": (
                    parsed.get("description")
                    or parsed.get("desc")
                    or ""
                ),
                "cpi_series": (
                    parsed.get("cpi_series")
                    or parsed.get("series")
                    or parsed.get("cpi_index")
                    or parsed.get("index")
                    or ""
                ),
                "base_period": (
                    parsed.get("base_period")
                    or parsed.get("base")
                    or parsed.get("reference_period")
                    or "valid_from"
                ),
                "adjustment_frequency": (
                    parsed.get("adjustment_frequency")
                    or parsed.get("frequency")
                    or "annually"
                ),
                "formula": (
                    parsed.get("formula")
                    or parsed.get("calculation")
                    or ""
                ),
                "applies_to": (
                    parsed.get("applies_to")
                    or parsed.get("applicable_items")
                    or parsed.get("items")
                    or "all"
                ),
                "notes": (
                    parsed.get("notes")
                    or parsed.get("conditions")
                    or parsed.get("caveats")
                    or ""
                ),
                "periods": parsed.get("periods", []),
                "cpi_series_id": parsed.get("cpi_series_id", ""),
            }

            # Validate: if LLM says "none", "n/a", or "not applicable"
            # for applies_to, treat as "all"
            if normalised["applies_to"].lower() in (
                "none", "n/a", "not applicable", "not specified", ""
            ):
                normalised["applies_to"] = "all"

            # Clean formula_type: strip surrounding quotes, pick first option
            ft = normalised["formula_type"].strip().strip('"').strip("'")
            if "|" in ft:
                ft = ft.split("|")[0].strip()
            normalised["formula_type"] = ft

            # If LLM says no formula found, return None
            if ft.lower() in ("none", "n/a", "not applicable", "not found", ""):
                logger.info(
                    f"LLM determined no adjustment formula in this clause: "
                    f"{normalised.get('description', '')}"
                )
                return None

            return normalised

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {e}")
            logger.error(f"Response was: {response[:500]}")
            return None
        except Exception as e:
            logger.error(f"LLM interpretation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Price adjustment
    # ------------------------------------------------------------------

    def _apply_adjustments(
        self,
        pricing_df: pd.DataFrame,
        formulas: List[Dict],
        cpi_df: pd.DataFrame,
        document_id: str,
    ) -> pd.DataFrame:
        new_rows = []
        latest_cpi_date = cpi_df["date"].max()

        for formula in formulas:
            cpi_col = formula.get("cpi_series", "")
            formula_type = formula.get("formula_type", "cpi_ratio")
            applies_to = formula.get("applies_to", "all")
            periods = formula.get("periods", [])

            # Also check for cpi_series_id to match by series ID
            cpi_series_id = formula.get("cpi_series_id", "")

            matched_col = self._match_cpi_column(cpi_df, cpi_col)
            if not matched_col:
                logger.warning(
                    f"[{document_id}] CPI series not found: {cpi_col}"
                )
                continue

            # Filter target rows by applies_to
            target_rows = self._filter_target_rows(
                pricing_df, applies_to, document_id
            )

            logger.info(
                f"[{document_id}]   Adjusting {len(target_rows)} rows "
                f"with {matched_col}, {len(periods)} period(s)"
            )

            if len(target_rows) > 0:
                sample_prices = target_rows["NET_PRICE"].head(3).tolist()
                sample_dates = target_rows["VALID_FROM"].head(3).tolist()
                logger.info(
                    f"[{document_id}]   Sample NET_PRICE: {sample_prices}"
                )
                logger.info(
                    f"[{document_id}]   Sample VALID_FROM: {sample_dates}"
                )

            if periods:
                # Track last Pn and CPIn from previous period
                # so they can be used as Po/CPIo for the next period
                prev_period_last_pn = None
                prev_period_last_cpin = None

                # Use explicit periods from the contract clause
                for pi, period in enumerate(periods):
                    period_start = self._parse_date(period.get("period_start", ""))
                    period_end = self._parse_date(period.get("period_end", ""))
                    base_price_override = period.get("base_price")
                    base_cpi_override = period.get("base_cpi")
                    period_formula = period.get("formula", "")
                    adj_frequency = period.get(
                        "adjustment_frequency",
                        formula.get("adjustment_frequency", "quarterly")
                    )
                    period_notes = period.get("notes", "")

                    # Determine locked vs rolling BEFORE any carry-forward
                    po_is_locked = base_price_override is not None
                    cpio_is_locked = base_cpi_override is not None

                    if period_end is None:
                        period_end = latest_cpi_date

                    logger.info(
                        f"[{document_id}]   Period {pi + 1}: "
                        f"{period_start.strftime(DATE_FMT) if period_start else '?'} "
                        f"to {period_end.strftime(DATE_FMT) if period_end else '?'}, "
                        f"Po={base_price_override}, CPIo={base_cpi_override}, "
                        f"freq={adj_frequency}"
                    )

                    if period_start is None:
                        logger.warning(f"[{document_id}]     No period_start, skipping")
                        continue

                    # Generate adjustment dates within this period
                    adj_dates = self._get_period_dates(
                        period_start, period_end, adj_frequency
                    )

                    logger.info(
                        f"[{document_id}]     {len(adj_dates)} adjustment date(s): "
                        f"{[d.strftime(DATE_FMT) for d in adj_dates[:5]]}"
                        f"{'...' if len(adj_dates) > 5 else ''}"
                    )

                    if po_is_locked:
                        logger.info(
                            f"[{document_id}]     Po LOCKED at "
                            f"{base_price_override} for entire period"
                        )
                        try:
                            locked_po = float(base_price_override)
                        except (ValueError, TypeError):
                            logger.warning(f"[{document_id}]     Cannot parse locked Po, skipping")
                            continue
                    else:
                        logger.info(
                            f"[{document_id}]     Po ROLLING: Pn becomes "
                            f"next quarter's Po"
                        )

                    if cpio_is_locked:
                        logger.info(
                            f"[{document_id}]     CPIo LOCKED at "
                            f"{base_cpi_override} for entire period"
                        )
                        try:
                            locked_cpio = float(base_cpi_override)
                        except (ValueError, TypeError):
                            logger.warning(f"[{document_id}]     Cannot parse locked CPIo, skipping")
                            continue
                    else:
                        logger.info(
                            f"[{document_id}]     CPIo ROLLING: looked up "
                            f"as (quarter_start - 3 months) from ABS data"
                        )

                    for _, row in target_rows.iterrows():
                        # Init rolling values for this row
                        if not po_is_locked:
                            if prev_period_last_pn is not None:
                                rolling_po = prev_period_last_pn
                            else:
                                rolling_po = self._parse_price(
                                    row.get("NET_PRICE", "")
                                )
                                if rolling_po is None:
                                    continue

                        if not cpio_is_locked:
                            if prev_period_last_cpin is not None:
                                rolling_cpio = prev_period_last_cpin
                            else:
                                rolling_cpio = self._lookup_cpi_prior_quarter(
                                    cpi_df, matched_col, adj_dates[0]
                                )

                        for di, q_date in enumerate(adj_dates):
                            # --- CPIn: ALWAYS looked up as (quarter_start - 3 months) ---
                            current_cpi = self._lookup_cpi_prior_quarter(
                                cpi_df, matched_col, q_date
                            )
                            if current_cpi is None:
                                continue

                            if po_is_locked and cpio_is_locked:
                                # ── Case 1: LOCKED ──
                                # Po and CPIo fixed for entire period
                                base_price = locked_po
                                base_cpi = locked_cpio
                            else:
                                # ── Case 2: ROLLING ──
                                if di == 0:
                                    # Q1: Po = last Pn from previous period
                                    #     CPIo = last CPIn from previous period
                                    base_price = rolling_po
                                    base_cpi = rolling_cpio
                                else:
                                    # Q2+: Po = previous Pn, CPIo = previous CPIn
                                    base_price = rolling_po
                                    base_cpi = rolling_cpio

                            if base_cpi is None or base_cpi == 0:
                                continue

                            # --- Calculate Pn ---
                            adjusted_price = self._calculate_price(
                                formula_type, base_price, base_cpi,
                                current_cpi
                            )

                            # --- Rolling updates for next quarter ---
                            if not po_is_locked:
                                rolling_po = adjusted_price
                            if not cpio_is_locked:
                                rolling_cpio = current_cpi

                            # Track for carry-forward to next period
                            prev_period_last_pn = adjusted_price
                            prev_period_last_cpin = current_cpi

                            # --- Build output row ---
                            if di + 1 < len(adj_dates):
                                valid_to = adj_dates[di + 1]
                            else:
                                valid_to = period_end

                            new_row = row.copy()
                            new_row["VALID_FROM"] = q_date.strftime(DATE_FMT)
                            new_row["VALID_TO"] = valid_to.strftime(DATE_FMT)
                            new_row["NET_PRICE"] = str(round(adjusted_price, 2))
                            new_row["CLAUSE_DESCRIPTION"] = str(
                                f"CPI Adjusted ({formula_type}) | "
                                f"Po: {base_price:.2f} | "
                                f"CPIo: {base_cpi:.1f} | "
                                f"CPIn: {current_cpi:.1f} | "
                                f"Pn: {adjusted_price:.2f} | "
                                f"Period: {period_start.strftime(DATE_FMT)}-"
                                f"{period_end.strftime(DATE_FMT)}"
                            )
                            if period_formula:
                                new_row["CLAUSE_DESCRIPTION"] += (
                                    f" | Formula: {period_formula}"
                                )
                            new_rows.append(new_row)

            else:
                # Fallback: no explicit periods, use row dates
                logger.info(
                    f"[{document_id}]   No explicit periods, "
                    f"using row VALID_FROM dates"
                )
                adj_frequency = formula.get("adjustment_frequency", "quarterly")
                skipped = {"no_date": 0, "no_cpi": 0, "no_price": 0, "no_quarters": 0}

                for _, row in target_rows.iterrows():
                    base_date = self._parse_date(row.get("VALID_FROM", ""))
                    if base_date is None:
                        skipped["no_date"] += 1
                        continue

                    base_cpi = self._lookup_cpi(cpi_df, matched_col, base_date)
                    if base_cpi is None or base_cpi == 0:
                        skipped["no_cpi"] += 1
                        continue

                    base_price = self._parse_price(row.get("NET_PRICE", ""))
                    if base_price is None:
                        skipped["no_price"] += 1
                        continue

                    quarters = self._get_period_dates(
                        base_date, latest_cpi_date, adj_frequency
                    )
                    if len(quarters) <= 1:
                        skipped["no_quarters"] += 1
                        continue

                    for di, q_date in enumerate(quarters):
                        current_cpi = self._lookup_cpi(cpi_df, matched_col, q_date)
                        if current_cpi is None:
                            continue

                        adjusted_price = self._calculate_price(
                            formula_type, base_price, base_cpi, current_cpi
                        )

                        if di + 1 < len(quarters):
                            valid_to = quarters[di + 1]
                        else:
                            valid_to = q_date + timedelta(days=89)

                        new_row = row.copy()
                        new_row["VALID_FROM"] = q_date.strftime(DATE_FMT)
                        new_row["VALID_TO"] = valid_to.strftime(DATE_FMT)
                        new_row["NET_PRICE"] = str(round(adjusted_price, 2))
                        new_row["CLAUSE_DESCRIPTION"] = str(
                            f"CPI Adjusted ({formula_type}) | "
                            f"Base CPI: {base_cpi:.1f} | "
                            f"Current CPI: {current_cpi:.1f} | "
                            f"Base Price: {base_price:.2f}"
                        )
                        new_rows.append(new_row)

                if any(v > 0 for v in skipped.values()):
                    logger.info(
                        f"[{document_id}]   Skipped: {skipped}"
                    )

        if new_rows:
            adjusted_df = pd.DataFrame(new_rows)
            result = pd.concat([pricing_df, adjusted_df], ignore_index=True)
            logger.info(f"[{document_id}] Added {len(new_rows)} adjusted rows")
            return result

        logger.info(f"[{document_id}] No adjusted rows generated")
        return pricing_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_target_rows(
        self, pricing_df: pd.DataFrame, applies_to: str, document_id: str
    ) -> pd.DataFrame:
        """Filter pricing rows based on applies_to keywords."""
        if not applies_to or applies_to.lower() == "all":
            return pricing_df

        keywords = self._extract_filter_keywords(applies_to)
        if not keywords:
            return pricing_df

        patterns = []
        for kw in keywords:
            if len(kw) <= 3:
                patterns.append(re.compile(
                    r"(?:^|[\s,;/\-\(\)])" +
                    re.escape(kw) +
                    r"(?:$|[\s,;/\-\(\)])",
                    re.IGNORECASE,
                ))
            else:
                patterns.append(kw)

        def _matches(desc: str) -> bool:
            desc_lower = desc.lower()
            for pat in patterns:
                if isinstance(pat, str):
                    if pat in desc_lower:
                        return True
                else:
                    if pat.search(desc_lower):
                        return True
            return False

        mask = pricing_df["ITEM_DESCRIPTION"].fillna("").apply(_matches)
        target_rows = pricing_df[mask]
        logger.info(f"[{document_id}]   Filter keywords: {keywords}")

        if len(target_rows) == 0:
            logger.warning(
                f"[{document_id}]   No rows matched filter keywords, "
                f"falling back to ALL rows"
            )
            return pricing_df

        return target_rows

    @staticmethod
    def _parse_price(val: Any) -> Optional[float]:
        """Parse a price string, stripping currency symbols and commas."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            price_str = re.sub(r"[^\d.\-]", "", str(val).strip())
            if not price_str:
                return None
            return float(price_str)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_filter_keywords(applies_to: str) -> List[str]:
        """
        Extract meaningful keywords from the LLM's applies_to text
        for matching against ITEM_DESCRIPTION.

        e.g. "Ammonium Nitrate prices (including AN MPC and AN NDC via
        AN MPC Price) and Emulsion (Titan 2000) prices"
        → ["ammonium nitrate", "an mpc", "an ndc", "emulsion",
            "titan 2000", "an"]
        """
        original_text = applies_to
        text = applies_to.lower()

        # Noise words to strip from phrases
        noise = {
            "prices", "price", "including", "via", "the", "for",
            "of", "with", "from", "to", "in", "on", "all", "any",
            "such", "as", "related", "applicable", "based",
        }

        # Step 1: Extract terms inside parentheses
        paren_terms = re.findall(r"\(([^)]+)\)", text)

        # Step 2: Process main text (without parenthetical content)
        main_text = re.sub(r"\([^)]*\)", " ", text)

        # Split by common delimiters
        parts = re.split(r"[,;/&]+|\band\b|\bor\b", main_text)

        keywords = []
        for part in parts:
            words = part.strip().split()
            cleaned = [w for w in words if w not in noise]
            if cleaned:
                phrase = " ".join(cleaned).strip()
                if phrase:
                    keywords.append(phrase)

        # Step 3: Add parenthetical terms
        for term in paren_terms:
            words = term.strip().lower().split()
            cleaned = [w for w in words if w not in noise]
            if cleaned:
                keywords.append(" ".join(cleaned))

        # Step 4: Extract acronyms (2-4 uppercase letters in original text)
        acronyms = re.findall(r"\b[A-Z]{2,4}\b", original_text)
        for acr in acronyms:
            acr_lower = acr.lower()
            if acr_lower not in noise and acr_lower not in ("cpi",):
                keywords.append(acr_lower)

        # Step 5: Extract known product-name patterns
        # Short uppercase words followed by other words (e.g., "AN MPC")
        acronym_phrases = re.findall(
            r"\b([A-Z]{2,4}(?:\s+[A-Z][a-zA-Z]*)*)\b", original_text
        )
        for phrase in acronym_phrases:
            p = phrase.lower().strip()
            if p and p not in noise and p not in ("cpi",):
                keywords.append(p)

        # Deduplicate preserving order
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen and kw:
                seen.add(kw)
                unique.append(kw)

        return unique

    @staticmethod
    def _extract_text_recursive(data: Any) -> str:
        """Recursively pull text from nested dicts/lists."""
        if isinstance(data, str):
            # Skip structural labels
            if data.strip() in (
                "paragraph", "subsection", "section_start",
                "table", "heading", "list_item", "bullet",
            ):
                return ""
            # Skip bare single numbers (section numbering)
            if data.strip().isdigit() and len(data.strip()) <= 2:
                return ""
            return data
        if isinstance(data, dict):
            # If dict has a 'text' key, prefer that
            if "text" in data:
                val = data["text"]
                if isinstance(val, str) and val.strip():
                    return val.strip()
            # Otherwise recurse all values
            return "\n".join(
                t for v in data.values()
                if (t := CPIAdjustmentAgent._extract_text_recursive(v))
            )
        if isinstance(data, list):
            return "\n".join(
                t for item in data
                if (t := CPIAdjustmentAgent._extract_text_recursive(item))
            )
        return ""

    @staticmethod
    def _match_cpi_column(cpi_df: pd.DataFrame, target: str) -> Optional[str]:
        for col in cpi_df.columns:
            if col == "date":
                continue
            if target and target.lower() in col.lower():
                return col
        # Fallback: Australia-wide Index Numbers
        for col in cpi_df.columns:
            if "Australia" in col and "Index Numbers" in col:
                return col
        return None

    @staticmethod
    def _lookup_cpi(
        cpi_df: pd.DataFrame, series_col: str, target_date: datetime
    ) -> Optional[float]:
        """
        Find the CPI value for a target date.
        Always looks up the most recent CPI record on or before target_date,
        regardless of whether the CPI data is monthly or quarterly.
        """
        lookup_date = datetime(target_date.year, target_date.month, 1)

        # Try exact match first
        row = cpi_df.loc[cpi_df["date"] == lookup_date]
        if row.empty:
            # Fall back to most recent record before target date
            earlier = cpi_df[cpi_df["date"] <= lookup_date]
            if earlier.empty:
                return None
            row = earlier.iloc[[-1]]

        val = row[series_col].values[0]
        return float(val) if pd.notna(val) else None

    @staticmethod
    def _lookup_cpi_prior_quarter(
        cpi_df: pd.DataFrame, series_col: str, quarter_start: datetime
    ) -> Optional[float]:
        """
        Look up the CPI for the quarter ending 3 months before quarter_start.

        E.g., if quarter_start = 01/07/2024, look up CPI for the quarter
        ending 3 months prior → April 2024 → which maps to the ABS
        quarter starting 01/03/2024 (March quarter).

        The ABS dates quarters as: Mar (Q1), Jun (Q2), Sep (Q3), Dec (Q4).
        So "3 months prior" from the quarter_start gives us the ABS date.
        """
        month = quarter_start.month - 3
        year = quarter_start.year
        if month <= 0:
            month += 12
            year -= 1
        lookup_date = datetime(year, month, 1)

        row = cpi_df.loc[cpi_df["date"] == lookup_date]
        if row.empty:
            earlier = cpi_df[cpi_df["date"] <= lookup_date]
            if earlier.empty:
                return None
            row = earlier.iloc[[-1]]

        val = row[series_col].values[0]
        return float(val) if pd.notna(val) else None

    @staticmethod
    def _get_period_dates(
        start: datetime, end: datetime, frequency: str = "quarterly"
    ) -> List[datetime]:
        """
        Generate adjustment dates from start up to (but not including) end.
        The end date belongs to the next period.

        For quarterly with start=01/01/2023, end=01/04/2024:
            01/01/2023, 01/04/2023, 01/07/2023, 01/10/2023, 01/01/2024
        """
        if frequency == "annually":
            step_months = 12
        elif frequency == "semi-annually":
            step_months = 6
        elif frequency == "monthly":
            step_months = 1
        else:
            step_months = 3

        dates = [start]
        current = start
        while True:
            month = current.month + step_months
            year = current.year
            while month > 12:
                month -= 12
                year += 1
            day = min(current.day, 28)
            current = datetime(year, month, day)
            if current >= end:
                break
            dates.append(current)

        return dates

    @staticmethod
    def _calculate_price(
        formula_type: str,
        base_price: float,
        base_cpi: float,
        current_cpi: float,
    ) -> float:
        if formula_type == "cpi_ratio":
            return base_price * (current_cpi / base_cpi)
        elif formula_type == "cpi_percentage":
            return base_price * (1 + (current_cpi - base_cpi) / base_cpi)
        else:
            return base_price * (current_cpi / base_cpi)

    @staticmethod
    def _parse_date(val: Any) -> Optional[datetime]:
        if pd.isna(val) or not val:
            return None
        try:
            return pd.to_datetime(val, format=DATE_FMT, dayfirst=True).to_pydatetime()
        except Exception:
            try:
                return pd.to_datetime(val, dayfirst=True).to_pydatetime()
            except Exception:
                return None

    @staticmethod
    def _build_report(
        document_id: str,
        clauses: List[Dict],
        formulas: List[Dict],
        original_rows: int,
        result_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        return {
            "document_id": document_id,
            "clauses_found": len(clauses),
            "formulas_parsed": len(formulas),
            "original_rows": original_rows,
            "adjusted_rows": len(result_df) - original_rows,
            "total_rows": len(result_df),
            "clauses": [
                {"section_name": c["section_name"], "section_type": c["section_type"]}
                for c in clauses
            ],
            "formulas": formulas,
            "result_df": result_df,
        }


# =====================================================================
# ABS CPI Data Loader (standalone utility)
# =====================================================================

def load_cpi_data(path: str) -> pd.DataFrame:
    """
    Parse ABS CPI Excel file into a clean DataFrame.
    Auto-detects quarterly (6401.0 Tables 1&2) vs monthly format.
    Stores detected frequency in df.attrs["frequency"].
    """
    xls = pd.ExcelFile(path)

    # Try Data1 sheet first (standard ABS format), fall back to first sheet
    if "Data1" in xls.sheet_names:
        sheet_name = "Data1"
    else:
        sheet_name = xls.sheet_names[0]

    df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

    # Find the header row (contains series descriptions) and data start row.
    # Standard ABS format: row 0 = descriptions, rows 1-9 = metadata,
    # data starts at row 10.
    # Other formats: scan for the first row where column 0 is a date.
    desc_row = 0
    data_start = None

    for i in range(min(20, len(df))):
        val = df.iloc[i, 0]
        if isinstance(val, (pd.Timestamp, datetime)):
            data_start = i
            break
        try:
            parsed = pd.to_datetime(val, errors="raise")
            if parsed is not pd.NaT:
                data_start = i
                break
        except Exception:
            continue

    if data_start is None:
        raise ValueError(
            f"Could not find date column in CPI file: {path}"
        )

    # Description row is typically row 0, but if data starts very early
    # use the row just before data
    if data_start > 1:
        desc_row = 0
    else:
        desc_row = 0

    descriptions = df.iloc[desc_row, 1:].tolist()
    col_names = [
        str(d).strip() if pd.notna(d) else f"series_{i}"
        for i, d in enumerate(descriptions)
    ]

    data = df.iloc[data_start:].copy()
    data.columns = ["date"] + col_names[: len(data.columns) - 1]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Auto-detect frequency from the date intervals
    if len(data) >= 3:
        diffs = data["date"].diff().dropna().dt.days
        median_diff = diffs.median()
        if median_diff < 45:
            frequency = "monthly"
        elif median_diff < 100:
            frequency = "quarterly"
        else:
            frequency = "annual"
    else:
        frequency = "quarterly"

    data.attrs["frequency"] = frequency

    logger.info(
        f"CPI data loaded: {len(data)} records, frequency={frequency}, "
        f"range={data['date'].min().strftime('%Y-%m')} to "
        f"{data['date'].max().strftime('%Y-%m')}"
    )

    return data


def get_cpi_series_names(cpi_df: pd.DataFrame) -> List[str]:
    """Return column names for CPI Index Numbers series."""
    return [
        c for c in cpi_df.columns
        if c != "date" and "Index Numbers" in str(c)
    ]