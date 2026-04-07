"""
Description Reconciler Agent — fully LLM-driven.

Reconciles inconsistent ITEM_DESCRIPTION and ORDER_UOM values across
contract amendments so that downstream GROUP BY operations work correctly.

Architecture:
  Step 1 — LLM Cleaning:
    Send all distinct raw descriptions to the LLM. Ask it to return the
    core item identity for each — stripping trailing UOM/rate qualifiers,
    currency codes, parenthetical formatting noise, and any other billing
    metadata that doesn't define the item itself.

  Step 2 — LLM Duplicate Detection:
    Send all distinct *cleaned* descriptions to the LLM. Ask it to group
    duplicates — catching abbreviations, word reordering, synonyms, and
    any domain-specific equivalences.

  Step 3 — Co-occurrence Enforcement:
    Any LLM-proposed merge where two descriptions co-occur in the same
    CONTRACT_ID is rejected (they are genuinely distinct items).

Only invisible-character sanitisation is done with regex (Step 0).
Everything semantic is handled by the LLM.
"""
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from src.tools.llm_provider import invoke_text
from src.utils import setup_logger, StorageManager

logger = setup_logger("description_reconciler")

# ── Well-known UOM normalisations (no LLM needed) ──────────────────
_UOM_CANONICAL = {
    "h": "Hour", "hr": "Hour", "hrs": "Hour", "hour": "Hour",
    "hourly": "Hour", "hours": "Hour", "per hour": "Hour",
    "$/hr": "Hour", "$/h": "Hour",
    "d": "Day", "dy": "Day", "day": "Day", "daily": "Day",
    "days": "Day", "per day": "Day", "$/day": "Day",
    "wk": "Week", "week": "Week", "weekly": "Week",
    "weeks": "Week", "per week": "Week",
    "mth": "Month", "mo": "Month", "month": "Month",
    "monthly": "Month", "months": "Month", "per month": "Month",
    "yr": "Year", "year": "Year", "annual": "Year",
    "annually": "Year", "per annum": "Year", "pa": "Year",
    "ea": "Each", "each": "Each", "per item": "Each",
    "per unit": "Each", "unit": "Each",
    "t": "Tonne", "tonne": "Tonne", "ton": "Tonne",
    "tonnes": "Tonne", "per tonne": "Tonne", "$/t": "Tonne",
    "kg": "Kilogram", "kilogram": "Kilogram", "per kg": "Kilogram",
    "ls": "Lump Sum", "lump sum": "Lump Sum", "lumpsum": "Lump Sum",
    "lump": "Lump Sum", "fixed": "Lump Sum",
    "m": "Metre", "metre": "Metre", "meter": "Metre",
    "per metre": "Metre", "per meter": "Metre",
    "m2": "Square Metre", "sqm": "Square Metre",
    "sq m": "Square Metre", "square metre": "Square Metre",
    "shift": "Shift", "per shift": "Shift",
    "trip": "Trip", "per trip": "Trip",
    "lot": "Lot", "per lot": "Lot",
    "km": "Kilometre", "kilometre": "Kilometre",
    "kilometer": "Kilometre",
}

# Batch size for LLM calls
_LLM_BATCH_SIZE = 60


class DescriptionReconciler:
    """
    Reconcile ITEM_DESCRIPTION and ORDER_UOM values across amendments.

    Fully LLM-driven: the LLM cleans descriptions AND finds duplicates.
    """

    def __init__(self):
        self.storage = StorageManager()
        self._load_config()

    def _load_config(self):
        try:
            from config.config_loader import get_task_config
            cfg = get_task_config().get("description_reconciliation", {})
            self._max_tokens = cfg.get("max_tokens", 16384)
            self._llm_batch_size = cfg.get("batch_size", _LLM_BATCH_SIZE)
        except Exception:
            self._max_tokens = 16384
            self._llm_batch_size = _LLM_BATCH_SIZE

    # ==================================================================
    # Public API
    # ==================================================================

    def reconcile(
        self,
        descriptions: List[str],
        contract_ids: List[str],
        uom_values: List[str] = None,
        document_id: str = "",
    ) -> Dict[str, Any]:
        """
        Reconcile ITEM_DESCRIPTION and ORDER_UOM values.

        Returns:
            {
                "cleaning_map": {raw: cleaned, ...},
                "item_descriptions": {variant: canonical, ...},
                "order_uom": {variant: canonical, ...},
                "stats": { ... }
            }
        """
        logger.info(
            f"[{document_id}] Reconciling {len(descriptions)} rows"
        )

        # ── Step 0: Sanitise invisible characters (regex only) ────────
        sanitised = [_sanitise_chars(d) for d in descriptions]
        sanitised_cids = [_sanitise_chars(c) for c in contract_ids]

        # ── Build description -> set of contract IDs ─────────────────
        desc_contracts = self._build_desc_contract_map(
            sanitised, sanitised_cids
        )
        distinct_raw = list(desc_contracts.keys())
        logger.info(
            f"[{document_id}] {len(distinct_raw)} distinct descriptions "
            f"(after sanitisation)"
        )

        # ── Step 1: LLM-powered cleaning ─────────────────────────────
        cleaning_map = self._llm_clean_descriptions(
            distinct_raw, document_id
        )
        logger.info(
            f"[{document_id}] Step 1 (LLM cleaning): "
            f"{len(cleaning_map)} descriptions cleaned"
        )

        # Build cleaned -> set of raw descriptions that map to it
        cleaned_to_raw: Dict[str, List[str]] = defaultdict(list)
        for raw in distinct_raw:
            cleaned = cleaning_map.get(raw, raw)
            cleaned_to_raw[cleaned].append(raw)

        # Phase 1 merges: raw descriptions that now share a cleaned form
        phase1_mapping: Dict[str, str] = {}
        for cleaned, raws in cleaned_to_raw.items():
            if len(raws) >= 2:
                # Check co-occurrence before merging
                sub_groups = self._split_by_cooccurrence(
                    raws, desc_contracts
                )
                for group in sub_groups:
                    if len(group) >= 2:
                        canonical = cleaned
                        for variant in group:
                            phase1_mapping[variant] = canonical
                            if variant != canonical:
                                logger.info(
                                    f"[{document_id}]   Clean-merge: "
                                    f"'{variant}' -> '{canonical}'"
                                )
            elif raws[0] != cleaned:
                phase1_mapping[raws[0]] = cleaned

        # Distinct cleaned descriptions for next steps
        distinct_cleaned = sorted(set(cleaned_to_raw.keys()))
        logger.info(
            f"[{document_id}] After cleaning: "
            f"{len(distinct_raw)} -> {len(distinct_cleaned)} distinct"
        )

        # ── Step 1.5: Case/punctuation normalisation merge ───────────
        # Catch trivial differences the LLM cleaning didn't unify:
        #   "IT Loader, UG" vs "It Loader Ug" (case + comma)
        #   "Leading Hand Mechanical" vs "leading hand mechanical"
        norm_mapping, distinct_after_norm = self._normalise_merge(
            distinct_cleaned, desc_contracts, cleaning_map,
            distinct_raw, document_id,
        )
        phase1_mapping.update(norm_mapping)
        logger.info(
            f"[{document_id}] After case/punct normalisation: "
            f"{len(distinct_cleaned)} -> {len(distinct_after_norm)} distinct"
        )

        # ── Step 2: LLM duplicate detection ──────────────────────────
        # Build contracts mapping for the normalised descriptions
        norm_contracts: Dict[str, Set[str]] = defaultdict(set)
        for raw in distinct_raw:
            cleaned = phase1_mapping.get(
                raw, cleaning_map.get(raw, raw)
            )
            norm_contracts[cleaned].update(
                desc_contracts.get(raw, set())
            )
        norm_contracts = dict(norm_contracts)

        phase2_mapping: Dict[str, str] = {}
        if len(distinct_after_norm) >= 2:
            phase2_mapping = self._llm_find_and_merge_duplicates(
                distinct_after_norm, norm_contracts, document_id
            )
            logger.info(
                f"[{document_id}] Step 2 (LLM duplicates): "
                f"{len(phase2_mapping)} additional mappings"
            )

        # ── Step 3: Final verification sweep ─────────────────────────
        # After all mappings, compute the current distinct set and do
        # one more aggressive LLM pass to catch anything still missed.
        # The list is now much smaller, so the LLM can focus harder.
        interim_mapping: Dict[str, str] = {}
        for raw in distinct_raw:
            cleaned = phase1_mapping.get(raw, cleaning_map.get(raw, raw))
            final = phase2_mapping.get(cleaned, cleaned)
            if raw != final:
                interim_mapping[raw] = final

        # Get current distinct descriptions after all merges
        current_distinct = sorted(set(
            interim_mapping.get(raw, phase1_mapping.get(
                raw, cleaning_map.get(raw, raw)
            ))
            for raw in distinct_raw
        ))

        # Build contracts for current distinct descriptions
        current_contracts: Dict[str, Set[str]] = defaultdict(set)
        for raw in distinct_raw:
            final = interim_mapping.get(raw, phase1_mapping.get(
                raw, cleaning_map.get(raw, raw)
            ))
            current_contracts[final].update(
                desc_contracts.get(raw, set())
            )
        current_contracts = dict(current_contracts)

        phase3_mapping: Dict[str, str] = {}
        if len(current_distinct) >= 2:
            logger.info(
                f"[{document_id}] Step 3 (final sweep): "
                f"{len(current_distinct)} distinct descriptions"
            )
            phase3_mapping = self._final_verification_sweep(
                current_distinct, current_contracts, document_id
            )
            logger.info(
                f"[{document_id}] Step 3 result: "
                f"{len(phase3_mapping)} additional mappings"
            )

        # ── Combine: raw -> final canonical ──────────────────────────
        desc_mapping: Dict[str, str] = {}
        for raw in distinct_raw:
            cleaned = phase1_mapping.get(raw, cleaning_map.get(raw, raw))
            after_p2 = phase2_mapping.get(cleaned, cleaned)
            final = phase3_mapping.get(after_p2, after_p2)
            if raw != final:
                desc_mapping[raw] = final

        # ── UOM reconciliation ───────────────────────────────────────
        uom_mapping = {}
        if uom_values:
            uom_mapping = self._reconcile_uom(uom_values, document_id)

        # ── Stats ────────────────────────────────────────────────────
        final_descs = set()
        for d in distinct_raw:
            final_descs.add(desc_mapping.get(d, d))

        result = {
            "cleaning_map": cleaning_map,
            "item_descriptions": desc_mapping,
            "order_uom": uom_mapping,
            "stats": {
                "total_rows": len(descriptions),
                "distinct_original": len(distinct_raw),
                "distinct_after_cleaning": len(distinct_cleaned),
                "distinct_after_normalisation": len(distinct_after_norm),
                "distinct_reconciled": len(final_descs),
                "cleaning_mappings": sum(
                    1 for r in distinct_raw
                    if cleaning_map.get(r, r) != r
                ),
                "duplicate_mappings": len(phase2_mapping),
                "verification_mappings": len(phase3_mapping),
                "total_mappings": len(desc_mapping),
            },
        }

        try:
            self.storage.save_reconciliation_result(document_id, result)
        except AttributeError:
            pass

        logger.info(
            f"[{document_id}] Reconciliation complete: "
            f"{len(distinct_raw)} -> {len(final_descs)} descriptions, "
            f"{len(desc_mapping)} total mappings"
        )

        return result

    # ==================================================================
    # Step 0: Invisible character sanitisation (only regex step)
    # ==================================================================

    # (see module-level _sanitise_chars function below)

    # ==================================================================
    # Step 1: LLM-powered description cleaning
    # ==================================================================

    def _llm_clean_descriptions(
        self,
        descriptions: List[str],
        document_id: str,
    ) -> Dict[str, str]:
        """
        Send descriptions to LLM to extract core item identity.

        Returns a mapping: raw_description -> cleaned_description.
        Only includes entries where cleaning changed the value.
        """
        mapping: Dict[str, str] = {}

        for batch_start in range(0, len(descriptions), self._llm_batch_size):
            batch = descriptions[
                batch_start:batch_start + self._llm_batch_size
            ]
            batch_idx = batch_start // self._llm_batch_size + 1
            total_batches = (
                (len(descriptions) + self._llm_batch_size - 1)
                // self._llm_batch_size
            )

            logger.info(
                f"[{document_id}] Cleaning batch "
                f"{batch_idx}/{total_batches}: {len(batch)} descriptions"
            )

            batch_result = self._llm_clean_batch(batch, document_id)
            mapping.update(batch_result)

        return mapping

    def _llm_clean_batch(
        self,
        descriptions: List[str],
        document_id: str,
    ) -> Dict[str, str]:
        """Ask the LLM to clean a batch of descriptions."""
        desc_list = "\n".join(
            f'  {i}. "{d}"' for i, d in enumerate(descriptions, 1)
        )

        prompt = (
            "You are a contract pricing data expert working with "
            "Australian mining and industrial contracts.\n\n"
            "TASK: For each item description below, produce a CLEAN "
            "CANONICAL form. Two descriptions that refer to the same "
            "item MUST produce the EXACT SAME cleaned output.\n\n"
            f"DESCRIPTIONS:\n{desc_list}\n\n"
            "STEP 1 — REMOVE billing/pricing noise:\n"
            "- Currency: (AUD), AUD, $, (USD), etc.\n"
            "- Billing periods: '- Rate per Week', '- Week', "
            "'- MONTH', '- per Hour', '- Hourly', '- HR', "
            "'- Day Rate', '- Annual'\n"
            "- Parentheses used as formatting: '(surface)' -> "
            "'Surface'\n\n"
            "STEP 2 — EXPAND all abbreviations to full words:\n"
            "- 'LH' -> 'Leading Hand'\n"
            "- 'Elec' -> 'Electrical'\n"
            "- 'Mech' -> 'Mechanical'\n"
            "- 'Tech' -> 'Technician'\n"
            "- 'Sup' / 'Supv' -> 'Supervisor'\n"
            "- 'Lvl' -> 'Level'\n"
            "- 'UG' -> 'Underground'\n"
            "- 'DS' -> 'Dayshift'\n"
            "- 'NS' -> 'Nightshift'\n"
            "- 'Eng' -> 'Engineer'\n"
            "- 'Sr' / 'Snr' -> 'Senior'\n"
            "- 'Jr' / 'Jnr' -> 'Junior'\n"
            "- 'Instrum' / 'Instrument' -> 'Instrumentation'\n"
            "- Any other abbreviation you recognise in this domain\n\n"
            "STEP 3 — NORMALISE format:\n"
            "- Title Case: 'leading hand electrical' -> "
            "'Leading Hand Electrical'\n"
            "- Consistent word order — put the ROLE/ITEM NAME first, "
            "then qualifiers:\n"
            "  'Electrical Leading Hand' -> "
            "'Leading Hand Electrical'\n"
            "  'Mechanical LH' -> 'Leading Hand Mechanical'\n"
            "  'Supervisor Elec' -> 'Electrical Supervisor'\n"
            "  HOWEVER for equipment, keep the natural order: "
            "'5T Excavator Surface' stays as is.\n"
            "- Remove redundant punctuation: commas, dashes "
            "between words\n"
            "- 'sheet metal' = 'sheetmetal' -> 'Sheet Metal'\n\n"
            "WHAT TO KEEP (these define DISTINCT items):\n"
            "- Shift types: Dayshift, Nightshift (NOT billing)\n"
            "- Locations: Surface, Underground\n"
            "- Levels: Level 1, Level 2, Senior, Junior\n"
            "- Sizes: 375cfm, 5T, 20T\n"
            "- Work type: Wet Hire, Dry Hire\n\n"
            "EXAMPLES:\n"
            "  'Mechanical LH - Dayshift' -> "
            "'Leading Hand Mechanical Dayshift'\n"
            "  'Leading Hand Mechanical Dayshift' -> "
            "'Leading Hand Mechanical Dayshift'\n"
            "  'Electrical LH' -> 'Leading Hand Electrical'\n"
            "  'Leading Hand (Electrical)' -> "
            "'Leading Hand Electrical'\n"
            "  'Leading Hand Elec' -> 'Leading Hand Electrical'\n"
            "  '375cfm Air Compressor (surface) - WEEK (AUD)' -> "
            "'375Cfm Air Compressor Surface'\n"
            "  '375cfm Air Compressor Surface - Rate per Week' -> "
            "'375Cfm Air Compressor Surface'\n"
            "  'Concreter - DS' -> 'Concreter Dayshift'\n"
            "  'Concreter - NS' -> 'Concreter Nightshift'\n"
            "  'IT Loader, UG' -> 'It Loader Underground'\n"
            "  'It Loader Ug' -> 'It Loader Underground'\n"
            "  'Instrument Tech' -> 'Instrumentation Technician'\n"
            "  'Supervisor Elec' -> 'Electrical Supervisor'\n\n"
            "CRITICAL: Two descriptions referring to the same item "
            "MUST map to the EXACT SAME output string. If you "
            "recognise them as the same item, produce identical "
            "output.\n\n"
            "Return ONLY a JSON object. Keys are the EXACT input "
            "strings (copy character-for-character), values are "
            "the cleaned canonical forms:\n"
            "{\n"
            '  "<exact input>": "<cleaned canonical>",\n'
            '  ...\n'
            "}\n\n"
            "Include ALL descriptions, even unchanged ones.\n"
            "Return the JSON now:"
        )

        retries = 3
        while retries > 0:
            try:
                response = invoke_text(
                    prompt=prompt, max_tokens=self._max_tokens,
                )
                return self._parse_cleaning_response(
                    response, descriptions, document_id
                )
            except Exception as e:
                error_str = str(e).lower()
                # Azure content filter — retrying same prompt won't help.
                # Split into smaller batches to isolate the problem.
                if "content_filter" in error_str or "content management policy" in error_str:
                    logger.warning(
                        f"[{document_id}] Azure content filter triggered "
                        f"on batch of {len(descriptions)} descriptions. "
                        f"Splitting into smaller batches..."
                    )
                    return self._clean_batch_with_split(
                        descriptions, document_id
                    )
                retries -= 1
                logger.warning(
                    f"[{document_id}] LLM cleaning batch failed "
                    f"({retries} retries left): {e}"
                )

        # Fallback: no cleaning
        return {}

    def _clean_batch_with_split(
        self,
        descriptions: List[str],
        document_id: str,
    ) -> Dict[str, str]:
        """
        When Azure content filter blocks a batch, split it into
        individual descriptions and process each one separately.
        Descriptions that trigger the filter are skipped (kept as-is).
        """
        if len(descriptions) <= 1:
            logger.warning(
                f"[{document_id}] Single description triggered content "
                f"filter: '{descriptions[0][:80]}...' — keeping as-is"
            )
            return {}

        # Split in half and recurse
        mid = len(descriptions) // 2
        left = descriptions[:mid]
        right = descriptions[mid:]

        result = {}
        for sub_batch in [left, right]:
            try:
                sub_result = self._llm_clean_batch(sub_batch, document_id)
                result.update(sub_result)
            except Exception as e:
                logger.warning(
                    f"[{document_id}] Sub-batch of {len(sub_batch)} "
                    f"also failed: {e}"
                )
                # If still failing, try individual descriptions
                if len(sub_batch) > 1:
                    for desc in sub_batch:
                        try:
                            individual = self._llm_clean_batch(
                                [desc], document_id
                            )
                            result.update(individual)
                        except Exception:
                            logger.warning(
                                f"[{document_id}] Skipping filtered "
                                f"description: '{desc[:80]}...'"
                            )
        return result

    def _parse_cleaning_response(
        self,
        response: str,
        descriptions: List[str],
        document_id: str,
    ) -> Dict[str, str]:
        """Parse LLM cleaning response."""
        cleaned = _strip_markdown(response)
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if not match:
            raise ValueError("No JSON object in cleaning response")

        data = json.loads(match.group(0))
        result: Dict[str, str] = {}

        # Build lookup for fuzzy key matching
        desc_set = set(d.strip() for d in descriptions)
        lower_map = {d.strip().lower(): d.strip() for d in descriptions}

        for key, val in data.items():
            key_s = str(key).strip()
            val_s = str(val).strip()

            if not val_s or val_s.lower() in ("null", "none", "n/a", ""):
                continue

            # Find the original description this key refers to
            original = None
            if key_s in desc_set:
                original = key_s
            elif key_s.lower() in lower_map:
                original = lower_map[key_s.lower()]
            else:
                # Fuzzy match
                original = _fuzzy_find(key_s, descriptions)

            if original and original != val_s:
                result[original] = val_s

        return result

    # ==================================================================
    # Step 2: LLM duplicate detection
    # ==================================================================

    def _llm_find_and_merge_duplicates(
        self,
        descriptions: List[str],
        desc_contracts: Dict[str, Set[str]],
        document_id: str,
    ) -> Dict[str, str]:
        """
        Send cleaned descriptions to LLM to find remaining duplicates.

        To avoid cross-batch blindness (where duplicates land in
        different batches and never get compared), we always send
        the FULL list of descriptions to the LLM. If the list is
        too large for a single call, we send the full list as
        reference context but ask about a subset at a time.
        """
        all_mapping: Dict[str, str] = {}

        # If small enough, single call with everything
        if len(descriptions) <= self._llm_batch_size:
            desc_contract_info = {
                d: sorted(desc_contracts.get(d, set()))
                for d in descriptions
            }
            groups = self._llm_find_duplicates(
                descriptions, desc_contract_info, document_id
            )
            for group in groups:
                valid_groups = self._split_by_cooccurrence(
                    group, desc_contracts
                )
                for valid_group in valid_groups:
                    if len(valid_group) >= 2:
                        canonical = max(valid_group, key=len)
                        for variant in valid_group:
                            if variant != canonical:
                                all_mapping[variant] = canonical
                                logger.info(
                                    f"[{document_id}]   Duplicate: "
                                    f"'{variant}' -> '{canonical}'"
                                )
            return all_mapping

        # Large list: process in batches but include the FULL list
        # as reference so the LLM can find cross-batch matches
        logger.info(
            f"[{document_id}] Large list ({len(descriptions)} descs), "
            f"using full-context batching"
        )

        # Build full reference list (compact, no contracts — just names)
        full_reference = "\n".join(
            f"  {i}. \"{d}\"" for i, d in enumerate(descriptions, 1)
        )

        for batch_start in range(0, len(descriptions), self._llm_batch_size):
            batch = descriptions[
                batch_start:batch_start + self._llm_batch_size
            ]
            batch_idx = batch_start // self._llm_batch_size + 1
            total_batches = (
                (len(descriptions) + self._llm_batch_size - 1)
                // self._llm_batch_size
            )

            logger.info(
                f"[{document_id}] Duplicate detection batch "
                f"{batch_idx}/{total_batches}: {len(batch)} descriptions"
            )

            desc_contract_info = {
                d: sorted(desc_contracts.get(d, set()))
                for d in batch
            }

            groups = self._llm_find_duplicates_with_context(
                batch, desc_contract_info,
                full_reference, len(descriptions),
                document_id,
            )

            for group in groups:
                # Validate all group members exist in the full list
                valid_members = [
                    d for d in group if d in set(descriptions)
                ]
                if len(valid_members) < 2:
                    continue
                valid_groups = self._split_by_cooccurrence(
                    valid_members, desc_contracts
                )
                for valid_group in valid_groups:
                    if len(valid_group) >= 2:
                        canonical = max(valid_group, key=len)
                        for variant in valid_group:
                            if variant != canonical:
                                all_mapping[variant] = canonical
                                logger.info(
                                    f"[{document_id}]   Duplicate: "
                                    f"'{variant}' -> '{canonical}'"
                                )

        return all_mapping

    def _llm_find_duplicates(
        self,
        descriptions: List[str],
        desc_contract_info: Dict[str, List[str]],
        document_id: str,
    ) -> List[List[str]]:
        """Ask the LLM to find duplicate descriptions."""
        desc_list = []
        for i, desc in enumerate(descriptions, 1):
            contracts = desc_contract_info.get(desc, [])
            contracts_str = ", ".join(contracts) if contracts else "unknown"
            desc_list.append(
                f'  {i}. "{desc}"  [contracts: {contracts_str}]'
            )
        desc_text = "\n".join(desc_list)

        prompt = (
            "You are a contract pricing data expert working with "
            "Australian mining and industrial contracts.\n\n"
            "TASK: Below is a list of ITEM_DESCRIPTION values that have "
            "already been cleaned. Some may STILL refer to the same "
            "item but are written differently.\n\n"
            "Your job: identify ALL groups of descriptions that refer "
            "to the SAME line item.\n\n"
            f"DESCRIPTIONS:\n{desc_text}\n\n"
            "PATTERNS TO LOOK FOR (these ARE the same item):\n"
            "- Abbreviations: 'Lvl'='Level', 'UG'='Underground', "
            "'Elec'='Electrical', 'Mech'='Mechanical', "
            "'LH'='Leading Hand', 'Tech'='Technician', "
            "'Sup'/'Supv'='Supervisor'\n"
            "- Punctuation: 'Fitter / Welder' = 'Fitter Welder'\n"
            "- Spacing: 'sheet metal' = 'sheetmetal'\n"
            "- Word order: 'Electrical Supervisor' = "
            "'Supervisor Elec'\n"
            "- Minor wording: 'Instrument Tech' = "
            "'Instrumentation Technician'\n"
            "- Commas: 'Fitter, Level 1' = 'Fitter Level 1'\n"
            "- Shift abbreviations: 'DS'='Dayshift', "
            "'NS'='Nightshift'\n"
            "- Singular/plural: 'Trade' = 'Trades'\n"
            "- Extra clarifying words that don't change the role: "
            "In Australian mining contracts, these role descriptions "
            "commonly refer to the same position even with extra "
            "words. Examples:\n"
            "  'Trade Assistant' = 'Trades Assistant Labourer' "
            "(same role, 'Labourer' is a clarifier)\n"
            "  'Trades Assistant' = 'Trade Assistant' "
            "(singular/plural)\n"
            "  'TA' = 'Trade Assistant' = 'Trades Assistant'\n"
            "  'Rigger' = 'Rigger/Scaffolder' (if no separate "
            "'Scaffolder' exists)\n"
            "- Use your domain knowledge of Australian mining/industrial "
            "contract roles and equipment to identify items that are "
            "the same despite wording differences. Think about whether "
            "a contractor would bill these as separate line items or "
            "the same one.\n\n"
            "THESE ARE DIFFERENT ITEMS — do NOT group them:\n"
            "- Different shift types: 'Concreter Dayshift' and "
            "'Concreter Nightshift' are DISTINCT items\n"
            "- Different levels: 'Fitter Level 1' and "
            "'Fitter Level 2' are DISTINCT\n"
            "- Different locations: 'Excavator Surface' and "
            "'Excavator Underground' are DISTINCT\n"
            "- Different sizes: '5T Excavator' and "
            "'20T Excavator' are DISTINCT\n\n"
            "CRITICAL RULES:\n"
            "1. If two descriptions appear in the SAME contract "
            "(shown in [contracts: ...] brackets), they are "
            "DIFFERENT items — do NOT group them.\n"
            "2. Shift, level, location, and size qualifiers make "
            "items DISTINCT — do not merge across these.\n"
            "3. When in doubt whether two descriptions are the same "
            "item, lean towards GROUPING them — it is better to "
            "merge borderline cases than to leave duplicates.\n\n"
            "For each group, pick the BEST canonical form (most "
            "complete and formal version).\n\n"
            "Return ONLY a JSON array of groups:\n"
            "[\n"
            "  {\n"
            '    "descriptions": ["exact string 1", "exact string 2"],\n'
            '    "canonical": "best version",\n'
            '    "reason": "brief explanation"\n'
            "  }\n"
            "]\n\n"
            "Only include groups with 2+ descriptions. Omit unique "
            "descriptions entirely.\n"
            "IMPORTANT: Copy description strings exactly as they "
            "appear in the numbered list — do not edit them.\n\n"
            "Return the JSON array now:"
        )

        retries = 3
        while retries > 0:
            try:
                response = invoke_text(
                    prompt=prompt, max_tokens=self._max_tokens,
                )
                return self._parse_duplicate_response(
                    response, descriptions, document_id
                )
            except Exception as e:
                error_str = str(e).lower()
                if "content_filter" in error_str or "content management policy" in error_str:
                    logger.warning(
                        f"[{document_id}] Azure content filter triggered "
                        f"on duplicate detection batch of "
                        f"{len(descriptions)} descriptions. "
                        f"Skipping this batch."
                    )
                    return []
                retries -= 1
                logger.warning(
                    f"[{document_id}] LLM duplicate detection failed "
                    f"({retries} retries left): {e}"
                )
        return []

    def _llm_find_duplicates_with_context(
        self,
        batch: List[str],
        desc_contract_info: Dict[str, List[str]],
        full_reference: str,
        total_count: int,
        document_id: str,
    ) -> List[List[str]]:
        """
        Find duplicates for a batch, with the full description list
        as reference context so cross-batch matches can be found.
        """
        desc_list = []
        for i, desc in enumerate(batch, 1):
            contracts = desc_contract_info.get(desc, [])
            contracts_str = ", ".join(contracts) if contracts else "unknown"
            desc_list.append(
                f'  {i}. "{desc}"  [contracts: {contracts_str}]'
            )
        desc_text = "\n".join(desc_list)

        prompt = (
            "You are a contract pricing data expert working with "
            "Australian mining and industrial contracts.\n\n"
            "TASK: Find duplicates for the FOCUS descriptions below. "
            "A FULL REFERENCE LIST of all descriptions in the dataset "
            "is provided so you can find matches across the entire set.\n\n"
            f"FOCUS DESCRIPTIONS (find duplicates for these):\n"
            f"{desc_text}\n\n"
            f"FULL REFERENCE LIST ({total_count} total):\n"
            f"{full_reference}\n\n"
            "For each FOCUS description, check if any description in "
            "the FULL REFERENCE LIST is a duplicate (same item, written "
            "differently).\n\n"
            "SAME ITEM patterns:\n"
            "- Abbreviations: LH=Leading Hand, Elec=Electrical, "
            "Mech=Mechanical, Tech=Technician, DS=Dayshift, "
            "NS=Nightshift, UG=Underground, Lvl=Level, TA=Trade "
            "Assistant\n"
            "- Word order, punctuation, spacing, singular/plural\n"
            "- Extra clarifying words that don't change the role: "
            "'Trade Assistant' = 'Trades Assistant Labourer', "
            "'Rigger' = 'Rigger/Scaffolder'\n"
            "- Use your domain knowledge — would a contractor "
            "bill these as the same line item?\n"
            "- When in doubt, lean towards GROUPING\n\n"
            "DIFFERENT items: different shifts, levels, locations, "
            "sizes. Same-contract items are always different.\n\n"
            "Return ONLY a JSON array of groups:\n"
            "[\n"
            '  {"descriptions": ["exact str 1", "exact str 2"], '
            '"canonical": "best", "reason": "why"}\n'
            "]\n"
            "Copy strings exactly. Only groups with 2+. "
            "Return JSON now:"
        )

        retries = 3
        while retries > 0:
            try:
                response = invoke_text(
                    prompt=prompt, max_tokens=self._max_tokens,
                )
                # Parse against the full list of valid descriptions
                all_valid = []
                for line in full_reference.split("\n"):
                    m = re.search(r'"([^"]+)"', line)
                    if m:
                        all_valid.append(m.group(1))
                return self._parse_duplicate_response(
                    response, all_valid, document_id
                )
            except Exception as e:
                error_str = str(e).lower()
                if "content_filter" in error_str or "content management policy" in error_str:
                    return []
                retries -= 1
                logger.warning(
                    f"[{document_id}] LLM context duplicate detection "
                    f"failed ({retries} retries left): {e}"
                )
        return []

    def _parse_duplicate_response(
        self,
        response: str,
        valid_descriptions: List[str],
        document_id: str,
    ) -> List[List[str]]:
        """Parse LLM duplicate detection response into groups."""
        cleaned = _strip_markdown(response)
        match = re.search(r'\[[\s\S]*\]', cleaned)
        if not match:
            raise ValueError("No JSON array in duplicate response")

        data = json.loads(match.group(0))

        valid_set = set(d.strip() for d in valid_descriptions)
        lower_map = {
            d.strip().lower(): d.strip() for d in valid_descriptions
        }

        groups: List[List[str]] = []

        for entry in data:
            if not isinstance(entry, dict):
                continue
            raw_descs = entry.get("descriptions", [])
            if not isinstance(raw_descs, list) or len(raw_descs) < 2:
                continue

            validated: List[str] = []
            for raw in raw_descs:
                raw_s = str(raw).strip()
                if raw_s in valid_set:
                    validated.append(raw_s)
                elif raw_s.lower() in lower_map:
                    validated.append(lower_map[raw_s.lower()])
                else:
                    best = _fuzzy_find(raw_s, valid_descriptions)
                    if best:
                        validated.append(best)
                    else:
                        logger.warning(
                            f"[{document_id}] LLM returned "
                            f"'{raw_s}' — no match found, skipping"
                        )

            # Deduplicate
            seen: Set[str] = set()
            deduped: List[str] = []
            for d in validated:
                if d not in seen:
                    seen.add(d)
                    deduped.append(d)

            if len(deduped) >= 2:
                groups.append(deduped)
                logger.info(
                    f"[{document_id}]   LLM group: {deduped} "
                    f"-> '{entry.get('canonical', '')}'"
                )

        return groups

    # ==================================================================
    # Step 3: Final verification sweep — PAIRWISE comparison
    # ==================================================================

    def _final_verification_sweep(
        self,
        descriptions: List[str],
        desc_contracts: Dict[str, Set[str]],
        document_id: str,
    ) -> Dict[str, str]:
        """
        Final aggressive pass using PAIRWISE comparison.

        Instead of asking the LLM to scan a list (which misses subtle
        cases), this step:
          1. Pre-filters candidate pairs by word overlap (cheap)
          2. For each candidate pair, asks the LLM a focused yes/no:
             "Are these two descriptions the same item?"
          3. The LLM is much more accurate on binary decisions than
             on scanning large lists.

        Pairs are batched into a single LLM call for efficiency.
        """
        # Pre-filter: find candidate pairs with significant word overlap
        candidates = self._find_candidate_pairs(
            descriptions, desc_contracts
        )

        if not candidates:
            logger.info(
                f"[{document_id}] Final sweep: no candidate pairs found"
            )
            return {}

        logger.info(
            f"[{document_id}] Final sweep: {len(candidates)} "
            f"candidate pairs to verify"
        )

        # Batch candidate pairs and ask LLM yes/no for each
        mapping: Dict[str, str] = {}
        batch_size = 20  # pairs per LLM call

        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start:batch_start + batch_size]
            batch_result = self._llm_pairwise_check(
                batch, desc_contracts, document_id
            )
            mapping.update(batch_result)

        return mapping

    def _find_candidate_pairs(
        self,
        descriptions: List[str],
        desc_contracts: Dict[str, Set[str]],
    ) -> List[Tuple[str, str]]:
        """
        Find pairs of descriptions that might be duplicates.

        Uses word-set overlap: if description A's words are mostly
        contained in description B (or vice versa), they're candidates.

        Also checks the co-occurrence constraint upfront.
        """
        candidates: List[Tuple[str, str]] = []

        # Build word sets
        word_sets: Dict[str, Set[str]] = {}
        for desc in descriptions:
            words = set(desc.lower().split())
            # Remove very common noise words
            words -= {
                'the', 'a', 'an', 'and', 'or', 'of', 'for', 'in',
                'on', 'at', 'to', 'with', '-', '–',
            }
            word_sets[desc] = words

        seen: Set[Tuple[str, str]] = set()

        for i, desc_a in enumerate(descriptions):
            words_a = word_sets[desc_a]
            if not words_a:
                continue
            contracts_a = desc_contracts.get(desc_a, set())

            for j in range(i + 1, len(descriptions)):
                desc_b = descriptions[j]
                words_b = word_sets[desc_b]
                if not words_b:
                    continue

                # Skip if they co-occur in the same contract
                contracts_b = desc_contracts.get(desc_b, set())
                if contracts_a & contracts_b:
                    continue

                # Check word overlap: is one mostly contained in the other?
                intersection = words_a & words_b
                if not intersection:
                    continue

                # Containment ratio: what fraction of the SMALLER set
                # is contained in the larger set?
                smaller = min(len(words_a), len(words_b))
                containment = len(intersection) / smaller

                # Require at least 50% of the smaller set to overlap
                if containment >= 0.5:
                    pair = (desc_a, desc_b)
                    if pair not in seen:
                        seen.add(pair)
                        candidates.append(pair)

        return candidates

    def _llm_pairwise_check(
        self,
        pairs: List[Tuple[str, str]],
        desc_contracts: Dict[str, Set[str]],
        document_id: str,
    ) -> Dict[str, str]:
        """
        Ask the LLM to verify a batch of candidate pairs.

        For each pair, the LLM answers: same item or different?
        """
        pair_list = []
        for i, (a, b) in enumerate(pairs, 1):
            pair_list.append(
                f'  {i}. A: "{a}"\n'
                f'     B: "{b}"'
            )
        pairs_text = "\n".join(pair_list)

        prompt = (
            "You are a contract pricing expert for Australian mining "
            "and industrial contracts.\n\n"
            "TASK: For each pair below, determine if A and B refer to "
            "the SAME line item (same role, same equipment, same "
            "service) or DIFFERENT items.\n\n"
            f"PAIRS:\n{pairs_text}\n\n"
            "GUIDANCE:\n"
            "- Extra clarifying words often don't change the item:\n"
            "  'Trade Assistant' = 'Trades Assistant Labourer' (SAME)\n"
            "  'Rigger' = 'Rigger Scaffolder' (SAME)\n"
            "  'Boilermaker' = 'Boilermaker Welder' (SAME)\n"
            "- Singular/plural: 'Trade' = 'Trades' (SAME)\n"
            "- But different qualifiers mean DIFFERENT items:\n"
            "  'Fitter Dayshift' vs 'Fitter Nightshift' (DIFFERENT)\n"
            "  'Excavator 5T' vs 'Excavator 20T' (DIFFERENT)\n"
            "  'Fitter Level 1' vs 'Fitter Level 2' (DIFFERENT)\n\n"
            "- When in doubt, answer SAME — missed duplicates cause "
            "more harm than false merges.\n\n"
            "Return ONLY a JSON array with one entry per pair:\n"
            "[\n"
            '  {"pair": 1, "same": true, "canonical": "best version", '
            '"reason": "brief explanation"},\n'
            '  {"pair": 2, "same": false, "reason": "why different"}\n'
            "]\n\n"
            "For 'canonical', pick the most complete/formal version "
            "of the two.\n"
            "Return the JSON array now:"
        )

        retries = 3
        while retries > 0:
            try:
                response = invoke_text(
                    prompt=prompt, max_tokens=self._max_tokens,
                )
                return self._parse_pairwise_response(
                    response, pairs, desc_contracts, document_id
                )
            except Exception as e:
                error_str = str(e).lower()
                if ("content_filter" in error_str
                        or "content management policy" in error_str):
                    return {}
                retries -= 1
                logger.warning(
                    f"[{document_id}] Pairwise check failed "
                    f"({retries} retries left): {e}"
                )

        return {}

    def _parse_pairwise_response(
        self,
        response: str,
        pairs: List[Tuple[str, str]],
        desc_contracts: Dict[str, Set[str]],
        document_id: str,
    ) -> Dict[str, str]:
        """Parse pairwise LLM response into mappings."""
        cleaned = _strip_markdown(response)
        match = re.search(r'\[[\s\S]*\]', cleaned)
        if not match:
            raise ValueError("No JSON array in pairwise response")

        data = json.loads(match.group(0))
        mapping: Dict[str, str] = {}

        for entry in data:
            if not isinstance(entry, dict):
                continue

            pair_idx = entry.get("pair")
            same = entry.get("same", False)
            canonical = entry.get("canonical", "")
            reason = entry.get("reason", "")

            if not same or pair_idx is None:
                continue

            if not (1 <= pair_idx <= len(pairs)):
                continue

            desc_a, desc_b = pairs[pair_idx - 1]

            # Double-check co-occurrence constraint
            contracts_a = desc_contracts.get(desc_a, set())
            contracts_b = desc_contracts.get(desc_b, set())
            if contracts_a & contracts_b:
                logger.info(
                    f"[{document_id}]   Pairwise: '{desc_a}' and "
                    f"'{desc_b}' co-occur — skipping merge"
                )
                continue

            # Pick canonical: LLM suggestion, or longest
            if not canonical or canonical not in (desc_a, desc_b):
                canonical = max([desc_a, desc_b], key=len)

            variant = desc_a if desc_a != canonical else desc_b
            if variant != canonical:
                mapping[variant] = canonical
                logger.info(
                    f"[{document_id}]   Pairwise merge: "
                    f"'{variant}' -> '{canonical}' ({reason})"
                )

        return mapping

    # ==================================================================
    # Step 1.5: Case/punctuation normalisation merge
    # ==================================================================

    def _normalise_merge(
        self,
        distinct_cleaned: List[str],
        desc_contracts: Dict[str, Set[str]],
        cleaning_map: Dict[str, str],
        distinct_raw: List[str],
        document_id: str,
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        Merge descriptions that differ only in case, punctuation,
        or trivial whitespace. No LLM needed — pure string ops.

        This catches what the LLM cleaning step missed:
          "IT Loader, UG" vs "It Loader Ug"
          "Leading Hand Mechanical" vs "leading hand mechanical"

        Returns (mapping, distinct_after) where mapping maps raw
        descriptions to their canonical form, and distinct_after
        is the reduced list of distinct descriptions.
        """
        # Build a normalisation key: lowercase, strip punctuation,
        # collapse whitespace, sort words (order-independent)
        def _norm_key(s: str) -> str:
            s = s.lower().strip()
            s = re.sub(r'[,./\-–—&()\[\]{}:;\'\"!?]', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            # Sort words so "Leading Hand Mechanical" ==
            # "Mechanical Leading Hand"
            return ' '.join(sorted(s.split()))

        # Group cleaned descriptions by their norm key
        norm_groups: Dict[str, List[str]] = defaultdict(list)
        for desc in distinct_cleaned:
            key = _norm_key(desc)
            norm_groups[key].append(desc)

        # Build raw -> cleaned contracts for co-occurrence checks
        # (cleaned descriptions inherit contracts from their raws)
        cleaned_contracts: Dict[str, Set[str]] = defaultdict(set)
        for raw in distinct_raw:
            cleaned = cleaning_map.get(raw, raw)
            cleaned_contracts[cleaned].update(
                desc_contracts.get(raw, set())
            )

        mapping: Dict[str, str] = {}
        merged_to: Dict[str, str] = {}  # cleaned -> canonical

        for key, members in norm_groups.items():
            if len(members) < 2:
                continue

            # Check co-occurrence
            sub_groups = self._split_by_cooccurrence(
                members, dict(cleaned_contracts)
            )
            for group in sub_groups:
                if len(group) >= 2:
                    # Pick canonical: longest, or first alphabetically
                    canonical = max(group, key=len)
                    for variant in group:
                        if variant != canonical:
                            merged_to[variant] = canonical
                            logger.info(
                                f"[{document_id}]   Norm-merge: "
                                f"'{variant}' -> '{canonical}'"
                            )

        # Also map raw descriptions whose cleaned form got merged
        for raw in distinct_raw:
            cleaned = cleaning_map.get(raw, raw)
            if cleaned in merged_to:
                final = merged_to[cleaned]
                mapping[raw] = final

        # Build the reduced distinct list
        remaining = set()
        for desc in distinct_cleaned:
            remaining.add(merged_to.get(desc, desc))
        distinct_after = sorted(remaining)

        return mapping, distinct_after

    # ==================================================================
    # Co-occurrence constraint
    # ==================================================================

    @staticmethod
    def _split_by_cooccurrence(
        members: List[str],
        desc_contracts: Dict[str, Set[str]],
    ) -> List[List[str]]:
        """Split a group so no two members co-occur in the same contract."""
        sub_groups: List[List[str]] = []
        for member in members:
            member_contracts = desc_contracts.get(member, set())
            placed = False
            for group in sub_groups:
                conflict = False
                for existing in group:
                    if member_contracts & desc_contracts.get(existing, set()):
                        conflict = True
                        break
                if not conflict:
                    group.append(member)
                    placed = True
                    break
            if not placed:
                sub_groups.append([member])
        return sub_groups

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _build_desc_contract_map(
        descriptions: List[str], contract_ids: List[str],
    ) -> Dict[str, Set[str]]:
        mapping: Dict[str, Set[str]] = defaultdict(set)
        for desc, cid in zip(descriptions, contract_ids):
            d = desc.strip()
            c = str(cid).strip()
            if d:
                mapping[d].add(c)
        return dict(mapping)

    # ==================================================================
    # UOM reconciliation
    # ==================================================================

    def _reconcile_uom(
        self, uom_values: List[str], document_id: str,
    ) -> Dict[str, str]:
        distinct = set(v.strip() for v in uom_values if v and v.strip())
        mapping: Dict[str, str] = {}
        unresolved: List[str] = []

        for uom in distinct:
            canonical = _UOM_CANONICAL.get(uom.lower().strip())
            if canonical:
                if uom != canonical:
                    mapping[uom] = canonical
            else:
                unresolved.append(uom)

        if unresolved:
            logger.info(
                f"[{document_id}] {len(unresolved)} unresolved UOM, "
                f"asking LLM: {unresolved}"
            )
            llm_uom = self._reconcile_uom_with_llm(
                unresolved, document_id
            )
            mapping.update(llm_uom)

        return mapping

    def _reconcile_uom_with_llm(
        self, unresolved: List[str], document_id: str,
    ) -> Dict[str, str]:
        known = sorted(set(_UOM_CANONICAL.values()))
        prompt = (
            "You are a contract pricing expert.\n\n"
            "Normalise these UOM values to standard forms.\n\n"
            f"VALUES: {json.dumps(unresolved)}\n"
            f"STANDARD FORMS: {json.dumps(known)}\n\n"
            "Return ONLY JSON: "
            '{{"<input>": "<standard>", ...}}\n'
            "Map to itself if already standard.\n"
            "Return JSON now:"
        )
        try:
            response = invoke_text(prompt, max_tokens=1024)
            cleaned = _strip_markdown(response)
            match = re.search(r'\{[\s\S]*\}', cleaned)
            if match:
                data = json.loads(match.group(0))
                return {
                    k.strip(): str(v).strip()
                    for k, v in data.items()
                    if k.strip() != str(v).strip()
                }
        except Exception as e:
            logger.warning(f"[{document_id}] UOM LLM failed: {e}")
        return {}


# =====================================================================
# Module-level utilities
# =====================================================================

def _sanitise_chars(text: str) -> str:
    """Replace invisible/special Unicode characters with ASCII equivalents."""
    if not text:
        return text
    s = text.strip()
    s = s.replace('\u00a0', ' ')     # non-breaking space
    s = s.replace('\u200b', '')      # zero-width space
    s = s.replace('\u200c', '')      # zero-width non-joiner
    s = s.replace('\u200d', '')      # zero-width joiner
    s = s.replace('\ufeff', '')      # BOM
    s = s.replace('\u2013', '-')     # en dash
    s = s.replace('\u2014', '-')     # em dash
    s = s.replace('\u2012', '-')     # figure dash
    s = s.replace('\u2015', '-')     # horizontal bar
    s = s.replace('\u2018', "'")     # left single quote
    s = s.replace('\u2019', "'")     # right single quote
    s = s.replace('\u201c', '"')     # left double quote
    s = s.replace('\u201d', '"')     # right double quote
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _strip_markdown(response: str) -> str:
    """Strip markdown code fences from LLM response."""
    s = response.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl > 0:
            s = s[nl + 1:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _fuzzy_find(
    target: str,
    candidates: List[str],
    threshold: float = 0.85,
) -> Optional[str]:
    """Find best fuzzy match using LCS similarity."""
    target_clean = re.sub(r'\s+', ' ', target.strip().lower())
    best_match = None
    best_score = 0.0

    for cand in candidates:
        cand_clean = re.sub(r'\s+', ' ', cand.strip().lower())
        if not cand_clean or not target_clean:
            continue
        ratio = (
            min(len(target_clean), len(cand_clean))
            / max(len(target_clean), len(cand_clean))
        )
        if ratio < 0.5:
            continue

        score = _lcs_similarity(target_clean, cand_clean)
        if score > best_score:
            best_score = score
            best_match = cand.strip()

    return best_match if best_score >= threshold else None


def _lcs_similarity(a: str, b: str) -> float:
    """LCS-based string similarity ratio."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return (2.0 * prev[n]) / (m + n)