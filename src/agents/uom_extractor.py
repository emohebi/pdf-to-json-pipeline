"""
Optional Stage: Unit of Measure (UOM) Extraction - config-driven.

Analyses the extracted sections of a document to identify the units
of measure used for contract pricing (e.g. Hourly, Daily, Each,
Per Tonne, Lump Sum, Per Metre, Monthly, etc.).

The agent examines pricing tables, rate schedules, and contract terms
to find every distinct UOM, the pricing context it applies to, and
which section it was found in.

Controlled by the TASK.uom_extraction.enabled flag.
"""
import json
import re
from typing import Dict, List, Any, Optional

from config.config_loader import (
    get_uom_extraction_config,
    get_prompt,
    render_prompt,
)
from config.settings import MODEL_MAX_TOKENS_EXTRACTION
from src.tools.llm_provider import invoke_text
from src.utils import setup_logger, StorageManager

logger = setup_logger("uom_extractor")


class UOMExtractor:
    """
    Extract units of measure for contract pricing from extracted sections.

    Produces a report with:
      - units_of_measure: list of distinct UOMs found, each with
        source section, pricing context, and confidence
      - summary: grouped view of UOMs
    """

    def __init__(self):
        self.storage = StorageManager()
        self._cfg = get_uom_extraction_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_uom(
        self,
        section_jsons: List[Dict],
        document_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyse extracted sections to find units of measure for pricing.

        Args:
            section_jsons: Section result dicts from extraction.
            document_id:   Identifier for logging / storage.

        Returns:
            A report dict::

                {
                    "document_id": "...",
                    "units_of_measure": [
                        {
                            "unit": "Hourly",
                            "normalised_unit": "hour",
                            "applies_to": "Labour rates for technicians",
                            "source_section": "SCHEDULE 2 - PRICING",
                            "confidence": "high",
                            "verbatim_text": "Rate ($/hr)"
                        }
                    ],
                    "distinct_units": ["hour", "day", "each", "lump_sum"],
                    "no_uom_found": false
                }

            Or None if UOM extraction is disabled.
        """
        logger.info(
            f"[{document_id}] Extracting units of measure from "
            f"{len(section_jsons)} sections"
        )

        section_summaries = self._build_section_summaries(section_jsons)

        report = self._ask_llm(
            section_summaries, document_id,
        )

        # Persist
        self.storage.save_uom_extraction_result(document_id, report)

        uoms = report.get("units_of_measure", [])
        distinct = report.get("distinct_units", [])
        if uoms:
            logger.info(
                f"[{document_id}] Found {len(uoms)} UOM reference(s), "
                f"{len(distinct)} distinct unit(s): {distinct}"
            )
        else:
            logger.info(
                f"[{document_id}] No units of measure identified"
            )

        return report

    # ------------------------------------------------------------------
    # Section summary builder
    # ------------------------------------------------------------------

    def _build_section_summaries(
        self, section_jsons: List[Dict]
    ) -> str:
        max_snippet = self._cfg.get("max_snippet_chars", 2000)
        parts: List[str] = []

        for idx, section in enumerate(section_jsons, 1):
            name = section.get("section_name", f"Section {idx}")
            stype = section.get("_metadata", {}).get(
                "section_type", "unknown"
            )
            data = section.get("data", {})
            snippet = self._extract_text_snippet(data, max_snippet)

            parts.append(
                f"--- Section {idx}: \"{name}\" (type: {stype}) ---\n"
                f"{snippet}"
            )

        return "\n\n".join(parts)

    def _extract_text_snippet(self, data: Any, max_chars: int) -> str:
        texts: List[str] = []
        self._collect_text(data, texts)
        combined = " ".join(texts)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "..."
        return combined

    def _collect_text(self, data: Any, acc: List[str]) -> None:
        if isinstance(data, str):
            stripped = data.strip()
            if stripped:
                acc.append(stripped)
        elif isinstance(data, dict):
            for key, val in data.items():
                if key.startswith("_") or key == "image":
                    continue
                self._collect_text(val, acc)
        elif isinstance(data, list):
            for item in data:
                self._collect_text(item, acc)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _ask_llm(
        self,
        section_summaries: str,
        document_id: str,
    ) -> Dict[str, Any]:
        template = get_prompt("uom_extraction.template")
        if not template:
            template = self._default_template()

        prompt = render_prompt(
            template,
            section_summaries=section_summaries,
        )

        max_tokens = self._cfg.get(
            "max_tokens", MODEL_MAX_TOKENS_EXTRACTION,
        )

        retries = 3
        while retries > 0:
            try:
                response = invoke_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                parsed = self._parse_response(response)
                parsed["document_id"] = document_id
                return parsed
            except Exception as e:
                retries -= 1
                logger.warning(
                    f"[{document_id}] UOM extraction failed "
                    f"({retries} retries left): {e}"
                )

        return {
            "document_id": document_id,
            "units_of_measure": [],
            "distinct_units": [],
            "no_uom_found": True,
        }

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: str) -> Dict[str, Any]:
        cleaned = response.strip()
        for pfx in ("```json", "```"):
            if cleaned.startswith(pfx):
                cleaned = cleaned[len(pfx):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise ValueError(
                "No JSON object found in UOM extraction response"
            )

        data = json.loads(match.group(0))

        # Normalise structure
        data.setdefault("units_of_measure", [])
        data.setdefault("distinct_units", [])
        data.setdefault("no_uom_found", False)

        # Validate each UOM entry
        validated = []
        for entry in data["units_of_measure"]:
            if isinstance(entry, dict) and entry.get("unit"):
                entry.setdefault("normalised_unit", "")
                entry.setdefault("applies_to", "")
                entry.setdefault("source_section", "")
                entry.setdefault("confidence", "medium")
                entry.setdefault("verbatim_text", "")
                validated.append(entry)
        data["units_of_measure"] = validated

        # Rebuild distinct_units from validated entries if needed
        if validated and not data["distinct_units"]:
            seen = []
            for e in validated:
                norm = e.get("normalised_unit", e["unit"]).lower()
                if norm and norm not in seen:
                    seen.append(norm)
            data["distinct_units"] = seen

        if not validated:
            data["no_uom_found"] = True

        return data

    # ------------------------------------------------------------------
    # Default prompt template
    # ------------------------------------------------------------------

    @staticmethod
    def _default_template() -> str:
        return "\n".join([
            "You are a contract pricing analysis expert.",
            "",
            "I will give you the extracted sections of a contract document.",
            "Your task is to identify every UNIT OF MEASURE (UOM) used for",
            "pricing in this contract.",
            "",
            "DOCUMENT SECTIONS:",
            "{section_summaries}",
            "",
            "WHAT TO LOOK FOR:",
            "- Rate tables, pricing schedules, or fee schedules that specify",
            "  how charges are measured (e.g. per hour, per day, each, etc.)",
            "- Contract clauses that define how pricing is calculated",
            "- Column headers in tables (e.g. 'Rate ($/hr)', 'Unit', 'UOM')",
            "- Pricing descriptions (e.g. 'Lump Sum', 'Per Tonne', 'Monthly')",
            "- Any reference to the basis on which charges are applied",
            "",
            "COMMON UNITS OF MEASURE:",
            "- Time-based: Hourly, Daily, Weekly, Monthly, Annually",
            "- Quantity-based: Each, Per Item, Per Unit, Per Lot",
            "- Weight/Volume: Per Tonne, Per Kilogram, Per Litre, Per Cubic Metre",
            "- Distance/Area: Per Metre, Per Square Metre, Per Kilometre",
            "- Fixed: Lump Sum, Fixed Fee, Flat Rate",
            "- Other: Per Trip, Per Mobilisation, Per Shift, Percentage",
            "",
            "Return ONLY a JSON object (no markdown, start with {):",
            "{",
            '  "units_of_measure": [',
            "    {",
            '      "unit": "<UOM as written in the document>",',
            '      "normalised_unit": "<standardised lowercase label, e.g. hour, day, each, lump_sum, tonne>",',
            '      "applies_to": "<what this unit prices -- e.g. Labour rates, Equipment hire, Materials>",',
            '      "source_section": "<section name where found>",',
            '      "confidence": "high | medium | low",',
            '      "verbatim_text": "<the exact text/phrase from the document where this UOM appears>"',
            "    }",
            "  ],",
            '  "distinct_units": ["<list of unique normalised_unit values>"],',
            '  "no_uom_found": false',
            "}",
            "",
            "RULES:",
            "- Include EVERY distinct UOM reference, even if the same unit",
            "  appears in multiple sections (list each occurrence).",
            "- distinct_units should be the deduplicated list of normalised_unit values.",
            "- If no pricing or UOM information exists, set no_uom_found to true.",
            "- Use the EXACT text as it appears in the document for 'unit' and 'verbatim_text'.",
            "- normalised_unit should be a short, lowercase, standardised label.",
            "",
            "Return the JSON now:",
        ])
