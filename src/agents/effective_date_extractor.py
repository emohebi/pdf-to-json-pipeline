"""
Optional Stage: Effective Date Extraction - config-driven.

Analyses the extracted sections of a document to identify the effective
date stamp -- i.e. when the original contract or a subsequent variation
should apply from.

The effective date may be:
  - Explicitly stated (e.g. "Effective Date: 1 January 2025")
  - Derived from context (e.g. Commencement Date of the original
    contract, the execution date of a variation, etc.)

The agent sends section text summaries to the LLM and asks it to
identify all candidate effective dates with confidence and reasoning.

Controlled by the TASK.effective_date.enabled flag.
"""
import json
import re
from typing import Dict, List, Any, Optional

from config.config_loader import (
    get_effective_date_config,
    get_prompt,
    render_prompt,
)
from config.settings import MODEL_MAX_TOKENS_EXTRACTION
from src.tools.llm_provider import invoke_text
from src.utils import setup_logger, StorageManager

logger = setup_logger("effective_date_extractor")


class EffectiveDateExtractor:
    """
    Extract effective date stamps from a document's extracted sections.

    Produces a report with:
      - primary_effective_date: the most likely effective date
      - all_dates_found: every candidate date with source, confidence,
        and reasoning
    """

    def __init__(self):
        self.storage = StorageManager()
        self._cfg = get_effective_date_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_effective_date(
        self,
        section_jsons: List[Dict],
        document_id: str,
        document_header: Dict = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyse extracted sections to find effective date(s).

        Args:
            section_jsons: Section result dicts from extraction.
            document_id:   Identifier for logging / storage.
            document_header: Optional document header dict (title, date,
                             etc.) for additional context.

        Returns:
            A report dict::

                {
                    "document_id": "...",
                    "primary_effective_date": {
                        "date": "1 January 2025",
                        "normalised": "2025-01-01",
                        "source_section": "COMMENCEMENT AND TERM",
                        "date_type": "commencement_date",
                        "confidence": "high",
                        "reason": "..."
                    },
                    "all_dates_found": [ ... ],
                    "no_date_found": false
                }

            Or None if effective date extraction is disabled.
        """
        logger.info(
            f"[{document_id}] Extracting effective date from "
            f"{len(section_jsons)} sections"
        )

        # Build context for the LLM
        section_summaries = self._build_section_summaries(section_jsons)
        header_context = self._build_header_context(document_header)

        report = self._ask_llm(
            section_summaries, header_context, document_id,
        )

        # Persist
        self.storage.save_effective_date_result(document_id, report)

        primary = report.get("primary_effective_date", {})
        if primary and primary.get("date"):
            logger.info(
                f"[{document_id}] Effective date: "
                f"{primary['date']} "
                f"(type: {primary.get('date_type', '?')}, "
                f"confidence: {primary.get('confidence', '?')})"
            )
        else:
            logger.info(
                f"[{document_id}] No effective date identified"
            )

        return report

    # ------------------------------------------------------------------
    # Section & header summary builders
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

    def _build_header_context(
        self, document_header: Dict = None
    ) -> str:
        if not document_header:
            return ""

        parts = ["DOCUMENT HEADER:"]
        for key, val in document_header.items():
            if key == "sections":
                continue
            if isinstance(val, dict):
                text = val.get("text", "")
            elif isinstance(val, str):
                text = val
            else:
                continue
            if text:
                parts.append(f"  {key}: {text}")

        return "\n".join(parts) if len(parts) > 1 else ""

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
        header_context: str,
        document_id: str,
    ) -> Dict[str, Any]:
        template = get_prompt("effective_date.template")
        if not template:
            template = self._default_template()

        prompt = render_prompt(
            template,
            section_summaries=section_summaries,
            header_context=header_context,
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
                    f"[{document_id}] Effective date extraction "
                    f"failed ({retries} retries left): {e}"
                )

        return {
            "document_id": document_id,
            "primary_effective_date": {},
            "all_dates_found": [],
            "no_date_found": True,
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
                "No JSON object found in effective date response"
            )

        data = json.loads(match.group(0))

        # Normalise structure
        data.setdefault("primary_effective_date", {})
        data.setdefault("all_dates_found", [])
        data.setdefault("no_date_found", False)

        primary = data["primary_effective_date"]
        if isinstance(primary, dict):
            primary.setdefault("date", "")
            primary.setdefault("normalised", "")
            primary.setdefault("source_section", "")
            primary.setdefault("date_type", "")
            primary.setdefault("confidence", "low")
            primary.setdefault("reason", "")

        # Validate each candidate
        validated = []
        for entry in data["all_dates_found"]:
            if isinstance(entry, dict) and entry.get("date"):
                entry.setdefault("normalised", "")
                entry.setdefault("source_section", "")
                entry.setdefault("date_type", "")
                entry.setdefault("confidence", "low")
                entry.setdefault("reason", "")
                validated.append(entry)
        data["all_dates_found"] = validated

        # If primary is empty but we have candidates, promote the
        # highest-confidence one
        if not primary.get("date") and validated:
            confidence_order = {"high": 0, "medium": 1, "low": 2}
            best = min(
                validated,
                key=lambda d: confidence_order.get(
                    d.get("confidence", "low"), 3
                ),
            )
            data["primary_effective_date"] = dict(best)
            data["no_date_found"] = False

        if not data["primary_effective_date"].get("date"):
            data["no_date_found"] = True

        return data

    # ------------------------------------------------------------------
    # Default prompt template
    # ------------------------------------------------------------------

    @staticmethod
    def _default_template() -> str:
        return "\n".join([
            "You are a contract analysis expert.",
            "",
            "I will give you the extracted sections of a document.",
            "Your task is to identify the EFFECTIVE DATE -- the date from",
            "which the contract or variation should apply.",
            "",
            "{header_context}",
            "",
            "DOCUMENT SECTIONS:",
            "{section_summaries}",
            "",
            "WHAT TO LOOK FOR:",
            "1. An explicitly stated 'Effective Date', 'Commencement Date',",
            "   or 'Start Date' defined in the contract terms.",
            "2. The execution or signing date of the contract or variation.",
            "3. A date derived from context (e.g. 'this Deed takes effect",
            "   from the Commencement Date defined in clause X').",
            "4. Any date that indicates when the agreement's terms begin",
            "   to apply.",
            "",
            "DATE TYPES (use these labels):",
            "- effective_date: Explicitly labelled as 'Effective Date'",
            "- commencement_date: A 'Commencement Date' or 'Start Date'",
            "- execution_date: The date the document was signed/executed",
            "- variation_date: The date a variation takes effect",
            "- other: Any other date that serves as the effective date",
            "",
            "Return ONLY a JSON object (no markdown, start with {):",
            "{",
            '  "primary_effective_date": {',
            '    "date": "<date as written in the document>",',
            '    "normalised": "<YYYY-MM-DD format, or empty if ambiguous>",',
            '    "source_section": "<section name where found>",',
            '    "date_type": "<one of the types above>",',
            '    "confidence": "high | medium | low",',
            '    "reason": "<brief explanation>"',
            "  },",
            '  "all_dates_found": [',
            "    {",
            '      "date": "<date text>",',
            '      "normalised": "<YYYY-MM-DD or empty>",',
            '      "source_section": "<section name>",',
            '      "date_type": "<type>",',
            '      "confidence": "high | medium | low",',
            '      "reason": "<why this could be the effective date>"',
            "    }",
            "  ],",
            '  "no_date_found": false',
            "}",
            "",
            "RULES:",
            "- primary_effective_date should be the MOST LIKELY effective date.",
            "- all_dates_found should include ALL candidate dates, even if",
            "  low confidence.",
            "- If no date can be identified at all, set no_date_found to true",
            "  and leave primary_effective_date fields empty.",
            "- Use the EXACT date text as it appears in the document.",
            "- normalised should be ISO format YYYY-MM-DD where possible.",
            "",
            "Return the JSON now:",
        ])
