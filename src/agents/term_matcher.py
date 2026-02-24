"""
Optional Stage: Term Matching Agent - config-driven.

For each term defined in config.json → TASK.term_matching.terms,
asks the LLM which of the extracted sections are relevant to that term,
and produces a report mapping term → list of related sections with
relevance reasoning.

The step is entirely optional and controlled by the
TASK.term_matching.enabled flag.
"""
import json
import re
from typing import Dict, List, Any, Optional

from config.config_loader import (
    get_term_matching_config,
    get_prompt,
    render_prompt,
    join_prompt,
)
from config.settings import MODEL_MAX_TOKENS_EXTRACTION
from src.tools.llm_provider import invoke_text
from src.utils import setup_logger, StorageManager

logger = setup_logger("term_matcher")

# Default batch size: how many terms to evaluate per LLM call.
# Batching reduces the number of calls while keeping context focused.
_TERMS_PER_BATCH = 10


class TermMatchingAgent:
    """
    Evaluate extracted sections against a configurable list of terms.

    For each term the LLM decides which sections are related and gives
    a short justification. Results are saved as an intermediate artifact.
    """

    def __init__(self):
        self.storage = StorageManager()
        self._cfg = get_term_matching_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match_terms(
        self,
        section_jsons: List[Dict],
        document_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Run term matching across all extracted sections.

        Args:
            section_jsons: List of section result dicts (as produced by
                           SectionExtractionAgent). Each has at least
                           ``section_name`` and ``data``.
            document_id:   Identifier for logging / storage.

        Returns:
            A report dict::

                {
                    "document_id": "...",
                    "terms": {
                        "<term>": {
                            "related_sections": [
                                {
                                    "section_name": "...",
                                    "relevance": "high | medium | low",
                                    "reason": "..."
                                }
                            ]
                        }
                    },
                    "unmatched_terms": ["<terms with no related sections>"]
                }

            Or None if term matching is disabled / no terms configured.
        """
        terms = self._cfg.get("terms", [])
        if not terms:
            logger.info(f"[{document_id}] Term matching: no terms configured")
            return None

        logger.info(
            f"[{document_id}] Term matching: {len(terms)} terms "
            f"against {len(section_jsons)} sections"
        )

        # Build a compact section summary for the LLM context.
        section_summaries = self._build_section_summaries(section_jsons)

        # Process terms in batches
        all_results: Dict[str, Any] = {}
        batch_size = self._cfg.get("terms_per_batch", _TERMS_PER_BATCH)

        for i in range(0, len(terms), batch_size):
            batch = terms[i : i + batch_size]
            logger.info(
                f"[{document_id}]   Batch {i // batch_size + 1}: "
                f"terms {i + 1}-{i + len(batch)}"
            )
            batch_result = self._evaluate_batch(
                batch, section_summaries, document_id,
            )
            all_results.update(batch_result)

        # Identify unmatched terms
        unmatched = [
            t for t in terms
            if not all_results.get(t, {}).get("related_sections")
        ]

        report = {
            "document_id": document_id,
            "terms": all_results,
            "unmatched_terms": unmatched,
        }

        # Persist
        self.storage.save_term_matching_result(document_id, report)

        matched_count = len(terms) - len(unmatched)
        logger.info(
            f"[{document_id}] Term matching complete: "
            f"{matched_count}/{len(terms)} terms matched"
        )

        return report

    # ------------------------------------------------------------------
    # Section summary builder
    # ------------------------------------------------------------------

    def _build_section_summaries(
        self, section_jsons: List[Dict]
    ) -> str:
        """
        Create a concise textual summary of every section so the LLM
        can decide relevance without receiving the full extraction.

        Each section is represented by its name, type, and a truncated
        plain-text snippet of its content.
        """
        max_snippet = self._cfg.get("max_snippet_chars", 1500)
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
        """Recursively pull text from nested dicts/lists, truncated."""
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
    # LLM evaluation
    # ------------------------------------------------------------------

    def _evaluate_batch(
        self,
        terms: List[str],
        section_summaries: str,
        document_id: str,
    ) -> Dict[str, Any]:
        """
        Ask the LLM which sections relate to each term in the batch.

        Returns a dict keyed by term.
        """
        # Build prompt from config template (or fallback)
        template = get_prompt("term_matching.template")
        if not template:
            template = self._default_template()

        terms_json = json.dumps(terms, indent=2)

        prompt = render_prompt(
            template,
            terms_json=terms_json,
            section_summaries=section_summaries,
        )

        max_tokens = self._cfg.get(
            "max_tokens",
            MODEL_MAX_TOKENS_EXTRACTION,
        )

        retries = 3
        while retries > 0:
            try:
                response = invoke_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                parsed = self._parse_response(response, terms)
                return parsed
            except Exception as e:
                retries -= 1
                logger.warning(
                    f"[{document_id}] Term batch failed "
                    f"({retries} retries left): {e}"
                )

        # All retries exhausted — return empty entries
        return {t: {"related_sections": []} for t in terms}

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, response: str, expected_terms: List[str]
    ) -> Dict[str, Any]:
        """
        Parse the LLM JSON response into a dict keyed by term.

        Expected response shape::

            {
                "<term>": {
                    "related_sections": [
                        {
                            "section_name": "...",
                            "relevance": "high|medium|low",
                            "reason": "..."
                        }
                    ]
                }
            }
        """
        cleaned = response.strip()
        for pfx in ("```json", "```"):
            if cleaned.startswith(pfx):
                cleaned = cleaned[len(pfx):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise ValueError("No JSON object found in term matching response")

        data = json.loads(match.group(0))

        # Normalise: ensure every expected term has an entry
        result: Dict[str, Any] = {}
        for term in expected_terms:
            entry = data.get(term, {})
            if not isinstance(entry, dict):
                entry = {"related_sections": []}
            entry.setdefault("related_sections", [])
            # Validate each related section entry
            validated = []
            for rs in entry["related_sections"]:
                if isinstance(rs, dict) and "section_name" in rs:
                    rs.setdefault("relevance", "medium")
                    rs.setdefault("reason", "")
                    validated.append(rs)
            entry["related_sections"] = validated
            result[term] = entry

        return result

    # ------------------------------------------------------------------
    # Default prompt template (used if config doesn't define one)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_template() -> str:
        return "\n".join([
            "You are a document analysis expert.",
            "",
            "I will give you:",
            "1. A list of TERMS to search for.",
            "2. SUMMARIES of all sections extracted from a document.",
            "",
            "TERMS:",
            "{terms_json}",
            "",
            "DOCUMENT SECTIONS:",
            "{section_summaries}",
            "",
            "TASK:",
            "For each term, identify which sections are related to it.",
            "A section is 'related' if it discusses, defines, references,",
            "or is materially relevant to the term.",
            "",
            "RELEVANCE LEVELS:",
            "- high: The section directly addresses or defines the term.",
            "- medium: The section contains meaningful references to the term.",
            "- low: The section has indirect or tangential relevance.",
            "",
            "Only include sections with genuine relevance. Do NOT force-match.",
            "If no section is related to a term, return an empty list for it.",
            "",
            "Return ONLY a JSON object (no markdown, start with {):",
            "{",
            '  "<term>": {',
            '    "related_sections": [',
            "      {",
            '        "section_name": "<exact section name>",',
            '        "relevance": "high | medium | low",',
            '        "reason": "<brief explanation>"',
            "      }",
            "    ]",
            "  }",
            "}",
            "",
            "Return the JSON now:",
        ])
