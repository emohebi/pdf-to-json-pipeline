"""
Optional Stage: Extraction Verifier - config-driven.

Takes an extraction file (CSV, Excel, or JSON) with a list of
"information" items and their associated "page numbers", plus a PDF.

For each item, the verifier:
  1. Extracts the relevant PDF page image(s).
  2. Sends the image(s) + the claimed information to the LLM.
  3. Asks: "Is this information correct, incorrect, or missing?"
  4. Records the verdict + reasoning.

Output:
  - Excel report (one row per item, with verdict + reasoning)
  - JSON accuracy summary

All prompts and field mappings are configurable via config.json under
TASK.extraction_verification.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.config_loader import (
    get_task_config,
    get_prompt,
    render_prompt,
    join_prompt,
)
from config.settings import MODEL_MAX_TOKENS_VALIDATION
from src.tools.llm_provider import invoke_multimodal
from src.tools.bedrock_vision import prepare_images_for_bedrock
from src.utils import setup_logger, StorageManager

logger = setup_logger("extraction_verifier")

# Verdicts
VERDICT_CORRECT = "CORRECT"
VERDICT_INCORRECT = "INCORRECT"
VERDICT_MISSING = "MISSING"
VERDICT_UNVERIFIABLE = "UNVERIFIABLE"

ALL_VERDICTS = (VERDICT_CORRECT, VERDICT_INCORRECT, VERDICT_MISSING, VERDICT_UNVERIFIABLE)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_verification_config() -> Dict[str, Any]:
    return get_task_config().get("extraction_verification", {})


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class ExtractionVerifier:
    """
    Verify extracted information items against the source PDF pages.

    Each item is sent to the LLM alongside the relevant page image(s).
    The LLM returns a structured verdict (CORRECT / INCORRECT / MISSING /
    UNVERIFIABLE) and a brief reasoning string.
    """

    def __init__(self):
        self.storage = StorageManager()
        self._cfg = get_verification_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        items: List[Dict[str, Any]],
        pages_data: List[Dict],
        document_id: str,
    ) -> Dict[str, Any]:
        """
        Verify a list of extraction items against PDF pages.

        Args:
            items: List of dicts. Each must have at least the keys
                   configured in extraction_verification.fields:
                   - information_field  (e.g. "information")
                   - page_field         (e.g. "page_number")
                   Plus any extra columns to carry through.
            pages_data: Full list of page dicts from pdf_processor.
            document_id: Identifier for logging / storage.

        Returns:
            {
                "document_id": "...",
                "results": [ { ...original_item, "verdict": "...",
                               "reasoning": "..." } ],
                "accuracy": {
                    "total": N,
                    "correct": N, "incorrect": N,
                    "missing": N, "unverifiable": N,
                    "accuracy_pct": 0.0–100.0
                }
            }
        """
        cfg = self._cfg
        info_field = cfg.get("information_field", "information")
        page_field = cfg.get("page_field", "page_number")
        context_pages = cfg.get("context_pages", 0)   # extra pages ± around target
        total_pages = len(pages_data)

        logger.info(
            f"[{document_id}] Verifying {len(items)} item(s) "
            f"against {total_pages} PDF pages"
        )

        results = []
        for idx, item in enumerate(items, 1):
            information = str(item.get(info_field, "")).strip()
            raw_page = item.get(page_field)

            if not information:
                logger.warning(f"  [{idx}] Empty information field — skipping")
                results.append({
                    **item,
                    "verdict": VERDICT_UNVERIFIABLE,
                    "reasoning": "No information provided.",
                })
                continue

            # Parse page number(s) — may be "5", "5-7", "5,6,7"
            page_nums = self._parse_pages(raw_page, total_pages)
            if not page_nums:
                logger.warning(
                    f"  [{idx}] Invalid or missing page number "
                    f"'{raw_page}' — marking UNVERIFIABLE"
                )
                results.append({
                    **item,
                    "verdict": VERDICT_UNVERIFIABLE,
                    "reasoning": f"Page number '{raw_page}' is invalid or out of range (document has {total_pages} pages).",
                })
                continue

            # Expand with context pages
            expanded = self._expand_page_range(
                page_nums, context_pages, total_pages
            )

            logger.info(
                f"  [{idx}/{len(items)}] Pages {expanded} | "
                f"'{information[:60]}{'...' if len(information) > 60 else ''}'"
            )

            verdict, reasoning = self._verify_item(
                information, expanded, pages_data, document_id, idx,
            )

            results.append({
                **item,
                "verdict": verdict,
                "reasoning": reasoning,
            })

        # Compute accuracy
        accuracy = self._compute_accuracy(results)
        report = {
            "document_id": document_id,
            "results": results,
            "accuracy": accuracy,
        }

        # Save
        self.storage.save_verification_result(document_id, report)

        logger.info(
            f"[{document_id}] Verification complete: "
            f"{accuracy['correct']}/{accuracy['total']} correct "
            f"({accuracy['accuracy_pct']:.1f}%)"
        )
        return report

    # ------------------------------------------------------------------
    # LLM verification
    # ------------------------------------------------------------------

    def _verify_item(
        self,
        information: str,
        page_nums: List[int],
        pages_data: List[Dict],
        document_id: str,
        item_idx: int,
    ) -> Tuple[str, str]:
        """Ask the LLM whether `information` is correct on the given pages."""
        page_dicts = [pages_data[p - 1] for p in page_nums]
        images = prepare_images_for_bedrock(page_dicts)

        template = get_prompt("extraction_verification.template")
        if not template:
            template = self._default_template()

        page_label = (
            str(page_nums[0])
            if len(page_nums) == 1
            else f"{page_nums[0]}-{page_nums[-1]}"
        )

        prompt = render_prompt(
            template,
            information=information,
            page_numbers=page_label,
            n_images=len(images),
        )

        max_tokens = self._cfg.get("max_tokens", MODEL_MAX_TOKENS_VALIDATION)
        retries = self._cfg.get("max_retries", 3)

        while retries > 0:
            try:
                response = invoke_multimodal(
                    images=images,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                verdict, reasoning = self._parse_response(response)
                return verdict, reasoning
            except Exception as e:
                retries -= 1
                logger.warning(
                    f"[{document_id}] Item {item_idx} verification "
                    f"failed ({retries} retries left): {e}"
                )

        return VERDICT_UNVERIFIABLE, "Verification failed after all retries."

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM JSON response into (verdict, reasoning)."""
        cleaned = response.strip()
        for pfx in ("```json", "```"):
            if cleaned.startswith(pfx):
                cleaned = cleaned[len(pfx):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise ValueError(f"No JSON in response: {cleaned[:200]}")

        data = json.loads(match.group(0))

        raw_verdict = str(data.get("verdict", "")).upper().strip()
        # Normalise common synonyms
        synonym_map = {
            "CORRECT": VERDICT_CORRECT,
            "RIGHT": VERDICT_CORRECT,
            "ACCURATE": VERDICT_CORRECT,
            "INCORRECT": VERDICT_INCORRECT,
            "WRONG": VERDICT_INCORRECT,
            "INACCURATE": VERDICT_INCORRECT,
            "ERROR": VERDICT_INCORRECT,
            "MISSING": VERDICT_MISSING,
            "NOT FOUND": VERDICT_MISSING,
            "ABSENT": VERDICT_MISSING,
            "UNVERIFIABLE": VERDICT_UNVERIFIABLE,
            "UNCLEAR": VERDICT_UNVERIFIABLE,
            "CANNOT VERIFY": VERDICT_UNVERIFIABLE,
        }
        verdict = synonym_map.get(raw_verdict, VERDICT_UNVERIFIABLE)
        reasoning = str(data.get("reasoning", "")).strip()
        return verdict, reasoning

    # ------------------------------------------------------------------
    # Accuracy computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_accuracy(results: List[Dict]) -> Dict[str, Any]:
        total = len(results)
        counts = {v: 0 for v in ALL_VERDICTS}
        for r in results:
            v = r.get("verdict", VERDICT_UNVERIFIABLE)
            if v in counts:
                counts[v] += 1

        # Accuracy = correct / (correct + incorrect + missing)
        # Unverifiable items are excluded from the denominator
        verifiable = counts[VERDICT_CORRECT] + counts[VERDICT_INCORRECT] + counts[VERDICT_MISSING]
        acc_pct = (
            round(counts[VERDICT_CORRECT] / verifiable * 100, 2)
            if verifiable > 0 else 0.0
        )
        return {
            "total": total,
            "correct": counts[VERDICT_CORRECT],
            "incorrect": counts[VERDICT_INCORRECT],
            "missing": counts[VERDICT_MISSING],
            "unverifiable": counts[VERDICT_UNVERIFIABLE],
            "verifiable": verifiable,
            "accuracy_pct": acc_pct,
        }

    # ------------------------------------------------------------------
    # Page parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_pages(raw: Any, total_pages: int) -> List[int]:
        """Parse 'page_number' into a list of 1-based ints."""
        if raw is None:
            return []
        s = str(raw).strip()
        pages: List[int] = []

        # "5-7"
        range_m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", s)
        if range_m:
            a, b = int(range_m.group(1)), int(range_m.group(2))
            pages = list(range(a, b + 1))
        else:
            # "5,6,7" or "5; 6; 7" or just "5"
            for part in re.split(r"[,;\s]+", s):
                if part.isdigit():
                    pages.append(int(part))

        # Filter to valid range
        return [p for p in pages if 1 <= p <= total_pages]

    @staticmethod
    def _expand_page_range(
        page_nums: List[int], context: int, total_pages: int
    ) -> List[int]:
        """Expand page list by ±context pages, clamp to valid range."""
        if not page_nums or context == 0:
            return page_nums
        mn = max(1, min(page_nums) - context)
        mx = min(total_pages, max(page_nums) + context)
        return list(range(mn, mx + 1))

    # ------------------------------------------------------------------
    # Default prompt template
    # ------------------------------------------------------------------

    @staticmethod
    def _default_template() -> str:
        return "\n".join([
            "You are a document verification expert.",
            "",
            "I am showing you {n_images} page image(s) from a document "
            "(page {page_numbers}).",
            "",
            "INFORMATION TO VERIFY:",
            "\"{information}\"",
            "",
            "TASK:",
            "Carefully examine the page image(s) and determine whether the "
            "information above is:",
            "",
            "- CORRECT   : The information exactly or substantially matches "
            "what is visible on the page(s).",
            "- INCORRECT : The information is present on the page(s) but "
            "contains errors (wrong values, misspellings, wrong dates, etc.).",
            "- MISSING   : The information is not present on the page(s) at all.",
            "- UNVERIFIABLE : The page(s) are illegible, the information is "
            "too vague to check, or you cannot determine the answer.",
            "",
            "RULES:",
            "1. Read ALL text visible in the image(s) carefully.",
            "2. Compare the information verbatim — check numbers, dates, "
            "names, and amounts precisely.",
            "3. Minor formatting differences (e.g. '01 Jan 2025' vs "
            "'1 January 2025') are still CORRECT if the meaning is the same.",
            "4. If the information is partially correct and partially wrong, "
            "return INCORRECT and explain what is wrong.",
            "5. Be concise in your reasoning (1–3 sentences).",
            "",
            "Return ONLY a JSON object:",
            "{{",
            "  \"verdict\": \"CORRECT | INCORRECT | MISSING | UNVERIFIABLE\",",
            "  \"reasoning\": \"<brief explanation of your decision>\"",
            "}}",
            "",
            "Return the JSON now:",
        ])
