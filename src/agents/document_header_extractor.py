"""
Document Header Extraction - config-driven prompt and fields.
No hardcoded field names or document types.
"""
import json
from typing import Dict, Any

from config.settings import MODEL_MAX_TOKENS_EXTRACTION
from config.config_loader import get_document_header_fields, get_header_prompt, render_prompt
from src.tools.llm_provider import invoke_multimodal
from src.tools.bedrock_vision import prepare_images_for_bedrock
from src.utils import setup_logger

logger = setup_logger("document_header_extractor")


class DocumentHeaderExtractor:
    def __init__(self):
        self.fields = get_document_header_fields()

    def extract_header(self, first_page_data: Dict, document_id: str) -> Dict[str, Any]:
        logger.info(f"[{document_id}] Extracting document header")
        try:
            images = prepare_images_for_bedrock([first_page_data])
            fields_json = json.dumps(
                {f: {"text": "exact text"} for f in self.fields},
                indent=2,
            )
            prompt = render_prompt(get_header_prompt(), fields_json=fields_json)
            response = invoke_multimodal(images=images, prompt=prompt, max_tokens=MODEL_MAX_TOKENS_EXTRACTION)
            return self._parse(response)
        except Exception as e:
            logger.error(f"[{document_id}] Header extraction failed: {e}")
            return self._empty()

    def _parse(self, response: str) -> Dict[str, Any]:
        try:
            resp = response.strip()
            for pfx in ("```json", "```"):
                if resp.startswith(pfx): resp = resp[len(pfx):]
            if resp.endswith("```"): resp = resp[:-3]
            data = json.loads(resp.strip())
            for f in self.fields:
                if f not in data or not data[f]:
                    data[f] = {"text": ""}
                elif not isinstance(data[f], dict):
                    v = str(data[f]); data[f] = {"text": v}
                else:
                    data[f].setdefault("text", data[f].get("text", ""))
            return data
        except Exception as e:
            logger.error(f"Parse failed: {e}")
            return self._empty()

    def _empty(self):
        return {f: {"text": ""} for f in self.fields}
