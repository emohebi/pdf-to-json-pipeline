"""Azure OpenAI provider implementation."""
import time
from typing import List, Optional
from src.tools.llm_provider import LLMProvider
from src.utils.logger import setup_logger
import logging

logger = setup_logger("azure_vision")


class AzureOpenAIProvider(LLMProvider):
    """LLM provider backed by Azure OpenAI Service."""

    def __init__(self):
        from config.settings import (
            AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
            AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
            AZURE_OPENAI_TIMEOUT, MODEL_TEMPERATURE, MAX_RETRIES, RETRY_DELAY,
        )
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            raise ValueError("Azure OpenAI endpoint and API key must be set.")
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("The 'openai' package is required. Install: pip install openai")
        
        # Initialize client
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        import httpx
        logging.getLogger("httpx").setLevel(logging.CRITICAL)
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        http_client = httpx.Client(verify=False)
        try:
            # self.client = AzureOpenAI(
            #     azure_ad_token_provider=token_provider,
            #     api_version=AZURE_OPENAI_API_VERSION,
            #     azure_endpoint=AZURE_OPENAI_ENDPOINT,
            #     http_client=http_client
            # )
            self.client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION, timeout=AZURE_OPENAI_TIMEOUT,
                http_client=http_client
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Azure OpenAI client: {e}")

        
        self.deployment = AZURE_OPENAI_DEPLOYMENT
        self.temperature = MODEL_TEMPERATURE
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY
        logger.info(f"Azure OpenAI initialised: deployment={self.deployment}")

    def invoke_vision(self, image_data: str, prompt: str, max_tokens: int = 8192) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}", "detail": "high"}},
            {"type": "text", "text": prompt},
        ]}]
        return self._call_with_retry(messages, max_tokens)

    def invoke_multimodal(self, images: List[str], prompt: str, max_tokens: int = 8192) -> str:
        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}", "detail": "high"}} for img in images]
        content.append({"type": "text", "text": prompt})
        return self._call_with_retry([{"role": "user", "content": content}], max_tokens)

    def invoke_text(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 4096) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._call_with_retry(messages, max_tokens)

    def _call_with_retry(self, messages: list, max_tokens: int) -> str:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment, messages=messages,
                    max_completion_tokens=max_tokens, temperature=self.temperature,
                )
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "content_filter":
                    raise ValueError(f"Azure content filter blocked the response")
                text = response.choices[0].message.content
                if text is None:
                    raise ValueError("Azure OpenAI returned empty content")
                return text
            except Exception as e:
                last_error = e
                logger.warning(f"Azure attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise last_error
