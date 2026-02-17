"""
Abstract LLM provider interface.
Defines a common API for vision, multimodal, and text invocations
across different backends (AWS Bedrock, Azure OpenAI, etc.).
"""
import abc
from typing import List, Optional
from src.utils.logger import setup_logger

logger = setup_logger("llm_provider")


class LLMProvider(abc.ABC):
    """Abstract base for all LLM providers."""

    @abc.abstractmethod
    def invoke_vision(self, image_data: str, prompt: str, max_tokens: int = 8192) -> str:
        """Single image + text prompt -> text response."""

    @abc.abstractmethod
    def invoke_multimodal(self, images: List[str], prompt: str, max_tokens: int = 8192) -> str:
        """Multiple images + text prompt -> text response."""

    @abc.abstractmethod
    def invoke_text(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 4096) -> str:
        """Text-only prompt -> text response."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_provider_instance: Optional[LLMProvider] = None


def get_llm_provider(force_reload: bool = False) -> LLMProvider:
    global _provider_instance
    if _provider_instance is not None and not force_reload:
        return _provider_instance

    from config.settings import PROVIDER_NAME

    if PROVIDER_NAME == "azure_openai":
        from src.tools.azure_vision import AzureOpenAIProvider
        _provider_instance = AzureOpenAIProvider()
    elif PROVIDER_NAME == "aws_bedrock":
        from src.tools.bedrock_vision import BedrockProvider
        _provider_instance = BedrockProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {PROVIDER_NAME}")

    logger.info(f"Initialized LLM provider: {PROVIDER_NAME}")
    return _provider_instance


# ---------------------------------------------------------------------------
# Convenience module-level functions
# ---------------------------------------------------------------------------

def invoke_vision(image_data: str, prompt: str, max_tokens: int = 8192) -> str:
    return get_llm_provider().invoke_vision(image_data, prompt, max_tokens)

def invoke_multimodal(images: List[str], prompt: str, max_tokens: int = 8192) -> str:
    return get_llm_provider().invoke_multimodal(images, prompt, max_tokens)

def invoke_text(prompt: str, system_prompt: str = None, max_tokens: int = 4096) -> str:
    return get_llm_provider().invoke_text(prompt, system_prompt, max_tokens)
