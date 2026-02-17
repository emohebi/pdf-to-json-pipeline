"""
AWS Bedrock provider implementation.
Implements LLMProvider for vision, multimodal, and text via AWS Bedrock.
"""
import boto3
import json
import base64
import time
from typing import List, Dict, Any, Optional
from botocore.config import Config as boto_Config
from botocore.exceptions import ClientError

try:
    from strands import tool
except ImportError:
    def tool(fn):
        return fn

from src.tools.llm_provider import LLMProvider
from src.utils.logger import setup_logger

logger = setup_logger("bedrock_tools")


class BedrockClient:
    """AWS Bedrock runtime client wrapper. Config-driven, no hardcoded values."""

    def __init__(self):
        from config.settings import (
            AWS_REGION, BEDROCK_NAMESPACE, BEDROCK_MODEL_ID,
            BEDROCK_ROLE_ARN_TEMPLATE, BEDROCK_READ_TIMEOUT,
            MAX_RETRIES, RETRY_DELAY,
        )
        self.namespace = BEDROCK_NAMESPACE
        self.aws_region = AWS_REGION
        self.model_id = BEDROCK_MODEL_ID
        self.role_arn_template = BEDROCK_ROLE_ARN_TEMPLATE
        self.read_timeout = BEDROCK_READ_TIMEOUT
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY
        self.client = self._create_client()

    def _create_client(self, consumer_bool: bool = True, runtime_client: bool = True):
        sts_client = boto3.client("sts")
        role_arn = self.role_arn_template.format(namespace=self.namespace)

        role_type = "bedrock-consumer" if consumer_bool else "bedrock-developer"
        try:
            assumed = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=role_type)
        except ClientError as e:
            logger.error(f"Failed to assume role {role_arn}: {e}")
            raise

        creds = assumed["Credentials"]
        config = boto_Config(
            retries={"total_max_attempts": 20, "mode": "standard"},
            read_timeout=self.read_timeout,
        )
        service_name = "bedrock-runtime" if runtime_client else "bedrock"
        endpoint = f"https://{service_name}.{self.aws_region}.amazonaws.com"

        client = boto3.client(
            service_name=service_name, region_name=self.aws_region,
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            config=config, endpoint_url=endpoint,
        )
        logger.info(f"Created {service_name} client with {role_type} role")
        return client

    def invoke_with_retry(self, request_body: Dict, max_retries: int = None) -> Dict:
        max_retries = max_retries or self.max_retries
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.invoke_model(modelId=self.model_id, body=json.dumps(request_body))
                body = response.get("body")
                if body is None:
                    raise ValueError("Response body is None")
                try:
                    body_bytes = body.read()
                except Exception:
                    chunks = []
                    while True:
                        try:
                            chunk = body.read(8192)
                            if not chunk: break
                            chunks.append(chunk)
                        except Exception: break
                    body_bytes = b"".join(chunks)
                if not body_bytes:
                    raise ValueError("Response body is empty")
                body_str = body_bytes.decode("utf-8") if isinstance(body_bytes, bytes) else str(body_bytes)
                return json.loads(body_str)
            except json.JSONDecodeError as e:
                last_error = e
                logger.error(f"JSON parsing error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else: raise
            except Exception as e:
                last_error = e
                logger.warning(f"Bedrock attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("All Bedrock retry attempts failed")
                    raise


class BedrockProvider(LLMProvider):
    """LLM provider backed by AWS Bedrock."""

    def __init__(self):
        from config.settings import MODEL_TEMPERATURE
        self._client = BedrockClient()
        self._temperature = MODEL_TEMPERATURE

    def invoke_vision(self, image_data: str, prompt: str, max_tokens: int = 8192) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
                {"type": "text", "text": prompt},
            ]}],
        }
        return self._client.invoke_with_retry(body)["content"][0]["text"]

    def invoke_multimodal(self, images: List[str], prompt: str, max_tokens: int = 8192) -> str:
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}} for img in images]
        content.append({"type": "text", "text": prompt})
        body = {
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": content}],
        }
        return self._client.invoke_with_retry(body)["content"][0]["text"]

    def invoke_text(self, prompt: str, system_prompt: str = None, max_tokens: int = 4096) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt
        return self._client.invoke_with_retry(body)["content"][0]["text"]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def prepare_images_for_bedrock(pages_data: List[Dict]) -> List[str]:
    return [encode_image(page["image"]) for page in pages_data]
