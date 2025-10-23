"""
AWS Bedrock tools for vision and text processing.
"""
import boto3
import json
import base64
import time
from typing import List, Dict, Any
from strands import tool

from config.settings import (
    BEDROCK_MODEL_ID, BEDROCK_MODEL_REGION,
    MODEL_TEMPERATURE, MAX_RETRIES, RETRY_DELAY
)
from src.utils.logger import setup_logger

logger = setup_logger('bedrock_tools')


class BedrockClient:
    """AWS Bedrock client wrapper."""
    
    def __init__(self):
        """Initialize Bedrock runtime client."""
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=BEDROCK_MODEL_REGION
        )
        self.model_id = BEDROCK_MODEL_ID
    
    def invoke_with_retry(
        self,
        request_body: Dict,
        max_retries: int = MAX_RETRIES
    ) -> Dict:
        """
        Invoke Bedrock model with retry logic.
        
        Args:
            request_body: Request body for Bedrock
            max_retries: Maximum number of retries
        
        Returns:
            Model response
        """
        for attempt in range(max_retries):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )
                
                result = json.loads(response['body'].read())
                return result
                
            except Exception as e:
                logger.warning(
                    f"Bedrock invocation attempt {attempt + 1} failed: {e}"
                )
                
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"All retry attempts failed")
                    raise


# Initialize global client
bedrock_client = BedrockClient()


@tool
def invoke_bedrock_vision(
    image_data: str,
    prompt: str,
    max_tokens: int = 8192
) -> str:
    """
    Invoke Claude on Bedrock with vision capability.
    
    Args:
        image_data: Base64 encoded image
        prompt: Text prompt for the model
        max_tokens: Maximum tokens in response
    
    Returns:
        Model's text response
    """
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": MODEL_TEMPERATURE,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    try:
        result = bedrock_client.invoke_with_retry(request_body)
        return result['content'][0]['text']
    
    except Exception as e:
        logger.error(f"Bedrock vision invocation error: {e}")
        raise


@tool
def invoke_bedrock_multimodal(
    images: List[str],
    prompt: str,
    max_tokens: int = 8192
) -> str:
    """
    Invoke Claude on Bedrock with multiple images.
    
    Args:
        images: List of base64 encoded images
        prompt: Text prompt for the model
        max_tokens: Maximum tokens in response
    
    Returns:
        Model's text response
    """
    # Build content array with all images
    content = []
    
    for img_data in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_data
            }
        })
    
    content.append({
        "type": "text",
        "text": prompt
    })
    
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": MODEL_TEMPERATURE,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }
    
    try:
        result = bedrock_client.invoke_with_retry(request_body)
        return result['content'][0]['text']
    
    except Exception as e:
        logger.error(f"Bedrock multimodal invocation error: {e}")
        raise


@tool
def invoke_bedrock_text(
    prompt: str,
    system_prompt: str = None,
    max_tokens: int = 4096
) -> str:
    """
    Invoke Claude on Bedrock with text-only input.
    
    Args:
        prompt: Text prompt
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens in response
    
    Returns:
        Model's text response
    """
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": MODEL_TEMPERATURE,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    if system_prompt:
        request_body["system"] = system_prompt
    
    try:
        result = bedrock_client.invoke_with_retry(request_body)
        return result['content'][0]['text']
    
    except Exception as e:
        logger.error(f"Bedrock text invocation error: {e}")
        raise


def encode_image(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64 string.
    
    Args:
        image_bytes: Image bytes
    
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_bytes).decode('utf-8')


def prepare_images_for_bedrock(pages_data: List[Dict]) -> List[str]:
    """
    Prepare page images for Bedrock invocation.
    
    Args:
        pages_data: List of page data dicts with 'image' key
    
    Returns:
        List of base64 encoded images
    """
    return [encode_image(page['image']) for page in pages_data]
