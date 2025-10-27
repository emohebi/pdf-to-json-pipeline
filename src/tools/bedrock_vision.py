"""
AWS Bedrock tools for vision and text processing - FIXED VERSION.
"""
import boto3
import json
import base64
import time
from typing import List, Dict, Any
from strands import tool
import boto3
from botocore.config import Config as boto_Config
from botocore.exceptions import ClientError

from config.settings import (
    BEDROCK_MODEL_ID, BEDROCK_MODEL_REGION, AWS_REGION,
    MODEL_TEMPERATURE, MAX_RETRIES, RETRY_DELAY, BEDROCK_NAMESPACE, MODEL
)
from src.utils.logger import setup_logger

logger = setup_logger('bedrock_tools')


class BedrockClient:
    """AWS Bedrock client wrapper."""
    
    def __init__(self):
        """Initialize Bedrock runtime client."""
        self.namespace = BEDROCK_NAMESPACE
        self.aws_region = AWS_REGION
        self.client = self._create_client(True, True)
        self.model_id = MODEL
    
    def _create_client(self, consumer_bool: bool = True, runtime_client: bool = True):
        """
        Create and configure Bedrock client with role assumption
        Maintains exact same credential setup as original
        
        Args:
            consumer_bool: Whether to use consumer or developer role
            runtime_client: Whether to create runtime or regular client
        
        Returns:
            Configured boto3 client
        """
        # Assume role using STS
        sts_client = boto3.client("sts")
        
        role_type = "bedrock-consumer" if consumer_bool else "bedrock-developer"
        role_arn = f"arn:aws:iam::533267133246:role/{self.namespace}-{role_type}"
        
        try:
            bedrock_account = sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName=role_type,
            )
        except ClientError as e:
            logger.error(f"Failed to assume role {role_arn}: {e}")
            raise
        
        # Extract credentials
        credentials = bedrock_account["Credentials"]
        
        # Configure retry and timeout settings
        config = boto_Config(
            retries={"total_max_attempts": 20, "mode": "standard"},
            read_timeout=1000
        )
        
        # Determine service and endpoint
        service_name = "bedrock-runtime" if runtime_client else "bedrock"
        endpoint = f"https://{service_name}.ap-southeast-2.amazonaws.com"
        
        # Create client with assumed role credentials
        client = boto3.client(
            service_name=service_name,
            region_name=self.aws_region,
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            config=config,
            endpoint_url=endpoint,
        )
        
        logger.info(f"Created {service_name} client with {role_type} role")
        return client
    
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
                
                # FIX: Properly handle the streaming response body
                response_body = response.get('body')
                
                if response_body is None:
                    raise ValueError("Response body is None")
                
                # Read the entire response body
                try:
                    # Method 1: Direct read
                    body_bytes = response_body.read()
                except Exception as read_error:
                    logger.warning(f"Direct read failed: {read_error}, trying chunked read")
                    # Method 2: Chunked read if direct read fails
                    chunks = []
                    while True:
                        try:
                            chunk = response_body.read(8192)
                            if not chunk:
                                break
                            chunks.append(chunk)
                        except Exception as chunk_error:
                            logger.error(f"Chunk read failed: {chunk_error}")
                            break
                    body_bytes = b''.join(chunks)
                
                if not body_bytes:
                    raise ValueError("Response body is empty")
                
                # Decode bytes to string
                if isinstance(body_bytes, bytes):
                    body_str = body_bytes.decode('utf-8')
                else:
                    body_str = str(body_bytes)
                
                # Parse JSON
                result = json.loads(body_str)
                
                logger.debug(f"Successfully invoked model on attempt {attempt + 1}")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise
                    
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