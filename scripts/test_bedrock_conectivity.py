#!/usr/bin/env python3
"""
Diagnostic script to test Bedrock connectivity and identify issues.
"""
import sys
import json
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.tools.bedrock_vision import bedrock_client, invoke_bedrock_text
    from src.utils import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root")
    sys.exit(1)

logger = setup_logger('bedrock_diagnostic')

def test_basic_connectivity():
    """Test basic Bedrock connectivity."""
    print("\n" + "="*60)
    print("TEST 1: Basic Connectivity")
    print("="*60)
    
    try:
        # Simple text-only request
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": "Say 'Connection successful' and nothing else."
                }
            ]
        }
        
        response = bedrock_client.client.invoke_model(
            modelId=bedrock_client.model_id,
            body=json.dumps(request_body)
        )
        
        print(f"✓ Model invoked successfully")
        print(f"  Response status: {response['ResponseMetadata']['HTTPStatusCode']}")
        
        # Check response headers
        headers = response['ResponseMetadata'].get('HTTPHeaders', {})
        content_length = headers.get('content-length', 'Unknown')
        print(f"  Content-Length: {content_length}")
        
        return response
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_response_reading_methods(response):
    """Test different methods of reading the response."""
    if not response:
        print("\n⚠ Skipping response reading tests (no response available)")
        return
    
    print("\n" + "="*60)
    print("TEST 2: Response Reading Methods")
    print("="*60)
    
    # Method 1: Direct read
    print("\nMethod 1: Direct read()")
    try:
        body = response['body']
        data = body.read()
        print(f"✓ Direct read successful: {len(data)} bytes")
        print(f"  Content preview: {data[:100]}...")
        
        # Try to parse
        result = json.loads(data.decode('utf-8'))
        print(f"✓ JSON parsing successful")
        print(f"  Response: {result['content'][0]['text']}")
        return True
        
    except Exception as e:
        print(f"✗ Direct read failed: {e}")
    
    # Method 2: Chunked read (need fresh response)
    print("\n⚠ Note: Chunked read test requires a fresh response")
    print("  (Stream already consumed by previous test)")
    
    return False


def test_with_retry_logic():
    """Test the actual function with retry logic."""
    print("\n" + "="*60)
    print("TEST 3: Using invoke_bedrock_text function")
    print("="*60)
    
    try:
        start_time = time.time()
        response = invoke_bedrock_text(
            prompt="Count from 1 to 5, one number per line.",
            max_tokens=100
        )
        duration = time.time() - start_time
        
        print(f"✓ Function call successful ({duration:.2f}s)")
        print(f"  Response: {response}")
        return True
        
    except Exception as e:
        print(f"✗ Function call failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_timeout_limits():
    """Test with different timeout configurations."""
    print("\n" + "="*60)
    print("TEST 4: Timeout Analysis")
    print("="*60)
    
    from botocore.config import Config as boto_Config
    import boto3
    
    configs = [
        ("Default (1000s)", boto_Config(read_timeout=1000)),
        ("Extended (2000s)", boto_Config(read_timeout=2000)),
        ("Conservative (500s)", boto_Config(read_timeout=500)),
    ]
    
    for name, config in configs:
        print(f"\nTesting {name}:")
        try:
            # This is a simplified test - in reality you'd need to create a new client
            print(f"  Config: read_timeout={config.read_timeout}")
            print(f"  ℹ Note: Actual test would require client recreation")
        except Exception as e:
            print(f"  Error: {e}")


def check_environment():
    """Check environment configuration."""
    print("\n" + "="*60)
    print("Environment Check")
    print("="*60)
    
    from config.settings import (
        AWS_REGION, BEDROCK_MODEL_ID, MODEL, 
        MAX_RETRIES, RETRY_DELAY
    )
    
    print(f"AWS Region: {AWS_REGION}")
    print(f"Model ID: {MODEL}")
    print(f"Max Retries: {MAX_RETRIES}")
    print(f"Retry Delay: {RETRY_DELAY}s")
    print(f"Bedrock Client Model: {bedrock_client.model_id}")


def main():
    """Run all diagnostic tests."""
    print("\n" + "="*60)
    print("BEDROCK DIAGNOSTIC TEST SUITE")
    print("="*60)
    
    check_environment()
    
    # Test 1: Basic connectivity
    response = test_basic_connectivity()
    
    # Test 2: Response reading
    test_response_reading_methods(response)
    
    # Test 3: Actual function
    test_with_retry_logic()
    
    # Test 4: Timeout analysis
    test_timeout_limits()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. If all tests pass, the issue may be intermittent")
    print("2. If Test 1 fails, check AWS credentials and permissions")
    print("3. If Test 2 fails, apply the enhanced reading fix")
    print("4. If Test 3 fails, check the retry logic implementation")
    print("\nSee BEDROCK_FIX_GUIDE.md for detailed solutions")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)