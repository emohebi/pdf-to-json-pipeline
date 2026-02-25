"""
Configuration settings derived from config.json with env-var overrides.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

from config.config_loader import (
    load_config, get_input_config, get_task_config, get_output_config,
    get_provider_name, get_model_params, get_processing_config, get_confidence_config,
    is_term_matching_enabled, is_effective_date_enabled,
)

try:
    load_config()
except FileNotFoundError:
    pass

_input = get_input_config()
_task = get_task_config()
_output = get_output_config()
_mp = get_model_params()
_proc = get_processing_config()
_conf = get_confidence_config()

BASE_DIR = Path(__file__).resolve().parent.parent
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_out_base = Path(os.getenv("OUTPUT_DIR", _output.get("output_directory", "output")))
OUTPUT_DIR = _out_base / timestamp if _output.get("create_timestamp_dir", True) else _out_base
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
FINAL_DIR = OUTPUT_DIR / "final"
LOGS_DIR = OUTPUT_DIR / "logs"
for d in (OUTPUT_DIR, INTERMEDIATE_DIR, FINAL_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DETECTION_DIR = INTERMEDIATE_DIR / "detection"
SECTIONS_DIR = INTERMEDIATE_DIR / "sections"
VALIDATION_QUEUE_DIR = INTERMEDIATE_DIR / "validation_queue"
IMG_DESC_DIR = INTERMEDIATE_DIR / "image_description"
TERM_MATCHING_DIR = INTERMEDIATE_DIR / "term_matching"
EFFECTIVE_DATE_DIR = INTERMEDIATE_DIR / "effective_date"
for d in (DETECTION_DIR, SECTIONS_DIR, VALIDATION_QUEUE_DIR, IMG_DESC_DIR, TERM_MATCHING_DIR, EFFECTIVE_DATE_DIR):
    d.mkdir(parents=True, exist_ok=True)

PDF_INPUT_DIR = Path(_input.get("pdf_directory", "./input"))
DPI = int(os.getenv("DPI", _input.get("dpi", 150)))
SUPPORTED_PDF_EXTENSIONS = _input.get("supported_extensions", [".pdf"])

PROVIDER_NAME = os.getenv("LLM_PROVIDER", get_provider_name())

_bedrock = _task.get("aws_bedrock", {})
AWS_REGION = os.getenv("AWS_REGION", _bedrock.get("region", "ap-southeast-2"))
BEDROCK_MODEL_REGION = os.getenv("BEDROCK_MODEL_REGION", _bedrock.get("model_region", "us-east-1"))
BEDROCK_NAMESPACE = _bedrock.get("namespace", "")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", _bedrock.get("model_id", ""))
BEDROCK_ROLE_ARN_TEMPLATE = _bedrock.get("role_arn_template", "")
BEDROCK_READ_TIMEOUT = _bedrock.get("read_timeout", 3600)
MODEL = BEDROCK_MODEL_ID

_azure = _task.get("azure_openai", {})
AZURE_OPENAI_ENDPOINT = os.getenv(_azure.get("endpoint_env", "AZURE_OPENAI_ENDPOINT"), _azure.get("endpoint", ""))
AZURE_OPENAI_API_KEY = os.getenv(_azure.get("api_key_env", "AZURE_OPENAI_API_KEY"), "")
AZURE_OPENAI_DEPLOYMENT = _azure.get("deployment_name", "gpt-5.1")
AZURE_OPENAI_API_VERSION = _azure.get("api_version", "2025-11-13")
AZURE_OPENAI_TIMEOUT = _azure.get("timeout", 600)

MAX_WORKERS = int(os.getenv("MAX_WORKERS", _proc.get("max_workers", 5)))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", _proc.get("batch_size", 100)))
PARALLEL = _proc.get("parallel", False)
REVIEW_ENABLED = _proc.get("review", False)
MAX_IMAGES_PER_BATCH = _proc.get("max_images_per_batch", 20)

# Term matching (optional step)
TERM_MATCHING_ENABLED = is_term_matching_enabled()

# Effective date extraction (optional step)
EFFECTIVE_DATE_ENABLED = is_effective_date_enabled()

MODEL_TEMPERATURE = _mp.get("temperature", 0)
MODEL_MAX_TOKENS_DETECTION = _mp.get("max_tokens_detection", 4096)
MODEL_MAX_TOKENS_EXTRACTION = _mp.get("max_tokens_extraction", 56000)
MODEL_MAX_TOKENS_VALIDATION = _mp.get("max_tokens_validation", 8192)

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", _conf.get("threshold", 0.85)))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", _conf.get("low_threshold", 0.70)))

MAX_RETRIES = int(os.getenv("MAX_RETRIES", _bedrock.get("max_retries", _azure.get("max_retries", 3))))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", _bedrock.get("retry_delay", _azure.get("retry_delay", 5))))

DEBUG = os.getenv("DEBUG", "False").lower() == "true"
VERBOSE = os.getenv("VERBOSE", "True").lower() == "true"

OUTPUT_FILE_EXTENSION = _output.get("file_extension", ".json")
SAVE_INTERMEDIATES = _output.get("save_intermediates", True)
SAVE_REVIEW_RESULTS = _output.get("save_review_results", True)

VALIDATION_STATE_PENDING = "pending"
PROGRESS_FILE = LOGS_DIR / "progress.json"
ERROR_LOG_FILE = LOGS_DIR / "errors.log"
PIPELINE_LOG_FILE = LOGS_DIR / "pipeline.log"
