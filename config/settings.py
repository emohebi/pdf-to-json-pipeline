"""
Configuration settings for PDF to JSON pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', 'output')) / f"{timestamp}"
INTERMEDIATE_DIR = OUTPUT_DIR / 'intermediate'
FINAL_DIR = OUTPUT_DIR / 'final'
LOGS_DIR = OUTPUT_DIR / 'logs'

# Create directories if they don't exist
for directory in [OUTPUT_DIR, INTERMEDIATE_DIR, FINAL_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Subdirectories for intermediate results
DETECTION_DIR = INTERMEDIATE_DIR / 'detection'
SECTIONS_DIR = INTERMEDIATE_DIR / 'sections'
VALIDATION_QUEUE_DIR = INTERMEDIATE_DIR / 'validation_queue'

for directory in [DETECTION_DIR, SECTIONS_DIR, VALIDATION_QUEUE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', "ap-southeast-2")
BEDROCK_MODEL_ID = os.getenv(
    'BEDROCK_MODEL_ID',
    'anthropic.claude-3-5-sonnet-20241022-v2:0'
)
BEDROCK_MODEL_REGION = os.getenv('BEDROCK_MODEL_REGION', 'us-east-1')
BEDROCK_NAMESPACE: str = "swscraping"
MODEL_ID_37: str = "apac.anthropic.claude-3-7-sonnet-20250219-v1:0"
MODEL_ID_40: str = "apac.anthropic.claude-sonnet-4-20250514-v1:0"# "au.anthropic.claude-sonnet-4-5-20250929-v1:0"
MODEL_ID_45: str = "au.anthropic.claude-sonnet-4-5-20250929-v1:0"
MODEL = MODEL_ID_40

# Processing Configuration
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '5'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '100'))
DPI = int(os.getenv('DPI', '150'))

# Confidence thresholds
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.85'))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv('LOW_CONFIDENCE_THRESHOLD', '0.70'))

# API Configuration
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', '5'))

# Debug settings
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
VERBOSE = os.getenv('VERBOSE', 'True').lower() == 'true'

# Model parameters
MODEL_TEMPERATURE = 0.1  # Deterministic for extraction
MODEL_MAX_TOKENS_DETECTION = 4096
MODEL_MAX_TOKENS_EXTRACTION = 56000
MODEL_MAX_TOKENS_VALIDATION = 8192

# Processing modes
PROCESSING_MODE_SINGLE = 'single'
PROCESSING_MODE_BATCH = 'batch'

# Validation states
VALIDATION_STATE_PENDING = 'pending'
VALIDATION_STATE_APPROVED = 'approved'
VALIDATION_STATE_REJECTED = 'rejected'
VALIDATION_STATE_IN_REVIEW = 'in_review'

# File extensions
SUPPORTED_PDF_EXTENSIONS = ['.pdf']
OUTPUT_FILE_EXTENSION = '.json'

# Progress tracking
PROGRESS_FILE = LOGS_DIR / 'progress.json'
ERROR_LOG_FILE = LOGS_DIR / 'errors.log'
PIPELINE_LOG_FILE = LOGS_DIR / 'pipeline.log'
