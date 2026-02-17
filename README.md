# PDF to JSON Pipeline

A configurable pipeline that converts PDF documents into structured JSON using LLM vision models. Supports **AWS Bedrock** and **Azure OpenAI** as LLM providers.

## Architecture

```
INPUT (PDF files)  →  TASK (detect → extract → review → validate)  →  OUTPUT (JSON)
```

All configuration is driven by **`config.json`** with three top-level sections:

| Section   | Purpose |
|-----------|---------|
| `INPUT`   | PDF source directory, DPI, supported extensions |
| `TASK`    | LLM provider, model params, section definitions, schemas, processing settings |
| `OUTPUT`  | Output directory, timestamp dirs, intermediate file saving |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure `config.json`

Edit `config.json` to set your provider and credentials:

**For AWS Bedrock:**
```json
{
  "TASK": {
    "provider": "aws_bedrock",
    "aws_bedrock": {
      "region": "ap-southeast-2",
      "model_id": "au.anthropic.claude-sonnet-4-5-20250929-v1:0",
      "namespace": "your-namespace"
    }
  }
}
```

**For Azure OpenAI:**
```json
{
  "TASK": {
    "provider": "azure_openai",
    "azure_openai": {
      "endpoint_env": "AZURE_OPENAI_ENDPOINT",
      "api_key_env": "AZURE_OPENAI_API_KEY",
      "deployment_name": "gpt-4o",
      "api_version": "2024-12-01-preview"
    }
  }
}
```

Then set environment variables:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"
```

### 3. Run

```bash
# Process all PDFs in the configured INPUT.pdf_directory
python scripts/run_single.py

# Process a specific PDF
python scripts/run_single.py --pdf ./path/to/document.pdf

# Override provider at runtime
python scripts/run_single.py --provider azure_openai

# Enable review stage
python scripts/run_single.py --review
```

## Configuration Reference

### INPUT
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `pdf_directory` | string | `"./input"` | Directory containing PDF files |
| `supported_extensions` | list | `[".pdf"]` | File extensions to process |
| `dpi` | int | `150` | DPI for page rendering |

### TASK
| Key | Type | Description |
|-----|------|-------------|
| `provider` | string | `"aws_bedrock"` or `"azure_openai"` |
| `section_definitions` | object | Map of section_type → description |
| `section_schemas` | object | JSON schemas for each section type |
| `document_header_fields` | list | Header fields to extract from page 1 |
| `model_params.temperature` | float | LLM temperature (0 = deterministic) |
| `processing.parallel` | bool | Enable parallel section extraction |
| `processing.review` | bool | Enable the review agent stage |

### OUTPUT
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `output_directory` | string | `"./output"` | Base output directory |
| `create_timestamp_dir` | bool | `true` | Create timestamped subdirectory |
| `save_intermediates` | bool | `true` | Save intermediate results |

## Customising Section Definitions and Schemas

The pipeline is fully configurable for any document type. Edit `config.json`:

1. **`section_definitions`** — Define what sections your documents contain and their descriptions. The detector uses these to classify document regions.

2. **`section_schemas`** — Define the JSON structure for each section type. The extractor fills in these schemas with content from the PDF.

3. **`document_header_fields`** — Define which metadata fields to extract from the first page.

## Project Structure

```
├── config.json                    # Main configuration
├── config/
│   ├── config_loader.py           # Config.json reader
│   ├── settings.py                # Derived settings from config
│   └── schemas_docuporter.py      # Schema helpers
├── scripts/
│   ├── run_single.py              # Main entry point
│   └── validate_outputs.py        # Output validation tool
├── src/
│   ├── pipeline.py                # Pipeline orchestrator
│   ├── agents/
│   │   ├── section_detector.py    # Stage 1: Detect sections
│   │   ├── section_extractor.py   # Stage 2: Extract content
│   │   ├── review_agent.py        # Stage 3.5: Quality review
│   │   ├── validator_docuporter.py # Stage 4: Validate & combine
│   │   └── document_header_extractor.py
│   ├── tools/
│   │   ├── llm_provider.py        # Abstract LLM interface
│   │   ├── bedrock_vision.py      # AWS Bedrock provider
│   │   ├── azure_vision.py        # Azure OpenAI provider
│   │   └── validation.py          # Data validation tools
│   └── utils/
│       ├── pdf_processor.py       # PDF → images + text
│       ├── image_descriptor.py    # Image position mapping
│       ├── storage.py             # File I/O management
│       ├── docuporter_processor.py # DocuPorter format helpers
│       └── logger.py
└── requirements.txt
```

## Environment Variable Overrides

Any setting from `config.json` can be overridden via environment variables:

| Variable | Overrides |
|----------|-----------|
| `LLM_PROVIDER` | `TASK.provider` |
| `AWS_REGION` | `TASK.aws_bedrock.region` |
| `BEDROCK_MODEL_ID` | `TASK.aws_bedrock.model_id` |
| `AZURE_OPENAI_ENDPOINT` | Azure endpoint |
| `AZURE_OPENAI_API_KEY` | Azure API key |
| `OUTPUT_DIR` | `OUTPUT.output_directory` |
| `DPI` | `INPUT.dpi` |
| `MAX_WORKERS` | `TASK.processing.max_workers` |
| `PIPELINE_CONFIG` | Path to config.json |
