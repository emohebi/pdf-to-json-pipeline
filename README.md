# PDF to JSON Pipeline

A semi-agentic pipeline for extracting structured JSON from PDF documents using AWS Bedrock and Strands SDK.

## Features

- 🤖 **Semi-Agentic Architecture**: Intelligent agents handle section detection and extraction
- 👁️ **Vision-Powered**: Claude models extract text from images within PDFs
- ⚡ **Parallel Processing**: Process multiple sections and documents simultaneously
- ✅ **Human Validation**: Intermediate results stored for review
- 🔄 **Batch & Single Mode**: Switch between processing modes
- 📊 **Progress Tracking**: Checkpointing for large batch jobs
- 🛡️ **Error Recovery**: Robust error handling and retry logic

## Architecture

```
┌─────────────────────────────────────────────────┐
│         PDF Document (20-30 pages)              │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  STAGE 1: Section Detection Agent               │
│  - Convert PDF pages to images                  │
│  - Use Claude Vision to identify sections       │
│  - Save intermediate results                    │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  STAGE 2: Parallel Section Extraction           │
│  - Each section → Extraction Agent              │
│  - Vision LLM extracts text from images         │
│  - Save section JSONs for validation            │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  STAGE 3: Validation & Aggregation Agent        │
│  - Validates each section JSON                  │
│  - Combines into final document JSON            │
│  - Human review if confidence < threshold       │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
                Final JSON
```

## Project Structure

```
pdf-to-json-pipeline/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configuration management
│   └── schemas.py            # Document and section schemas
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── section_detector.py    # Stage 1: Section detection
│   │   ├── section_extractor.py   # Stage 2: Data extraction
│   │   └── validator.py            # Stage 3: Validation
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── bedrock_vision.py      # Bedrock API tools
│   │   └── validation.py          # Validation utilities
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py       # PDF to image conversion
│   │   ├── storage.py             # File I/O operations
│   │   └── logger.py              # Logging configuration
│   └── pipeline.py                 # Main orchestrator
├── scripts/
│   ├── run_single.py              # Process single PDF
│   ├── run_batch.py               # Process batch of PDFs
│   └── validate_outputs.py        # Human validation helper
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_pipeline.py
│   └── test_data/
└── output/
    ├── intermediate/              # For human validation
    │   ├── sections/              # Individual section JSONs
    │   ├── detection/             # Section detection results
    │   └── validation_queue/      # Items needing review
    ├── final/                     # Final document JSONs
    └── logs/                      # Processing logs
```

## Setup

### Prerequisites

- Python 3.9+
- AWS Account with Bedrock access
- AWS credentials configured
- Claude 3.5/4 Sonnet model access enabled in Bedrock

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-to-json-pipeline
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

5. Configure AWS credentials:
```bash
aws configure
# Or set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
```

## Usage

### Single Document Processing

```bash
python scripts/run_single.py --pdf path/to/document.pdf
```

With custom output directory:
```bash
python scripts/run_single.py \
    --pdf path/to/document.pdf \
    --output-dir custom_output/
```

### Batch Processing

```bash
python scripts/run_batch.py --input-dir path/to/pdfs/
```

With custom settings:
```bash
python scripts/run_batch.py \
    --input-dir path/to/pdfs/ \
    --output-dir custom_output/ \
    --workers 10 \
    --batch-size 50
```

### Resume Failed Batch

```bash
python scripts/run_batch.py \
    --input-dir path/to/pdfs/ \
    --resume
```

### Human Validation Workflow

1. Check validation queue:
```bash
python scripts/validate_outputs.py --list
```

2. Review specific document:
```bash
python scripts/validate_outputs.py --review DOC_ID
```

3. Approve or reject:
```bash
python scripts/validate_outputs.py --approve DOC_ID
python scripts/validate_outputs.py --reject DOC_ID --reason "Missing data in section 3"
```

## Configuration

### Customize Schemas

Edit `config/schemas.py` to define your document structure:

```python
SECTION_DEFINITIONS = {
    'header': 'Title page, document information',
    'summary': 'Executive summary',
    'body': 'Main content',
    # Add your sections...
}

SECTION_SCHEMAS = {
    'header': {
        'type': 'object',
        'properties': {
            'title': {'type': 'string'},
            # Define your fields...
        }
    }
}
```

### Adjust Settings

Edit `config/settings.py` for pipeline configuration:

```python
MAX_WORKERS = 5
BATCH_SIZE = 100
CONFIDENCE_THRESHOLD = 0.85
DPI = 150
MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
```

## Cost Estimation

**For 90k documents (25 pages avg):**
- Stage 1 (Section Detection): ~90k API calls
- Stage 2 (Extraction): ~450k API calls (5 sections/doc)
- Stage 3 (Validation): Minimal API usage

**Estimated Total Cost:** $65k-$95k
**Processing Time:** 4-7 days (50 workers)

## Output Structure

### Intermediate Results
```
output/intermediate/
├── detection/
│   └── DOC_ID_sections.json        # Detected sections
├── sections/
│   ├── DOC_ID_section_1.json       # Individual sections
│   └── DOC_ID_section_2.json
└── validation_queue/
    └── DOC_ID_review.json          # Flagged for review
```

### Final Results
```
output/final/
└── DOC_ID.json                     # Complete document JSON
```

## Monitoring

View logs:
```bash
tail -f output/logs/pipeline.log
```

Check progress:
```bash
cat output/logs/progress.json
```

## Troubleshooting

### Common Issues

**Issue:** Bedrock authentication error
```
Solution: Check AWS credentials and region configuration
aws sts get-caller-identity
```

**Issue:** Model access denied
```
Solution: Enable Claude model access in Bedrock console
```

**Issue:** Out of memory
```
Solution: Reduce DPI or MAX_WORKERS in settings.py
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

## License

MIT License

## Support

For issues and questions, please open a GitHub issue.
