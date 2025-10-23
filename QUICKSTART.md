# Quick Start Guide

## Setup (5 minutes)

### 1. Install Dependencies
```bash
cd pdf-to-json-pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure AWS Credentials
```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
```

### 3. Enable Bedrock Model Access
- Go to AWS Console → Bedrock → Model access
- Enable "Claude 3.5 Sonnet" and "Claude 4 Sonnet"

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings (optional)
```

## Usage

### Process Single PDF
```bash
python scripts/run_single.py --pdf path/to/document.pdf
```

Output will be in:
- `output/intermediate/` - Section detection and individual sections  
- `output/final/` - Complete document JSON (if passed validation)
- `output/intermediate/validation_queue/` - Documents needing human review

### Process Batch of PDFs
```bash
python scripts/run_batch.py --input-dir path/to/pdfs/
```

With options:
```bash
python scripts/run_batch.py \
    --input-dir path/to/pdfs/ \
    --workers 10 \
    --resume  # Resume from previous run
```

### Review Validation Queue
```bash
# List documents needing review
python scripts/validate_outputs.py --list

# Review specific document
python scripts/validate_outputs.py --review DOCUMENT_ID

# Approve document
python scripts/validate_outputs.py --approve DOCUMENT_ID --reviewer "Your Name"

# Reject document
python scripts/validate_outputs.py --reject DOCUMENT_ID --reason "Missing data" --reviewer "Your Name"
```

## Understanding Output Structure

```
output/
├── intermediate/
│   ├── detection/          # Section detection results
│   │   └── doc_id_sections.json
│   ├── sections/           # Individual section JSONs
│   │   ├── doc_id_section_1.json
│   │   └── doc_id_section_2.json
│   └── validation_queue/   # Docs needing human review
│       └── doc_id_review.json
├── final/                  # Approved final JSONs
│   └── doc_id.json
└── logs/                   # Processing logs
    ├── pipeline.log
    └── progress.json
```

## Customizing for Your Documents

### 1. Edit Section Definitions
Edit `config/schemas.py`:
```python
SECTION_DEFINITIONS = {
    'your_section_type': 'Description of this section',
    # Add your sections...
}
```

### 2. Define Section Schemas
```python
SECTION_SCHEMAS = {
    'your_section_type': {
        'type': 'object',
        'properties': {
            'your_field': {'type': 'string'},
            # Define your fields...
        },
        'required': ['your_field']
    }
}
```

### 3. Adjust Settings
Edit `config/settings.py` or `.env`:
- `MAX_WORKERS` - Parallel processing workers
- `DPI` - Image resolution (higher = better quality, slower)
- `CONFIDENCE_THRESHOLD` - Minimum confidence for auto-approval

## Troubleshooting

### Issue: "Model access denied"
**Solution**: Enable Claude models in AWS Bedrock console

### Issue: "Out of memory"
**Solution**: Reduce `DPI` or `MAX_WORKERS` in settings

### Issue: "Low confidence scores"
**Solution**: 
- Check if PDFs are high quality
- Increase DPI for better image extraction
- Review section schemas - may be too strict

## Cost Optimization

For 1,000 documents:
- **Standard**: ~$700-1000 (all vision API calls)
- **Optimized**: ~$300-500 (sample pages for detection)

To reduce costs:
1. Use smaller sample sizes for section detection
2. Use fallback detection for simple documents
3. Batch process during off-peak hours

## Next Steps

1. Test with 5-10 sample PDFs
2. Review intermediate outputs in `output/intermediate/`
3. Adjust schemas based on your document structure
4. Fine-tune confidence thresholds
5. Scale to full batch processing

## Support

Check logs:
```bash
tail -f output/logs/pipeline.log
```

View progress:
```bash
cat output/logs/progress.json
```
