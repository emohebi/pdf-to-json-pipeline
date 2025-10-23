# System Architecture

## Overview

This is a **Semi-Agentic Vision Pipeline** that processes PDF documents into structured JSON using AWS Bedrock (Claude Vision) and Strands SDK.

## Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    PDF Document                          │
│                   (20-30 pages)                          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ PDFProcessor
                       │ (PyMuPDF: PDF → Images)
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 Pages as Images                          │
│              (High-res PNG, DPI=150)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
╔══════════════════════════════════════════════════════════╗
║              STAGE 1: Section Detection                  ║
║                                                           ║
║  ┌──────────────────────────────────────────────────┐   ║
║  │  SectionDetectionAgent (Strands Agent)           │   ║
║  │  - Samples pages (every 5th + first/last)        │   ║
║  │  - Claude Vision analyzes structure              │   ║
║  │  - Returns section boundaries                    │   ║
║  └──────────────────────────────────────────────────┘   ║
║                                                           ║
║  Tools: invoke_bedrock_multimodal                        ║
║  Model: Claude 4 Sonnet (Bedrock)                        ║
╚══════════════════════┬═══════════════════════════════════╝
                       │
                       │ SAVES TO:
                       │ output/intermediate/detection/
                       │ doc_id_sections.json
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Detected Sections                           │
│  [                                                       │
│    {section_type, start_page, end_page, name}           │
│  ]                                                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
╔══════════════════════════════════════════════════════════╗
║          STAGE 2: Parallel Section Extraction            ║
║                                                           ║
║  ┌────────────┐  ┌────────────┐  ┌────────────┐        ║
║  │ Section 1  │  │ Section 2  │  │ Section N  │        ║
║  │ Extractor  │  │ Extractor  │  │ Extractor  │        ║
║  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘        ║
║        │                │                │               ║
║        │    SectionExtractionAgent       │               ║
║        │    (One per section)            │               ║
║        │                                 │               ║
║        │    Tools: invoke_bedrock_multimodal            ║
║        │    Model: Claude 4 Sonnet                      ║
║        │    + calculate_confidence_score                ║
║        │                                 │               ║
║        ▼                ▼                ▼               ║
║   Section JSON    Section JSON    Section JSON          ║
║   (confidence)    (confidence)    (confidence)           ║
╚═══════════════════════┬═════════════════════════════════╝
                        │
                        │ SAVES TO:
                        │ output/intermediate/sections/
                        │ doc_id_section_N.json
                        ▼
┌─────────────────────────────────────────────────────────┐
│              All Section JSONs                           │
│  [ {metadata, data, confidence}, ... ]                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
╔══════════════════════════════════════════════════════════╗
║        STAGE 3: Validation & Aggregation                 ║
║                                                           ║
║  ┌──────────────────────────────────────────────────┐   ║
║  │  ValidationAgent                                 │   ║
║  │  - Calculate overall confidence                  │   ║
║  │  - Validate document structure                   │   ║
║  │  - Check quality issues                          │   ║
║  └──────────────────────────────────────────────────┘   ║
║                                                           ║
║  Tools: validate_document_structure                      ║
╚══════════════════════┬═══════════════════════════════════╝
                       │
                       │ Decision Point
                       ▼
              ┌────────────────┐
              │ Confidence OK? │
              └────┬──────┬────┘
                   │      │
          Yes ─────┘      └───── No
           │                      │
           ▼                      ▼
  ┌────────────────┐    ┌──────────────────┐
  │  SAVE TO       │    │  QUEUE FOR       │
  │  final/        │    │  validation/     │
  │                │    │                  │
  │  doc_id.json   │    │  doc_id_review   │
  │                │    │     .json        │
  │  ✓ APPROVED    │    │                  │
  └────────────────┘    │  ⚠ NEEDS REVIEW  │
                        └──────────────────┘
                                │
                                │ Human Reviewer
                                ▼
                        ┌──────────────┐
                        │ validate_    │
                        │  outputs.py  │
                        └──────┬───────┘
                               │
                      ┌────────┴────────┐
                      │                 │
                  Approve           Reject
                      │                 │
                      ▼                 ▼
              ┌──────────────┐  ┌──────────────┐
              │ Move to      │  │ Mark as      │
              │ final/       │  │ rejected     │
              └──────────────┘  └──────────────┘
```

## Component Responsibilities

### 1. PDFProcessor (`src/utils/pdf_processor.py`)
- Converts PDF pages to high-resolution images
- Extracts embedded images
- Preserves text as fallback

### 2. SectionDetectionAgent (`src/agents/section_detector.py`)
- **Input**: PDF page images
- **Process**: 
  - Samples representative pages
  - Uses Claude Vision to identify document structure
  - Determines section boundaries and types
- **Output**: List of sections with page ranges
- **Saves**: `output/intermediate/detection/`

### 3. SectionExtractionAgent (`src/agents/section_extractor.py`)
- **Input**: Section pages + schema for section type
- **Process**:
  - Extracts all text (including from images)
  - Structures data according to JSON schema
  - Calculates confidence score
- **Output**: Structured section JSON
- **Saves**: `output/intermediate/sections/`
- **Runs**: In parallel (configurable workers)

### 4. ValidationAgent (`src/agents/validator.py`)
- **Input**: All section JSONs
- **Process**:
  - Combines sections into document
  - Validates structure and quality
  - Calculates overall confidence
  - Routes based on confidence threshold
- **Output**: Complete document JSON
- **Saves**: `output/final/` OR `output/intermediate/validation_queue/`

### 5. StorageManager (`src/utils/storage.py`)
- Saves all intermediate results
- Manages validation queue
- Tracks progress for batch processing
- Handles approval/rejection workflow

## Data Flow

### Input
```
document.pdf (25 pages)
```

### Stage 1 Output
```json
{
  "document_id": "document",
  "sections": [
    {
      "section_type": "header",
      "section_name": "Title Page",
      "start_page": 1,
      "end_page": 1
    },
    {
      "section_type": "body",
      "section_name": "Main Content",
      "start_page": 2,
      "end_page": 20
    }
  ]
}
```

### Stage 2 Output (per section)
```json
{
  "title": "Document Title",
  "author": "John Doe",
  "_metadata": {
    "section_type": "header",
    "page_range": [1, 1],
    "confidence": 0.95
  }
}
```

### Stage 3 Output (final)
```json
{
  "document_id": "document",
  "metadata": {
    "total_pages": 25,
    "confidence_score": 0.92,
    "processing_timestamp": "2025-10-23T12:00:00"
  },
  "sections": [...],
  "validation_status": "pending"
}
```

## Parallelization Strategy

```
Document 1 ──┐
Document 2 ──┼─→ Process Queue (max_workers=5)
Document 3 ──┘

Within each document:
  Section 1 ──┐
  Section 2 ──┼─→ Parallel Extraction
  Section N ──┘
```

## Error Handling

1. **Retry Logic**: Bedrock API calls retry 3x with exponential backoff
2. **Fallback Detection**: If AI detection fails, use rule-based splitting
3. **Progress Tracking**: Resume from checkpoint on failure
4. **Validation Queue**: Low-confidence results routed for review

## Configuration Points

### `config/settings.py`
- `MAX_WORKERS`: Parallel processing workers
- `DPI`: Image resolution
- `CONFIDENCE_THRESHOLD`: Auto-approval threshold
- `MODEL_ID`: Bedrock model to use

### `config/schemas.py`
- `SECTION_DEFINITIONS`: Section types
- `SECTION_SCHEMAS`: JSON schemas for each type
- Customize for your document structure

## Performance Characteristics

### Single Document (25 pages)
- **Time**: 30-60 seconds
- **API Calls**: 6-10 (1 detection + 5 sections)
- **Cost**: ~$0.50-1.00

### Batch (90k documents)
- **Time**: 4-7 days (50 workers)
- **API Calls**: ~540,000 total
- **Cost**: $65k-$95k
- **Throughput**: ~300-500 docs/hour

## Quality Assurance

1. **Confidence Scoring**: Each section scored 0.0-1.0
2. **Schema Validation**: Pydantic enforces structure
3. **Quality Checks**: Detects empty fields, suspicious patterns
4. **Human Review**: Low-confidence results queued
5. **Approval Workflow**: Review → Approve/Reject

## Extensibility

### Adding New Section Types
1. Add to `SECTION_DEFINITIONS` in `schemas.py`
2. Define schema in `SECTION_SCHEMAS`
3. No code changes needed!

### Custom Validation Rules
1. Extend `ValidationAgent.validate_and_combine()`
2. Add custom quality checks to `tools/validation.py`

### Alternative Models
1. Update `BEDROCK_MODEL_ID` in settings
2. Supports any Bedrock model with vision

This architecture balances automation with human oversight, ensuring high-quality structured data extraction at scale.
