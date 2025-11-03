#!/usr/bin/env python3
"""
Script to process a single PDF document.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PDFToJSONPipeline
from src.utils import setup_logger

logger = setup_logger('run_single')

def main():
    parser = argparse.ArgumentParser(
        description='Process a single PDF document to JSON'
    )
    # parser.add_argument(
    #     '--pdf',
    #     type=str,
    #     required=True,
    #     help='Path to PDF file'
    # )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory (optional)'
    )
    
    args = parser.parse_args()
    args.pdf = "./input/0025772.pdf"
    args.pdf = "./output/20251103_104937/final/0025772.json"
    # Validate PDF exists
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    logger.info(f"Starting single document processing: {pdf_path.name}")
    
    try:
        pipeline = PDFToJSONPipeline()
        if ".pdf" in str(pdf_path):
            result = pipeline.process_single_pdf(str(pdf_path))
            
            logger.info("Processing successful")
            logger.info(f"Document ID: {result['document_id']}")
            logger.info(f"Confidence: {result['metadata']['confidence_score']:.2f}")
        else:
            import json
            with open(str(pdf_path), 'r') as file:
                json_ = json.load(file)
            pipeline.review_json(json_['sections'], "doc_id")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
