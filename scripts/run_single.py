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
    args.pdf = "./input/0184410.pdf"
    args.json_dir = "./output/20251126_124423/intermediate/sections"
    args.review_only = False
    # Validate PDF exists
    pdf_path = Path(args.pdf)
    json_dir = Path(args.json_dir)
    
    try:
        pipeline = PDFToJSONPipeline()
        if not args.review_only:
            logger.info(f"Starting single document processing: {pdf_path.name}")
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                sys.exit(1)
            pipeline.process_single_pdf(str(pdf_path), review=False)
        else: # review only
            import json
            from tqdm import tqdm
            logger.info(f"Starting reviewing documents: {pdf_path.name} vs {json_dir.name}")
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                sys.exit(1)
            if not json_dir.exists():
                logger.error(f"JSON file not found: {json_dir}")
                sys.exit(1)
            json_files = json_dir.glob("*.json")
            sections = []
            for json_file in tqdm(json_files, desc="Loading json files:"):
                with open(str(json_file), 'r') as file:
                    json_ = json.load(file)
                sections.append(json_['data'])
            logger.info("STAGE 1: Extracting PDF pages...")
            pages_data = pipeline.pdf_processor.pdf_to_images(str(pdf_path), extract_with_bedrock=False)
            logger.info(f"Extracted {len(pages_data)} pages")
            pipeline.call_review_agent(sections, pdf_path.stem, pages_data=pages_data)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
