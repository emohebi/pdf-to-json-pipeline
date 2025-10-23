#!/usr/bin/env python3
"""
Script to process batch of PDF documents.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PDFToJSONPipeline
from src.utils import setup_logger

logger = setup_logger('run_batch')


def main():
    parser = argparse.ArgumentParser(
        description='Process batch of PDF documents to JSON'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing PDF files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory (optional)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of parallel workers (default: 5)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous progress'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.pdf',
        help='File pattern to match (default: *.pdf)'
    )
    
    args = parser.parse_args()
    
    # Find PDF files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    pdf_files = list(input_dir.glob(args.pattern))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Resume: {args.resume}")
    
    # Process batch
    try:
        pipeline = PDFToJSONPipeline(max_workers=args.workers)
        results = pipeline.process_batch(
            [str(p) for p in pdf_files],
            resume=args.resume
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total documents: {results['total']}")
        logger.info(f"Completed: {results['completed_count']}")
        logger.info(f"Failed: {results['failed_count']}")
        
        if results['failed']:
            logger.warning("\nFailed documents:")
            for fail in results['failed']:
                logger.warning(f"  - {fail['document_id']}: {fail['error']}")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
