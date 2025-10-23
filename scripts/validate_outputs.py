#!/usr/bin/env python3
"""
Script to review and validate outputs queued for human review.
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import StorageManager, setup_logger

logger = setup_logger('validate_outputs')
storage = StorageManager()


def list_validation_queue():
    """List all documents in validation queue."""
    queue = storage.get_validation_queue()
    
    if not queue:
        logger.info("Validation queue is empty!")
        return
    
    logger.info(f"\n{len(queue)} document(s) in validation queue:\n")
    logger.info(f"{'#':<4} {'Document ID':<30} {'Reason':<50}")
    logger.info("-" * 84)
    
    for i, item in enumerate(queue, 1):
        logger.info(
            f"{i:<4} {item['document_id']:<30} {item['reason']:<50}"
        )


def review_document(document_id: str):
    """Show document for review."""
    file_path = Path(f"output/intermediate/validation_queue/{document_id}_review.json")
    
    if not file_path.exists():
        logger.error(f"Document not found: {document_id}")
        return
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"\nDocument ID: {data['document_id']}")
    logger.info(f"Status: {data['status']}")
    logger.info(f"Reason: {data['reason']}")
    logger.info(f"Queued: {data['timestamp']}")
    logger.info(f"\nDocument Structure:")
    logger.info(f"  Sections: {len(data['document']['sections'])}")
    logger.info(f"  Confidence: {data['document']['metadata']['confidence_score']:.2f}")
    
    logger.info(f"\nSections:")
    for i, section in enumerate(data['document']['sections'], 1):
        metadata = section.get('_metadata', {})
        logger.info(
            f"  {i}. {metadata.get('section_name', 'Unknown')} "
            f"(confidence: {metadata.get('confidence', 0):.2f})"
        )
    
    logger.info(f"\nFile location: {file_path}")


def approve_document(document_id: str, reviewer: str = None):
    """Approve a document."""
    storage.approve_document(document_id, reviewer)
    logger.info(f"✓ Document approved: {document_id}")


def reject_document(document_id: str, reason: str, reviewer: str = None):
    """Reject a document."""
    storage.reject_document(document_id, reason, reviewer)
    logger.warning(f"✗ Document rejected: {document_id}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate and review extracted documents'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all documents in validation queue'
    )
    parser.add_argument(
        '--review',
        type=str,
        help='Review a specific document'
    )
    parser.add_argument(
        '--approve',
        type=str,
        help='Approve a document'
    )
    parser.add_argument(
        '--reject',
        type=str,
        help='Reject a document'
    )
    parser.add_argument(
        '--reason',
        type=str,
        help='Rejection reason (required with --reject)'
    )
    parser.add_argument(
        '--reviewer',
        type=str,
        help='Reviewer name'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_validation_queue()
    
    elif args.review:
        review_document(args.review)
    
    elif args.approve:
        approve_document(args.approve, args.reviewer)
    
    elif args.reject:
        if not args.reason:
            logger.error("--reason is required with --reject")
            sys.exit(1)
        reject_document(args.reject, args.reason, args.reviewer)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
