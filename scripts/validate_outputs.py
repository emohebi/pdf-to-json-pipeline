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

logger = setup_logger("validate_outputs")
storage = StorageManager()


def list_validation_queue():
    queue = storage.get_validation_queue()
    if not queue:
        logger.info("Validation queue is empty!")
        return
    logger.info(f"\n{len(queue)} document(s) in validation queue:\n")
    for i, item in enumerate(queue, 1):
        logger.info(f"{i:<4} {item['document_id']:<30} {item['reason']:<50}")


def main():
    parser = argparse.ArgumentParser(description="Validate and review extracted documents")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--approve", type=str)
    parser.add_argument("--reject", type=str)
    parser.add_argument("--reason", type=str)
    parser.add_argument("--reviewer", type=str)
    args = parser.parse_args()

    if args.list:
        list_validation_queue()
    elif args.approve:
        storage.approve_document(args.approve, args.reviewer)
    elif args.reject:
        if not args.reason:
            logger.error("--reason is required with --reject")
            sys.exit(1)
        storage.reject_document(args.reject, args.reason, args.reviewer)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
