"""
Storage utilities for intermediate and final results.
Extended with review agent support.
"""
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


# Import from config - we'll create a minimal version if needed
try:
    from config.settings import (
        DETECTION_DIR, SECTIONS_DIR, VALIDATION_QUEUE_DIR,
        FINAL_DIR, PROGRESS_FILE, VALIDATION_STATE_PENDING,
        INTERMEDIATE_DIR
    )
except ImportError:
    # Fallback for testing
    from pathlib import Path
    INTERMEDIATE_DIR = Path('output/intermediate')
    DETECTION_DIR = INTERMEDIATE_DIR / 'detection'
    SECTIONS_DIR = INTERMEDIATE_DIR / 'sections'
    VALIDATION_QUEUE_DIR = INTERMEDIATE_DIR / 'validation_queue'
    FINAL_DIR = Path('output/final')
    PROGRESS_FILE = Path('output/logs/progress.json')
    VALIDATION_STATE_PENDING = 'pending'

from src.utils.logger import setup_logger

logger = setup_logger('storage')


class StorageManager:
    """Manages storage of intermediate and final results."""
    
    def __init__(self):
        """Initialize storage manager and create review directories."""
        # Create review directories if they don't exist
        self.review_dir = INTERMEDIATE_DIR / 'review'
        self.plain_text_dir = self.review_dir / 'plain_text'
        self.review_results_dir = self.review_dir / 'results'
        
        for directory in [self.review_dir, self.plain_text_dir, self.review_results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_plain_text(self, document_id: str, plain_text: str) -> Path:
        """
        Save plain text version of document for review.
        
        Args:
            document_id: Unique document identifier
            plain_text: Plain text content
        
        Returns:
            Path to saved file
        """
        output_file = self.plain_text_dir / f"{document_id}_plaintext.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(plain_text)
        
        logger.info(f"Saved plain text: {output_file}")
        return output_file
    
    def save_review_results(self, document_id: str, review_results: Dict[str, Any]) -> Path:
        """
        Save review results from the review agent.
        
        Args:
            document_id: Unique document identifier
            review_results: Dictionary containing review findings
        
        Returns:
            Path to saved file
        """
        output_file = self.review_results_dir / f"{document_id}_review.json"
        
        data = {
            'document_id': document_id,
            'timestamp': datetime.now().isoformat(),
            'review_results': review_results,
            'has_issues': any(
                len(issues) > 0 
                for key, issues in review_results.items() 
                if key != 'error' and isinstance(issues, list)
            ),
            'total_issues': sum(
                len(issues) 
                for key, issues in review_results.items() 
                if key != 'error' and isinstance(issues, list)
            )
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        if data['has_issues']:
            logger.warning(
                f"Saved review results with {data['total_issues']} issue(s): {output_file}"
            )
        else:
            logger.info(f"Saved review results (no issues): {output_file}")
        
        return output_file
    
    @staticmethod
    def save_detection_result(document_id: str, sections: List[Dict]) -> Path:
        """
        Save section detection results for human validation.
        
        Args:
            document_id: Unique document identifier
            sections: List of detected sections
        
        Returns:
            Path to saved file
        """
        output_file = DETECTION_DIR / f"{document_id}_sections.json"
        
        data = {
            'document_id': document_id,
            'timestamp': datetime.now().isoformat(),
            'sections': sections,
            'section_count': len(sections)
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved detection result: {output_file}")
        return output_file
    
    @staticmethod
    def save_section_json(
        document_id: str, 
        section_name: str, 
        section_data: Dict,
        confidence: float = None
    ) -> Path:
        """
        Save individual section JSON for validation.
        
        Args:
            document_id: Unique document identifier
            section_name: Name of the section
            section_data: Extracted section data
            confidence: Optional confidence score
        
        Returns:
            Path to saved file
        """
        # Sanitize section name for filename
        safe_name = section_name.replace(' ', '_').replace('/', '_')
        output_file = SECTIONS_DIR / f"{document_id}_{safe_name}.json"
        
        data = {
            'document_id': document_id,
            'section_name': section_name,
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'data': section_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved section: {output_file}")
        return output_file
    
    @staticmethod
    def save_final_json(document_id: str, document_data: Dict) -> Path:
        """
        Save final combined document JSON.
        
        Args:
            document_id: Unique document identifier
            document_data: Complete document data
        
        Returns:
            Path to saved file
        """
        output_file = FINAL_DIR / f"{document_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(document_data, f, indent=2)
        
        logger.info(f"Saved final document: {output_file}")
        return output_file
    
    @staticmethod
    def queue_for_validation(
        document_id: str,
        document_data: Dict,
        reason: str = "Low confidence score"
    ) -> Path:
        """
        Queue document for human validation.
        
        Args:
            document_id: Unique document identifier
            document_data: Document data needing review
            reason: Reason for validation
        
        Returns:
            Path to validation queue file
        """
        output_file = VALIDATION_QUEUE_DIR / f"{document_id}_review.json"
        
        data = {
            'document_id': document_id,
            'timestamp': datetime.now().isoformat(),
            'status': VALIDATION_STATE_PENDING,
            'reason': reason,
            'document': document_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.warning(f"Queued for validation: {document_id} - {reason}")
        return output_file
    
    @staticmethod
    def load_json(file_path: Path) -> Dict:
        """
        Load JSON file.
        
        Args:
            file_path: Path to JSON file
        
        Returns:
            Loaded JSON data
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def update_progress(
        completed: List[str],
        failed: List[str] = None,
        total: int = None
    ):
        """
        Update progress tracking file.
        
        Args:
            completed: List of completed document IDs
            failed: List of failed document IDs
            total: Total number of documents to process
        """
        progress_data = {
            'last_updated': datetime.now().isoformat(),
            'completed': completed,
            'failed': failed or [],
            'completed_count': len(completed),
            'failed_count': len(failed) if failed else 0,
            'total': total
        }
        
        if total:
            progress_data['progress_percentage'] = (
                len(completed) / total * 100
            )
        
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    @staticmethod
    def load_progress() -> Dict:
        """
        Load progress tracking data.
        
        Returns:
            Progress data dictionary
        """
        if not PROGRESS_FILE.exists():
            return {
                'completed': [],
                'failed': [],
                'completed_count': 0,
                'failed_count': 0
            }
        
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def get_validation_queue() -> List[Dict]:
        """
        Get all documents in validation queue.
        
        Returns:
            List of documents needing validation
        """
        queue = []
        
        for file_path in VALIDATION_QUEUE_DIR.glob('*_review.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
                if data.get('status') == VALIDATION_STATE_PENDING:
                    queue.append({
                        'document_id': data['document_id'],
                        'timestamp': data['timestamp'],
                        'reason': data['reason'],
                        'file_path': str(file_path)
                    })
        
        return sorted(queue, key=lambda x: x['timestamp'])
    
    @staticmethod
    def approve_document(document_id: str, reviewer: str = None):
        """
        Approve a document from validation queue.
        
        Args:
            document_id: Document to approve
            reviewer: Name of reviewer
        """
        file_path = VALIDATION_QUEUE_DIR / f"{document_id}_review.json"
        
        if not file_path.exists():
            logger.error(f"Validation file not found: {file_path}")
            return
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Update status
        data['status'] = 'approved'
        data['reviewed_at'] = datetime.now().isoformat()
        data['reviewer'] = reviewer
        
        # Save updated status
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Move to final
        StorageManager.save_final_json(document_id, data['document'])
        
        logger.info(f"Document approved: {document_id}")
    
    @staticmethod
    def reject_document(document_id: str, reason: str, reviewer: str = None):
        """
        Reject a document from validation queue.
        
        Args:
            document_id: Document to reject
            reason: Rejection reason
            reviewer: Name of reviewer
        """
        file_path = VALIDATION_QUEUE_DIR / f"{document_id}_review.json"
        
        if not file_path.exists():
            logger.error(f"Validation file not found: {file_path}")
            return
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Update status
        data['status'] = 'rejected'
        data['reviewed_at'] = datetime.now().isoformat()
        data['reviewer'] = reviewer
        data['rejection_reason'] = reason
        
        # Save updated status
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.warning(f"Document rejected: {document_id} - {reason}")