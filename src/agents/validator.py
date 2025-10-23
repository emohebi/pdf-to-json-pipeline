"""
Stage 3: Validation & Aggregation Agent
Validates and combines section JSONs into final document.
"""
from typing import Dict, List
from datetime import datetime

from config.settings import CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD
from src.tools.validation import validate_document_structure
from src.utils import setup_logger, StorageManager

logger = setup_logger('validator')


class ValidationAgent:
    """Agent to validate and combine section JSONs."""
    
    def __init__(self):
        self.storage = StorageManager()
    
    def validate_and_combine(
        self,
        section_jsons: List[Dict],
        document_metadata: Dict,
        document_id: str
    ) -> Dict:
        logger.info(f"[{document_id}] Validating {len(section_jsons)} sections")
        
        # Calculate overall confidence
        confidences = [
            s.get('_metadata', {}).get('confidence', 0.5)
            for s in section_jsons
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Build final document
        document_json = {
            'document_id': document_id,
            'metadata': {
                **document_metadata,
                'confidence_score': avg_confidence,
                'section_count': len(section_jsons)
            },
            'sections': section_jsons,
            'validation_status': 'pending'
        }
        
        # Validate structure
        is_valid, errors = validate_document_structure(document_json)
        
        if not is_valid:
            logger.warning(f"[{document_id}] Validation errors: {errors}")
            document_json['validation_errors'] = errors
        
        # Determine if human validation needed
        needs_review = (
            avg_confidence < CONFIDENCE_THRESHOLD or
            not is_valid or
            any(c < LOW_CONFIDENCE_THRESHOLD for c in confidences)
        )
        
        if needs_review:
            reason = self._determine_review_reason(
                avg_confidence, is_valid, confidences
            )
            
            self.storage.queue_for_validation(
                document_id,
                document_json,
                reason
            )
            
            logger.warning(f"[{document_id}] Queued for review: {reason}")
        else:
            # Save to final output
            self.storage.save_final_json(document_id, document_json)
            logger.info(f"[{document_id}] Validation passed, saved to final")
        
        return document_json
    
    def _determine_review_reason(
        self,
        avg_confidence: float,
        is_valid: bool,
        confidences: List[float]
    ) -> str:
        reasons = []
        
        if avg_confidence < LOW_CONFIDENCE_THRESHOLD:
            reasons.append(f"Very low confidence ({avg_confidence:.2f})")
        elif avg_confidence < CONFIDENCE_THRESHOLD:
            reasons.append(f"Low confidence ({avg_confidence:.2f})")
        
        if not is_valid:
            reasons.append("Structure validation failed")
        
        low_sections = sum(1 for c in confidences if c < LOW_CONFIDENCE_THRESHOLD)
        if low_sections > 0:
            reasons.append(f"{low_sections} section(s) with very low confidence")
        
        return "; ".join(reasons) if reasons else "Unknown issue"
