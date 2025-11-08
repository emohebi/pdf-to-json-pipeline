"""
Stage 3: Validation & Aggregation Agent - DocuPorter Format
Validates and combines section JSONs into final document with DocuPorter format.
"""
from typing import Dict, List, Any
from datetime import datetime

from config.settings import CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD
from src.utils import setup_logger, StorageManager
from src.utils.docuporter_processor import (
    format_section_for_docuporter,
    clean_empty_fields
)

logger = setup_logger('validator_docuporter')


class ValidationAgentDocuPorter:
    """Agent to validate and combine section JSONs in DocuPorter format."""
    
    def __init__(self):
        self.logger = logger
        self.storage = StorageManager()
    
    def validate_and_combine(
        self,
        document_header: Dict,
        section_jsons: List[Dict],
        document_metadata: Dict,
        document_id: str
    ) -> Dict:
        """
        Combine document header and sections into final DocuPorter format.
        
        Args:
            document_header: Extracted document header
            section_jsons: List of extracted sections
            document_metadata: Document processing metadata
            document_id: Document ID
            
        Returns:
            Complete document in DocuPorter format
        """
        logger.info(f"[{document_id}] Combining document with {len(section_jsons)} sections")
        
        # Initialize document structure
        document_json = {
            'Document_Id': document_id,
            'Document_header': document_header
        }
        
        # Add section names to header
        section_names = []
        for section in section_jsons:
            section_name = section.get('section_name', 'Unknown')
            if section_name not in section_names:
                section_names.append(section_name)
        document_json['Document_header']['Sections'] = section_names
        
        # Process each section
        sections_by_type = {}
        unhandled_content = []
        
        for section in section_jsons:
            section_type = section.get('_metadata', {}).get('section_type', 'unhandled_content')
            section_data = section.get('data', [])
            
            # Map old names to new names
            if section_type == 'attached_images':
                section_type = 'attached images'
            
            # Format section data for DocuPorter
            formatted_data = format_section_for_docuporter(section_data, section_type)
            
            # Handle different section structures
            if section_type in ['safety', 'execution_condition']:
                # These are single object sections
                if section_type not in sections_by_type:
                    sections_by_type[section_type] = formatted_data
                else:
                    # Merge if multiple found
                    if isinstance(formatted_data, dict):
                        sections_by_type[section_type].update(formatted_data)
            elif section_type in [
                'material_risks_and_controls',
                'additional_controls_required',
                'additional_ppe_required',
                'specific_competencies_knowledge_and_skills',
                'tooling_equipment_required',
                'reference_documentation',
                'reference_drawings',
                'attached images',
                'task_activities'
            ]:
                # These are array sections
                if section_type not in sections_by_type:
                    sections_by_type[section_type] = []
                
                if isinstance(formatted_data, list):
                    sections_by_type[section_type].extend(formatted_data)
                else:
                    sections_by_type[section_type].append(formatted_data)
            else:
                # Unhandled content
                unhandled_item = {
                    'section': section.get('section_name', 'Unknown'),
                    'orig_text': '',
                    'orig_image': '',
                    'text': '',
                    'image': ''
                }
                # Try to extract some text from the data
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if isinstance(value, str) and value:
                            unhandled_item['orig_text'] = value
                            unhandled_item['text'] = value
                            break
                unhandled_content.append(unhandled_item)
        
        # Add sections to document
        section_order = [
            'safety',
            'material_risks_and_controls',
            'additional_controls_required',
            'additional_ppe_required',
            'specific_competencies_knowledge_and_skills',
            'tooling_equipment_required',
            'reference_documentation',
            'attached images',
            'task_activities',
            'unhandled_content'
        ]
        
        for section_type in section_order:
            if section_type in sections_by_type:
                document_json[section_type] = sections_by_type[section_type]
            elif section_type == 'unhandled_content':
                document_json['unhandled_content'] = unhandled_content
            else:
                # Add empty structure for missing sections
                document_json[section_type] = self._get_empty_section(section_type)
        
        # Clean empty fields
        document_json = clean_empty_fields(document_json)
        
        # Calculate confidence
        confidences = [
            s.get('_metadata', {}).get('confidence', 0.5)
            for s in section_jsons
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Determine if review needed
        needs_review = (
            avg_confidence < CONFIDENCE_THRESHOLD or
            any(c < LOW_CONFIDENCE_THRESHOLD for c in confidences)
        )
        
        # Store metadata separately if needed
        document_metadata.update({
            'confidence_score': avg_confidence,
            'needs_review': needs_review,
            'section_count': len(section_jsons)
        })
        
        logger.info(
            f"[{document_id}] Document combined successfully "
            f"(confidence: {avg_confidence:.2f}, review: {needs_review})"
        )
        self.storage.save_final_json(document_id, document_json)
        return document_json, document_metadata
    
    def _get_empty_section(self, section_type: str) -> Any:
        """Get empty structure for a section type."""
        from config.schemas_docuporter import SECTION_SCHEMAS
        
        schema = SECTION_SCHEMAS.get(section_type)
        if schema:
            # Return clean empty version
            return clean_empty_fields(schema)
        
        # Default empty structures
        if section_type == 'safety':
            return {
                'safety_icon': {},
                'safety_statement': [],
                'safety_additional': []
            }
        elif section_type in [
            'material_risks_and_controls',
            'additional_controls_required',
            'additional_ppe_required',
            'specific_competencies_knowledge_and_skills',
            'tooling_equipment_required',
            'reference_documentation',
            'attached images',
            'task_activities'
        ]:
            return []
        else:
            return {}
