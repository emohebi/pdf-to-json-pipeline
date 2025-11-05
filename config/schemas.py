"""
Document and section schemas definition.
Customized for safety procedure documents.
FIXED: task_activities now uses proper hierarchical structure (sequences contain steps)
"""
from typing import Dict, Any

# ============================================================================
# SECTION DEFINITIONS
# ============================================================================

SECTION_DEFINITIONS = {
    'safety': 'Safety information section including icons, statements, and additional safety notes',
    'material_risks_and_controls': 'Material risks and their associated critical controls',
    'task_activities': 'Task activities with sequences, steps, descriptions, photos/diagrams, and notes',
    'additional_controls_required': 'Additional controls required with control types and reasons',
    'additional_ppe_required': 'Additional personal protective equipment (PPE) required',
    'specific_competencies_knowledge_and_skills': 'Specific competencies, knowledge and skills required',
    'tooling_equipment_required': 'Tooling and equipment required including tool sets',
    'reference_documentation': 'Reference documentation with document numbers and descriptions',
    'reference_drawings': 'Reference drawings including mechanical and structural/civil drawings',
    'attached_images': 'Attached images and diagrams',
    'general': 'General content section (fallback)'
}


# ============================================================================
# SECTION SCHEMAS
# ============================================================================

SECTION_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "safety": {
        "safety_icon": {"text": "", "image": ""},
        "safety_statement": [{"text": "", "image": ""}],
        "safety_additional": [{"seq": "", "text" : ""}]
    },
    "material_risks_and_controls": [
        {
            "risk": {"text": "", "image": ""},
            "risk_description": {"text": "", "image": ""},
            "critical_controls": [{"text": "", "image": ""}]
        }
    ],
    "additional_controls_required": [
        {
            "control_type": {"text": "", "image": ""},
            "reason_for_control": [{"text": "", "image": ""}]
        }
    ],
    "additional_ppe_required": [
        {"text": "", "image": ""}
    ],
    "specific_competencies_knowledge_and_skills": [
        {"text": "", "image": ""}
    ],
    "tooling_equipment_required": [
        {
            "tool_set": {"text": ""},
            "tools": [{"text": ""}]
        }
    ],
    "reference_documentation": [
        {
            "document_reference_number": {"text": "", "image": ""},
            "document_description": {"text": "", "image": ""}
        }
    ],
    "reference_drawings": [
        {
            "mechanical_drawings": [{"text": "", "image": ""}],
            "structural_civil_drawings": [{"text": "", "image": ""}]
        }
    ],
    "attached_images": [{"text": "", "image": ""}],
    "task_activities": [
        {
            "sequence_no": {"text": ""},
            "sequence_name": {"text": ""},
            "equipment_asset": {"text": ""},
            "maintainable_item": [{"text": ""}],
            "lmi": [{"text": ""}],
            "steps": [
                {
                    "step_no": {"text": "", "image": ""},
                    "step_description": [{"text": "", "image": ""}],
                    "photo_diagram": [{"text": "", "image": ""}],
                    "notes": [{"text": "", "image": ""}],
                    "acceptable_limit": [{"text": "", "image": ""}],
                    "question": [{"text": "", "image": ""}],
                    "corrective_action": [{"text": "", "image": ""}],
                    "execution_condition": {"text": "", "image": ""},
                    "other_content": [{"text": "", "image": ""}]
                }
            ]
        }
    ],
    "general": [
        {"text": "", "image": ""}
    ]
}


# ============================================================================
# DOCUMENT SCHEMA
# ============================================================================

DOCUMENT_SCHEMA = {
    'type': 'object',
    'properties': {
        'document_id': {
            'type': 'string',
            'description': 'Unique document identifier'
        },
        'metadata': {
            'type': 'object',
            'properties': {
                'source_file': {'type': 'string'},
                'total_pages': {'type': 'integer'},
                'processing_timestamp': {'type': 'string'},
                'processing_duration': {'type': 'number'},
                'model_used': {'type': 'string'},
                'confidence_score': {'type': 'number'}
            }
        },
        'sections': {
            'type': 'array',
            'items': {
                'type': 'object',
                'description': 'Document sections'
            }
        },
        'validation_status': {
            'type': 'string',
            'enum': ['pending', 'approved', 'rejected', 'in_review']
        }
    },
    'required': ['document_id', 'metadata', 'sections']
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_section_schema(section_type: str) -> Dict[str, Any]:
    """
    Get schema for a specific section type.
    Falls back to 'general' if type not found.
    """
    return SECTION_SCHEMAS.get(section_type, SECTION_SCHEMAS['general'])


def get_all_section_types() -> list:
    """Get list of all defined section types."""
    return list(SECTION_DEFINITIONS.keys())


def validate_section_type(section_type: str) -> bool:
    """Check if section type is valid."""
    return section_type in SECTION_DEFINITIONS