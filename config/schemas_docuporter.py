"""
Document and section schemas definition - DocuPorter Format V1.
Updated to match the new JSON structure with orig_text/text pairs and document header.
"""
from typing import Dict, Any

# ============================================================================
# DOCUMENT HEADER SCHEMA
# ============================================================================

DOCUMENT_HEADER_SCHEMA = {
    "document_source": {
        "orig_text": "",
        "text": ""
    },
    "document_type": {
        "orig_text": "",
        "text": ""
    },
    "document_number": {
        "orig_text": "",
        "text": ""
    },
    "document_version_number": {
        "orig_text": "",
        "text": ""
    },
    "work_description": {
        "orig_text": "",
        "text": ""
    },
    "purpose": {
        "orig_text": "",
        "text": ""
    },
    "sections": []
}

# ============================================================================
# SECTION DEFINITIONS
# ============================================================================

SECTION_DEFINITIONS = {
    'safety': 'Safety information section including icons, statements, and additional safety notes',
    'material_risks_and_controls': 'Material risks and their associated critical controls',
    'task_activities': 'Task activities, including Pre-Task Activities and Post-Task Activities, with sequences and steps in flat structure',
    'additional_controls_required': 'Additional controls required with control types and reasons',
    'additional_ppe_required': 'Additional personal protective equipment (PPE) required',
    'specific_competencies_knowledge_and_skills': 'Specific competencies, knowledge and skills required',
    'tooling_equipment_required': 'Tooling and equipment required including tool sets',
    'reference_documentation': 'Reference documentation with document numbers and descriptions',
    'reference_drawings': 'Reference drawings including mechanical and structural/civil drawings',
    'attached images': 'Attached images and diagrams',
    'unhandled_content': 'Content that does not fit into defined sections'
}


# ============================================================================
# SECTION SCHEMAS
# ============================================================================

SECTION_SCHEMAS: Dict[str, Any] = {
    "safety": {
        "safety_icon": {
            "orig_text": "", 
            "orig_image": "",
            "text": "", 
            "image": ""
        },
        "safety_statement": [{
            "orig_text": "", 
            "orig_image": "",
            "text": "", 
            "image": ""
        }],
        "safety_additional": [{
            "orig_seq": "",
            "orig_text": "",
            "seq": "", 
            "text": ""
        }]
    },
    
    "material_risks_and_controls": [
        {
            "risk": {
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            },
            "risk_description": {
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            },
            "critical_controls": [{
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            }]
        }
    ],
    
    "additional_controls_required": [
        {
            "control_type": {
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            },
            "reason_for_control": [{
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            }]
        }
    ],
    
    "additional_ppe_required": [
        {
            "orig_text": "", 
            "orig_image": "",
            "text": "", 
            "image": ""
        }
    ],
    
    "specific_competencies_knowledge_and_skills": [
        {
            "orig_text": "", 
            "orig_image": "",
            "text": "", 
            "image": ""
        }
    ],
    
    "tooling_equipment_required": [
        {
            "tool_set": {
                "orig_text": "",
                "text": ""
            },
            "tools": [{
                "orig_text": "",
                "text": ""
            }]
        }
    ],
    
    "reference_documentation": [
        {
            "document_reference_number": {
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            },
            "document_description": {
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            }
        }
    ],
    
    "reference_drawings": [
        {
            "mechanical_drawings": [{
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            }],
            "structural_civil_drawings": [{
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            }]
        }
    ],
    
    "attached images": [
        {
            "orig_text": "", 
            "orig_image": "",
            "text": "", 
            "image": ""
        }
    ],
    
    "task_activities": [
        {
            "equipment_asset": {
                "orig_text": "",
                "text": ""
            },
            "sequence_no": {
                "orig_text": "",
                "text": ""
            },
            "sequence_name": {
                "orig_text": "",
                "text": ""
            },
            "maintainable_item": [{
                "orig_text": "",
                "text": ""
            }],
            "lmi": [{
                "orig_text": "",
                "text": ""
            }],
            "step_no": {
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            },
            "step_description": [{
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            }],
            "photo_diagram": [{
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            }],
            "notes": [{
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            }],
            "acceptable_limit": [{
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            }],
            "question": [{
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            }],
            "corrective_action": [{
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            }],
            "execution_condition": {
                "orig_text": "",
                "text": ""
            },
            "other_content": [{
                "orig_text": "", 
                "orig_image": "",
                "text": "", 
                "image": ""
            }]
        }
    ],
    
    "unhandled_content": [
        {
            "section": "",
            "orig_text": "", 
            "orig_image": "",
            "text": "", 
            "image": ""
        }
    ]
}


# ============================================================================
# DOCUMENT SCHEMA
# ============================================================================

DOCUMENT_SCHEMA = {
    'document_id': '',
    'document_header': DOCUMENT_HEADER_SCHEMA,
    'safety': SECTION_SCHEMAS['safety'],
    'material_risks_and_controls': SECTION_SCHEMAS['material_risks_and_controls'],
    'additional_controls_required': SECTION_SCHEMAS['additional_controls_required'],
    'additional_ppe_required': SECTION_SCHEMAS['additional_ppe_required'],
    'specific_competencies_knowledge_and_skills': SECTION_SCHEMAS['specific_competencies_knowledge_and_skills'],
    'tooling_equipment_required': SECTION_SCHEMAS['tooling_equipment_required'],
    'reference_documentation': SECTION_SCHEMAS['reference_documentation'],
    'attached images': SECTION_SCHEMAS['attached images'],
    'task_activities': SECTION_SCHEMAS['task_activities'],
    'unhandled_content': SECTION_SCHEMAS['unhandled_content']
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_section_schema(section_type: str) -> Any:
    """
    Get schema for a specific section type.
    Falls back to 'unhandled_content' if type not found.
    """
    # Map old names to new names if needed
    name_mapping = {
        'attached_images': 'attached images'
    }
    
    section_type = name_mapping.get(section_type, section_type)
    
    return SECTION_SCHEMAS.get(section_type, SECTION_SCHEMAS['unhandled_content'])


def get_all_section_types() -> list:
    """Get list of all defined section types."""
    return list(SECTION_DEFINITIONS.keys())


def validate_section_type(section_type: str) -> bool:
    """Check if section type is valid."""
    return section_type in SECTION_DEFINITIONS


def clean_empty_fields(data: Any) -> Any:
    """
    Clean up empty fields according to rule:
    - [{"text": "", "image": ""}] -> []
    - {"text": "", "image": ""} -> {}
    """
    if isinstance(data, dict):
        # Check if all text/image fields are empty
        has_content = False
        cleaned = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                if value:
                    has_content = True
                cleaned[key] = value
            else:
                cleaned_value = clean_empty_fields(value)
                # Check if cleaned value has content
                if cleaned_value:
                    has_content = True
                cleaned[key] = cleaned_value
        
        # Special case: if dict only has text/image fields and both are empty
        if not has_content and set(data.keys()).issubset({'text', 'image', 'orig_text', 'orig_image'}):
            return {}
        
        return cleaned if has_content else {}
        
    elif isinstance(data, list):
        cleaned = []
        for item in data:
            cleaned_item = clean_empty_fields(item)
            # Only add non-empty items
            if cleaned_item:
                cleaned.append(cleaned_item)
        return cleaned
        
    else:
        return data


def duplicate_to_orig_fields(data: Any) -> Any:
    """
    Duplicate text/image values to orig_text/orig_image fields.
    Both fields will have the same value.
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key == 'text' and 'orig_text' not in data:
                result['orig_text'] = value
                result['text'] = value
            elif key == 'image' and 'orig_image' not in data:
                result['orig_image'] = value
                result['image'] = value
            elif key == 'seq' and 'orig_seq' not in data:
                result['orig_seq'] = value
                result['seq'] = value
            elif key not in ['orig_text', 'orig_image', 'orig_seq']:
                result[key] = duplicate_to_orig_fields(value)
            else:
                result[key] = value
        return result
        
    elif isinstance(data, list):
        return [duplicate_to_orig_fields(item) for item in data]
    else:
        return data
