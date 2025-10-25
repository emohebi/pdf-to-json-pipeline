"""
Document and section schemas definition.
Customized for safety procedure documents.
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
    'safety': {
        'type': 'object',
        'properties': {
            'safety_icon': {
                'type': 'object',
                'properties': {
                    'text': {'type': 'string', 'description': 'Text content from safety icon'},
                    'image': {'type': 'string', 'description': 'Base64 image or description of safety icon'}
                },
                'required': ['text', 'image']
            },
            'safety_statement': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'Safety statement text'},
                        'image': {'type': 'string', 'description': 'Associated image or icon'}
                    },
                    'required': ['text', 'image']
                },
                'description': 'List of safety statements'
            },
            'safety_additional': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'seq': {'type': 'string', 'description': 'Sequence number'},
                        'text': {'type': 'string', 'description': 'Additional safety note text'}
                    },
                    'required': ['seq', 'text']
                },
                'description': 'Additional safety notes'
            }
        },
        'required': ['safety_icon', 'safety_statement', 'safety_additional']
    },
    
    'material_risks_and_controls': {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'risk': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'Risk name or title'},
                        'image': {'type': 'string', 'description': 'Risk icon or image'}
                    },
                    'required': ['text', 'image']
                },
                'risk_description': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'Detailed risk description'},
                        'image': {'type': 'string', 'description': 'Risk diagram or image'}
                    },
                    'required': ['text', 'image']
                },
                'critical_controls': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'text': {'type': 'string', 'description': 'Control measure text'},
                            'image': {'type': 'string', 'description': 'Control measure icon or image'}
                        },
                        'required': ['text', 'image']
                    },
                    'description': 'List of critical controls for this risk'
                }
            },
            'required': ['risk', 'risk_description', 'critical_controls']
        },
        'description': 'Array of material risks and their controls'
    },
    
    'task_activities': {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'sequence_no': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'Sequence number'}
                    },
                    'required': ['text']
                },
                'sequence_name': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'Sequence name or title'}
                    },
                    'required': ['text']
                },
                'step_no': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'Step number'},
                        'image': {'type': 'string', 'description': 'Step number badge or icon'}
                    },
                    'required': ['text', 'image']
                },
                'step_description': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'text': {'type': 'string', 'description': 'Step description text'},
                            'image': {'type': 'string', 'description': 'Step image or diagram'}
                        },
                        'required': ['text', 'image']
                    },
                    'description': 'Step descriptions'
                },
                'photo_diagram': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'text': {'type': 'string', 'description': 'Photo/diagram caption'},
                            'image': {'type': 'string', 'description': 'Photo or diagram image'}
                        },
                        'required': ['text', 'image']
                    },
                    'description': 'Photos or diagrams for this step'
                },
                'notes': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'text': {'type': 'string', 'description': 'Note text'},
                            'image': {'type': 'string', 'description': 'Note icon or image'}
                        },
                        'required': ['text', 'image']
                    },
                    'description': 'Additional notes for this step'
                },
                'execution_condition': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'Execution condition or prerequisite'}
                    },
                    'required': ['text']
                }
            },
            'required': ['sequence_no', 'sequence_name', 'step_no', 'step_description', 'photo_diagram', 'notes', 'execution_condition']
        },
        'description': 'Array of task activities'
    },
    
    'additional_controls_required': {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'control_type': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'Type of control'},
                        'image': {'type': 'string', 'description': 'Control type icon'}
                    },
                    'required': ['text', 'image']
                },
                'reason_for_control': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'text': {'type': 'string', 'description': 'Reason text'},
                            'image': {'type': 'string', 'description': 'Reason icon or diagram'}
                        },
                        'required': ['text', 'image']
                    },
                    'description': 'Reasons for this control'
                }
            },
            'required': ['control_type', 'reason_for_control']
        },
        'description': 'Array of additional controls'
    },
    
    'additional_ppe_required': {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'text': {'type': 'string', 'description': 'PPE item description'},
                'image': {'type': 'string', 'description': 'PPE item icon or image'}
            },
            'required': ['text', 'image']
        },
        'description': 'Array of required PPE items'
    },
    
    'specific_competencies_knowledge_and_skills': {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'text': {'type': 'string', 'description': 'Competency, knowledge or skill description'},
                'image': {'type': 'string', 'description': 'Competency icon or badge'}
            },
            'required': ['text', 'image']
        },
        'description': 'Array of required competencies, knowledge and skills'
    },
    
    'tooling_equipment_required': {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'tool_set': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'Tool set name or category'}
                    },
                    'required': ['text']
                },
                'tools': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'text': {'type': 'string', 'description': 'Tool name or description'}
                        },
                        'required': ['text']
                    },
                    'description': 'List of tools in this set'
                }
            },
            'required': ['tool_set', 'tools']
        },
        'description': 'Array of tool sets and their tools'
    },
    
    'reference_documentation': {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'document_reference_number': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'Document reference number'},
                        'image': {'type': 'string', 'description': 'Document icon or badge'}
                    },
                    'required': ['text', 'image']
                },
                'document_description': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'Document description or title'},
                        'image': {'type': 'string', 'description': 'Document thumbnail or icon'}
                    },
                    'required': ['text', 'image']
                }
            },
            'required': ['document_reference_number', 'document_description']
        },
        'description': 'Array of reference documents'
    },
    
    'reference_drawings': {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'mechanical_drawings': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'text': {'type': 'string', 'description': 'Mechanical drawing reference or description'},
                            'image': {'type': 'string', 'description': 'Mechanical drawing image'}
                        },
                        'required': ['text', 'image']
                    },
                    'description': 'Mechanical drawings'
                },
                'structural_civil_drawings': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'text': {'type': 'string', 'description': 'Structural/civil drawing reference or description'},
                            'image': {'type': 'string', 'description': 'Structural/civil drawing image'}
                        },
                        'required': ['text', 'image']
                    },
                    'description': 'Structural and civil drawings'
                }
            },
            'required': ['mechanical_drawings', 'structural_civil_drawings']
        },
        'description': 'Array of reference drawings grouped by type'
    },
    
    'attached_images': {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'text': {'type': 'string', 'description': 'Image caption or description'},
                'image': {'type': 'string', 'description': 'Attached image'}
            },
            'required': ['text', 'image']
        },
        'description': 'Array of attached images'
    },
    
    'general': {
        'type': 'object',
        'properties': {
            'content': {
                'type': 'string',
                'description': 'General content'
            },
            'images_text': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Text from images'
            }
        },
        'required': ['content']
    }
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