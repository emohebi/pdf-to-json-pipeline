"""
Document and section schemas definition.
Customize these schemas based on your document structure.
"""
from typing import Dict, Any

# ============================================================================
# SECTION DEFINITIONS
# ============================================================================

SECTION_DEFINITIONS = {
    'header': 'Title page, document information, metadata, cover page',
    'summary': 'Executive summary, abstract, or overview',
    'introduction': 'Introduction, background, or context section',
    'body': 'Main content sections, detailed information',
    'tables': 'Data tables, statistical information, tabular data',
    'figures': 'Charts, graphs, diagrams, images with captions',
    'results': 'Results, findings, outcomes section',
    'discussion': 'Discussion, analysis, interpretation section',
    'conclusion': 'Conclusion, summary of findings',
    'appendix': 'Supplementary information, additional data',
    'references': 'Citations, bibliography, references',
    'general': 'General content section (fallback)'
}


# ============================================================================
# SECTION SCHEMAS
# ============================================================================

SECTION_SCHEMAS: Dict[str, Dict[str, Any]] = {
    'header': {
        'type': 'object',
        'properties': {
            'title': {
                'type': 'string',
                'description': 'Document title'
            },
            'subtitle': {
                'type': 'string',
                'description': 'Document subtitle if present'
            },
            'author': {
                'type': ['string', 'array'],
                'description': 'Author name(s)'
            },
            'date': {
                'type': 'string',
                'description': 'Document date'
            },
            'document_number': {
                'type': 'string',
                'description': 'Document ID or reference number'
            },
            'organization': {
                'type': 'string',
                'description': 'Organization or company name'
            },
            'version': {
                'type': 'string',
                'description': 'Document version'
            }
        },
        'required': ['title']
    },
    
    'summary': {
        'type': 'object',
        'properties': {
            'text': {
                'type': 'string',
                'description': 'Full summary text'
            },
            'key_points': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'List of key points or highlights'
            },
            'word_count': {
                'type': 'integer',
                'description': 'Approximate word count'
            }
        },
        'required': ['text']
    },
    
    'introduction': {
        'type': 'object',
        'properties': {
            'paragraphs': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Introduction paragraphs'
            },
            'objectives': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Document objectives if stated'
            },
            'scope': {
                'type': 'string',
                'description': 'Scope of the document'
            }
        },
        'required': ['paragraphs']
    },
    
    'body': {
        'type': 'object',
        'properties': {
            'subsections': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'heading': {'type': 'string'},
                        'content': {'type': 'string'},
                        'level': {'type': 'integer'}
                    }
                },
                'description': 'Main content organized by subsections'
            },
            'paragraphs': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Content paragraphs'
            },
            'images_text': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Text extracted from images in this section'
            },
            'bullet_points': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Bullet points or lists'
            }
        },
        'required': ['paragraphs']
    },
    
    'tables': {
        'type': 'object',
        'properties': {
            'tables': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'title': {'type': 'string'},
                        'headers': {
                            'type': 'array',
                            'items': {'type': 'string'}
                        },
                        'rows': {
                            'type': 'array',
                            'items': {
                                'type': 'array',
                                'items': {'type': 'string'}
                            }
                        },
                        'caption': {'type': 'string'},
                        'table_number': {'type': 'string'}
                    }
                },
                'description': 'Structured table data'
            }
        },
        'required': ['tables']
    },
    
    'figures': {
        'type': 'object',
        'properties': {
            'figures': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'figure_number': {'type': 'string'},
                        'title': {'type': 'string'},
                        'caption': {'type': 'string'},
                        'description': {'type': 'string'},
                        'text_in_image': {
                            'type': 'array',
                            'items': {'type': 'string'},
                            'description': 'Text extracted from the figure'
                        }
                    }
                },
                'description': 'Figures, charts, and diagrams'
            }
        },
        'required': ['figures']
    },
    
    'results': {
        'type': 'object',
        'properties': {
            'findings': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Key findings'
            },
            'data': {
                'type': 'array',
                'items': {'type': 'object'},
                'description': 'Result data points'
            },
            'text': {
                'type': 'string',
                'description': 'Full results text'
            }
        },
        'required': ['text']
    },
    
    'discussion': {
        'type': 'object',
        'properties': {
            'paragraphs': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Discussion paragraphs'
            },
            'implications': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Implications mentioned'
            }
        },
        'required': ['paragraphs']
    },
    
    'conclusion': {
        'type': 'object',
        'properties': {
            'summary': {
                'type': 'string',
                'description': 'Conclusion summary'
            },
            'recommendations': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Recommendations if any'
            },
            'future_work': {
                'type': 'string',
                'description': 'Future work section if present'
            }
        },
        'required': ['summary']
    },
    
    'appendix': {
        'type': 'object',
        'properties': {
            'sections': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'title': {'type': 'string'},
                        'content': {'type': 'string'}
                    }
                },
                'description': 'Appendix sections'
            }
        }
    },
    
    'references': {
        'type': 'object',
        'properties': {
            'references': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'List of references'
            },
            'count': {
                'type': 'integer',
                'description': 'Total reference count'
            }
        },
        'required': ['references']
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
