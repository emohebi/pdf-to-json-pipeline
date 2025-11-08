"""
Configuration package for PDF to JSON pipeline.
"""
from .settings import *
from .schemas_docuporter import *

__all__ = [
    'SECTION_DEFINITIONS',
    'SECTION_SCHEMAS',
    'DOCUMENT_SCHEMA',
    'get_section_schema',
    'get_all_section_types',
    'validate_section_type'
]
