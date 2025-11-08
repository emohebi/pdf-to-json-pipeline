"""
Document Header Extraction Agent
Extracts document metadata from the first page of PDF documents.
"""
import json
from typing import Dict, Any

from config.settings import MODEL_MAX_TOKENS_EXTRACTION
from src.tools import invoke_bedrock_multimodal, prepare_images_for_bedrock
from src.utils import setup_logger

logger = setup_logger('document_header_extractor')


class DocumentHeaderExtractor:
    """Agent to extract document header information from the first page."""
    
    def __init__(self):
        self.logger = logger
    
    def extract_header(
        self,
        first_page_data: Dict,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Extract document header information from the first page.
        
        Args:
            first_page_data: Page data dict with image
            document_id: Document ID
            
        Returns:
            Document header dictionary
        """
        logger.info(f"[{document_id}] Extracting document header from first page")
        
        try:
            # Prepare image
            images_b64 = prepare_images_for_bedrock([first_page_data])
            
            # Build prompt
            prompt = self._build_header_extraction_prompt()
            
            # Extract header
            response = invoke_bedrock_multimodal(
                images=images_b64,
                prompt=prompt,
                max_tokens=MODEL_MAX_TOKENS_EXTRACTION
            )
            
            # Parse response
            header_data = self._parse_header_response(response)
            
            # Add document ID and sections list
            header_data['Sections'] = []
            
            logger.info(f"[{document_id}] Successfully extracted document header")
            return header_data
            
        except Exception as e:
            logger.error(f"[{document_id}] Failed to extract header: {e}")
            # Return empty header structure
            return self._get_empty_header()
    
    def _build_header_extraction_prompt(self) -> str:
        """Build prompt for document header extraction."""
        return """Extract document header information from this first page of the document.

Look for the following information typically found at the top of technical/safety documents:
1. Document Source - The organization or entity that created the document
2. Document Type - Type of document (e.g., "Work Method Statement", "Safety Procedure", etc.)
3. Document Number - The unique document identifier/reference number
4. Document Version Number - Version or revision number
5. Work Description - Title or description of the work/procedure
6. Purpose - The purpose or objective of the document

CRITICAL: For each field, extract the EXACT text as it appears in the document.
Duplicate the same value for both "orig_text" and "text" fields.

REQUIRED JSON STRUCTURE:
{
    "Document_Source": {
        "orig_text": "exact text from document",
        "text": "exact text from document"
    },
    "Document_Type": {
        "orig_text": "exact text from document",
        "text": "exact text from document"
    },
    "Document_Number": {
        "orig_text": "exact text from document",
        "text": "exact text from document"
    },
    "Document_Version_Number": {
        "orig_text": "exact text from document",
        "text": "exact text from document"
    },
    "Work_Description": {
        "orig_text": "exact text from document",
        "text": "exact text from document"
    },
    "Purpose": {
        "orig_text": "exact text from document",
        "text": "exact text from document"
    }
}

EXTRACTION RULES:
1. Look for labeled fields (e.g., "Document No:", "Version:", "Type:", etc.)
2. Extract the EXACT text - do not paraphrase or modify
3. Both "orig_text" and "text" should have the SAME value
4. If a field is not found, use empty strings for both orig_text and text
5. Common locations:
   - Headers and footers
   - Title blocks
   - Document control sections
   - Top of the first page

EXAMPLES:
- If you see "Document No: WMS-2024-001" → extract "WMS-2024-001"
- If you see "Rev 2.1" → extract "2.1" for version
- If you see "Work Method Statement" → extract exactly as shown

Return ONLY the JSON object (no markdown, start with {):
"""
    
    def _parse_header_response(self, response: str) -> Dict[str, Any]:
        """Parse the header extraction response."""
        try:
            # Clean response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            header_data = json.loads(response)
            
            # Validate structure and ensure all fields exist
            required_fields = [
                'Document_Source', 'Document_Type', 'Document_Number',
                'Document_Version_Number', 'Work_Description', 'Purpose'
            ]
            
            for field in required_fields:
                if field not in header_data:
                    header_data[field] = {"orig_text": "", "text": ""}
                elif not isinstance(header_data[field], dict):
                    # Convert to proper structure
                    value = str(header_data[field])
                    header_data[field] = {"orig_text": value, "text": value}
                else:
                    # Ensure both orig_text and text exist and match
                    if 'text' in header_data[field] and 'orig_text' not in header_data[field]:
                        header_data[field]['orig_text'] = header_data[field]['text']
                    elif 'orig_text' in header_data[field] and 'text' not in header_data[field]:
                        header_data[field]['text'] = header_data[field]['orig_text']
            
            return header_data
            
        except Exception as e:
            logger.error(f"Failed to parse header response: {e}")
            return self._get_empty_header()
    
    def _get_empty_header(self) -> Dict[str, Any]:
        """Get empty header structure."""
        return {
            "Document_Source": {"orig_text": "", "text": ""},
            "Document_Type": {"orig_text": "", "text": ""},
            "Document_Number": {"orig_text": "", "text": ""},
            "Document_Version_Number": {"orig_text": "", "text": ""},
            "Work_Description": {"orig_text": "", "text": ""},
            "Purpose": {"orig_text": "", "text": ""}
        }
