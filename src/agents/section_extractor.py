"""
Stage 2: Section Extraction Agent - Exact Text Extraction Version
Extracts structured data from document sections with EXACT text preservation.
"""
import json
import re
from typing import Dict, List, Tuple, Any, Union

from config.settings import MODEL_MAX_TOKENS_EXTRACTION
from config.schemas import get_section_schema
from src.tools import invoke_bedrock_multimodal, prepare_images_for_bedrock
from src.utils import setup_logger, StorageManager

logger = setup_logger('section_extractor')


# ============================================================================
# HELPER FUNCTIONS (self-contained, no external dependencies)
# ============================================================================

def clean_json_response(response: str) -> str:
    """Clean and fix common JSON issues from LLM responses."""
    response = response.strip()
    
    # Remove markdown code blocks
    if response.startswith('```json'):
        response = response[7:]
    if response.startswith('```'):
        response = response[3:]
    if response.endswith('```'):
        response = response[:-3]
    
    response = response.strip()
    
    # Fix trailing commas
    response = re.sub(r',\s*([}\]])', r'\1', response)
    
    # Try to find valid JSON if response has extra text
    # Look for first { or [
    start_idx = -1
    for i, char in enumerate(response):
        if char in ['{', '[']:
            start_idx = i
            break
    
    if start_idx > 0:
        response = response[start_idx:]
    
    # Try to find last } or ]
    end_idx = -1
    for i in range(len(response) - 1, -1, -1):
        if response[i] in ['}', ']']:
            end_idx = i + 1
            break
    
    if end_idx > 0:
        response = response[:end_idx]
    
    return response


def check_dict_empty(data: Dict[str, Any]) -> bool:
    """
    Check if a dictionary has mostly empty values.
    
    Args:
        data: Dictionary to check
        
    Returns:
        True if mostly empty, False otherwise
    """
    if not data:
        return True
    
    empty_count = 0
    total_count = 0
    
    for key, value in data.items():
        total_count += 1
        
        if value is None or value == "":
            empty_count += 1
        elif isinstance(value, str) and value.strip() == "":
            empty_count += 1
        elif isinstance(value, list) and len(value) == 0:
            empty_count += 1
        elif isinstance(value, dict):
            if check_dict_empty(value):
                empty_count += 1
    
    # Consider mostly empty if more than 50% of fields are empty
    return empty_count > (total_count * 0.5) if total_count > 0 else True


def calculate_confidence_score(
    extracted_data: Any,
    section_type: str
) -> Tuple[float, List[str]]:
    """
    Calculate confidence score for extracted section data.
    
    Args:
        extracted_data: The extracted data (can be dict, list, or any JSON type)
        section_type: Type of section
        
    Returns:
        Tuple of (confidence_score, list_of_issues)
    """
    issues = []
    confidence = 1.0
    
    try:
        # Handle different data types
        if extracted_data is None:
            issues.append("No data extracted")
            return 0.0, issues
        
        # For array-type sections
        if isinstance(extracted_data, list):
            if len(extracted_data) == 0:
                issues.append("Empty array")
                confidence -= 0.3
            else:
                # Check items in array
                empty_items = 0
                for idx, item in enumerate(extracted_data):
                    if isinstance(item, dict):
                        item_empty = check_dict_empty(item)
                        if item_empty:
                            empty_items += 1
                
                if empty_items > 0:
                    ratio = empty_items / len(extracted_data)
                    confidence -= (ratio * 0.2)
                    issues.append(f"{empty_items}/{len(extracted_data)} items are empty or incomplete")
        
        # For object-type sections
        elif isinstance(extracted_data, dict):
            # Check if dict has content
            if not extracted_data or len(extracted_data) == 0:
                issues.append("Empty object")
                confidence -= 0.3
            else:
                empty_fields = check_dict_empty(extracted_data)
                if empty_fields:
                    confidence -= 0.2
                    issues.append("Some fields are empty")
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence, issues
        
    except Exception as e:
        logger.error(f"Error calculating confidence: {e}")
        return 0.5, [f"Error in validation: {str(e)}"]


# ============================================================================
# SECTION EXTRACTION AGENT
# ============================================================================

class SectionExtractionAgent:
    """Agent to extract structured data from document sections with exact text preservation."""
    
    def __init__(self, section_schema: Dict):
        self.section_schema = section_schema
        self.storage = StorageManager()
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        return f"""You are an expert data extraction agent specializing in EXACT text extraction from documents.

Your task:
1. Analyze all provided document pages/images carefully
2. Extract ALL text EXACTLY as it appears - DO NOT paraphrase, reword, or summarize
3. Extract text embedded in images, diagrams, and charts EXACTLY as shown
4. Preserve original spelling, capitalization, punctuation, and formatting
5. Structure the data according to the provided JSON schema
6. Be thorough - don't miss any information

CRITICAL RULES FOR TEXT EXTRACTION:
✓ Copy text EXACTLY word-for-word from the document
✓ Preserve ALL original text including:
  - Exact spelling (even if misspelled)
  - Original capitalization
  - Original punctuation
  - Numbers and codes as written
  - Special characters and symbols
✗ DO NOT paraphrase or reword any text
✗ DO NOT summarize or shorten text
✗ DO NOT correct spelling or grammar
✗ DO NOT generate or infer text that isn't visible
✗ DO NOT use placeholder text like "N/A", "Unknown", "See image", etc.

RULES FOR "text" FIELDS:
- Extract the visible text EXACTLY as it appears
- If text is in an image, extract it EXACTLY
- If no text is present, use empty string: ""
- If text is partially visible, extract what you can see
- Preserve line breaks and formatting where relevant

RULES FOR "image" FIELDS:
- Provide a brief factual description of what the image shows
- Example: "Warning triangle icon", "Photo of worker wearing hard hat", "Diagram showing valve assembly"
- If no image is present, use empty string: ""
- DO NOT use generic placeholders like "image_1.png" or "N/A"

RULES FOR EMPTY OR MISSING DATA:
- Use empty string "" for text fields with no content
- Use empty array [] for array fields with no items
- DO NOT use null, "N/A", "Unknown", or any placeholder text
- If a section doesn't exist, return the minimum valid structure

Schema to follow:
{json.dumps(self.section_schema, indent=2)}

OUTPUT FORMAT:
- Return ONLY valid JSON matching the schema exactly
- No markdown code blocks (no ```json```)
- No additional text, explanations, or comments
- Start directly with {{ or [
- End with }} or ]
- CRITICAL: Escape special characters in strings:
  * Use \\" for quotes inside strings
  * Use \\\\ for backslashes
  * Use \\n for newlines
- Ensure ALL strings are properly closed with "
- Remove trailing commas before ] or }}

EXAMPLES OF CORRECT EXTRACTION:

Document shows: "WARNING: Do not operate without safety guard"
Correct: {{"text": "WARNING: Do not operate without safety guard", "image": ""}}
Wrong: {{"text": "Warning about safety guard operation", "image": ""}}

Document shows: "Step 1: Remove bolts A, B, and C"
Correct: {{"text": "Step 1: Remove bolts A, B, and C", "image": ""}}
Wrong: {{"text": "Remove the three bolts", "image": ""}}

Document shows image with text: "DANGER HIGH VOLTAGE"
Correct: {{"text": "DANGER HIGH VOLTAGE", "image": "Red warning sign with lightning bolt"}}
Wrong: {{"text": "High voltage warning", "image": "warning.png"}}
"""
    
    def extract_section(
        self,
        section_pages: List[Dict],
        section_info: Dict,
        next_section_name: str,
        document_id: str
    ) -> Dict:
        logger.info(
            f"[{document_id}] Extracting: {section_info['section_name']} "
            f"(pages {section_info['start_page']}-{section_info['end_page']})"
        )
        
        try:
            images_b64 = prepare_images_for_bedrock(section_pages)
            prompt = self._build_extraction_prompt(section_info, next_section_name)
            
            response = invoke_bedrock_multimodal(
                images=images_b64,
                prompt=prompt,
                max_tokens=MODEL_MAX_TOKENS_EXTRACTION
            )
            
            section_json = self._parse_extraction_response(response)
            
            # Calculate confidence
            confidence, issues = calculate_confidence_score(
                section_json,
                section_info['section_type']
            )
            
            # Wrap the result in a dict to add metadata
            # For array sections, the data IS the array
            # For object sections, the data IS the object
            result = {
                'section_name': section_info['section_name'],
                'page_range': [section_info['start_page'], section_info['end_page']],
                'data': section_json,  # Can be list or dict
                '_metadata': {
                    'section_type': section_info['section_type'],
                    # 'section_name': section_info['section_name'],
                    # 'page_range': [section_info['start_page'], section_info['end_page']],
                    'confidence': confidence,
                    'quality_issues': issues
                }
            }
            
            # Save intermediate result
            self.storage.save_section_json(
                document_id,
                section_info['section_name'],
                result,
                confidence
            )
            
            logger.info(
                f"[{document_id}] Extracted {section_info['section_name']} "
                f"(confidence: {confidence:.2f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"[{document_id}] Failed to extract {section_info['section_name']}: {e}"
            )
            raise
    
    def _build_extraction_prompt(self, section_info: Dict, next_section_name: str = None) -> str:
        return f"""Extract all information from this document section into JSON format.

Section: {section_info['section_name']} ({section_info['section_type']})
Pages: {section_info['start_page']} to {section_info['end_page']}

REQUIRED JSON SCHEMA:
{json.dumps(self.section_schema, indent=2)}

CRITICAL REMINDER:
- Extract ALL text EXACTLY as written below the section "{section_info['section_name']}" but before next section: "{next_section_name}" - do not paraphrase or reword
- Do not start from the top of the pages only extract information below the section until next section
- Copy text word-for-word from the section
- Preserve original order, spelling, capitalization, and punctuation
- For "image" fields: describe what the image shows (not a filename)
- Use empty string "" for missing data, never use "N/A" or placeholder text
- Return ONLY the JSON, no markdown code blocks or extra text

EXTRACTION STEPS:
1. Look at ALL pages shown ({section_info['start_page']} to {section_info['end_page']})
2. Identify all text in the section {section_info['section_name']} (including text in images) util you reach the next section: {next_section_name}
3. Extract each piece of text EXACTLY as it appears (DO NOT not duplicate the information if the section is across two pages)
4. Structure strictly according to the schema above
5. Extract text from images if image is available
6. All text which you think should go under "other_content" key please put them under "notes" key in the JSON format
6. Populate all the related values in the above JSON schema based on the section's text, if no related inforamtion available for a particular key in the schema then leave it as "".

Return the complete JSON object now (start with {{ or [, no markdown):
"""
    
    def _parse_extraction_response(self, response: str) -> Union[Dict, List]:
        """Parse LLM response to extract section data."""
        # Clean the response
        response = clean_json_response(response)
        
        try:
            parsed = json.loads(response)
            
            # If we got an array but expected an object structure, wrap it
            # This happens for array-type sections like material_risks_and_controls
            if isinstance(parsed, list):
                # The array IS the section content, return it as-is
                # The calling code will handle wrapping if needed
                return parsed
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction response: {e}")