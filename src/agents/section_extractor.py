"""
Stage 2: Section Extraction Agent - Exact Text Extraction Version
Extracts structured data from document sections with EXACT text preservation.
FIXED: task_activities now uses proper hierarchical structure (sequence -> steps)
ENHANCED: Better handling of graphical elements and multiple images
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

CRITICAL RULES FOR SCHEMA PRESERVATION:
✓ ALWAYS include EVERY key from the provided schema
✓ NEVER omit fields, even if they are empty
✓ NEVER add fields that aren't in the schema
✓ NEVER rename or modify field names
✓ Use appropriate empty values for missing data:
  - String fields: ""
  - Object fields: include all subfields with empty values

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
- Extract text embedded in images
- Example: Document shows image with text: "DANGER HIGH VOLTAGE" extract it exactly
- If no image is present, use empty string: ""

RULES FOR EMPTY OR MISSING DATA:
- Use empty string "" for text fields with no content
- Empty object with "text" property: {{"text": ""}}
- Empty object with "text" and "image" properties: {{"text": "", "image": ""}}
- Empty array of objects with "text" property: [{{"text": ""}}]
- Empty array of objects with "text" and "image" properties: [{{"text": "", "image": ""}}]
- But the field MUST STILL BE PRESENT
- DO NOT use null, "N/A", "Unknown", or any placeholder text
- If a section doesn't exist, return the minimum valid structure

Schema to follow:
{json.dumps(self.section_schema, indent=2)}

OUTPUT FORMAT:
- Return ONLY valid JSON matching the schema exactly
- Include EVERY field from the schema
- Use empty values, never omit fields
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
Correct: {{"text": "DANGER HIGH VOLTAGE", "image": ""}}
Wrong: {{"text": "High voltage warning", "image": ""}}
"""
    
    def extract_section(
        self,
        section_pages: List[Dict],
        section_info: Dict,
        next_section_name: str,
        document_id: str,
        image_descriptions: Dict[str, str] = None  # CHANGED: Now a dict of description->path
    ) -> Dict:
        """
        Extract section with image information.
        
        Args:
            section_pages: List of page data dicts
            section_info: Section metadata
            next_section_name: Name of next section
            document_id: Document ID
            section_images: List of images in this section (NEW)
        """
        logger.info(
            f"[{document_id}] Extracting: {section_info['section_name']} "
            f"(pages {section_info['start_page']}-{section_info['end_page']})"
        )
        
        
        try:
            images_b64 = prepare_images_for_bedrock(section_pages)
            
            # Build prompt with image information
            if section_info['section_type'] == 'task_activities':
                prompt = self._build_task_activities_prompt(section_info, next_section_name, image_descriptions)
            else:
                prompt = self._build_extraction_prompt(section_info, next_section_name, image_descriptions)

            
            response = invoke_bedrock_multimodal(
                images=images_b64,
                prompt=prompt,
                max_tokens=MODEL_MAX_TOKENS_EXTRACTION
            )
            
            section_json = self._parse_extraction_response(response)

            # section_json = self._validate_and_fix_schema(section_json, self.section_schema)
            
            # Rest of the method remains the same...
            confidence, issues = calculate_confidence_score(
                section_json,
                section_info['section_type']
            )
            
            result = {
                'section_name': section_info['section_name'],
                'page_range': [section_info['start_page'], section_info['end_page']],
                'data': section_json,
                '_metadata': {
                    'section_type': section_info['section_type'],
                    'confidence': confidence,
                    'quality_issues': issues
                }
            }
            
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
    
    def _build_task_activities_prompt(
        self,
        section_info: Dict,
        next_section_name: str = None,
        image_descriptions: Dict[str, str] = None  # CHANGED: Dict instead of List
    ) -> str:
        """Build task activities prompt with image descriptions."""
        
        # Format image descriptions
        image_info = ""
        if image_descriptions:
            image_info = "\\n\\nIMAGES AVAILABLE IN THIS SECTION:\\n"
            for desc, path in image_descriptions.items():
                image_info += f"- {desc}: {path}\\n"
            image_info += """
    \\nCRITICAL RULES FOR IMAGES AND GRAPHICAL ELEMENTS:
    1. STEP DESCRIPTION field:
       - Contains the main textual instructions for the step
       - Small inline icons/symbols that are part of the text flow stay here
       - For graphical elements in step description: extract any embedded text (not describe the icon)
       - If multiple small icons exist, create separate list entries for each
    
    2. PHOTO/DIAGRAM field:
       - ONLY for actual photos, diagrams, or larger illustrations
       - NOT for small icons or symbols that are part of step text
       - Typically shows equipment, assembly diagrams, or reference photos
       - Each distinct photo/diagram should be a separate list entry
    
    3. TEXT EXTRACTION from graphical elements:
       - Extract any TEXT visible INSIDE icons/images (e.g., "HOLD POINT" text in a flame icon)
       - Do NOT describe what the icon looks like
       - If no text is embedded in the graphical element, use empty string ""
    
    4. MULTIPLE ELEMENTS:
       - If multiple icons/images exist, create SEPARATE list entries for each
       - Do not combine multiple elements into one entry
       
    5. NOTES field:
       - Extract ALL notes, including those shown as graphical text elements
       - If "Notes" appears as a graphical element, extract the text content
       - Multiple notes = multiple list entries
"""
        
        return f"""Extract all information from this TASK ACTIVITIES section into JSON format.

    Section: {section_info['section_name']} ({section_info['section_type']})
    Pages: {section_info['start_page']} to {section_info['end_page']}
    {image_info}

    IMPORTANT - HIERARCHICAL STRUCTURE:
    Task activities are organized as SEQUENCES containing STEPS:
    - Each SEQUENCE has: sequence_no, sequence_name, equipment_asset, maintainable_item, lmi
    - Each SEQUENCE contains multiple STEPS
    - Each STEP has: step_no, step_description, photo_diagram, notes, acceptable_limit, question, corrective_action, execution_condition, other_content.

REQUIRED JSON STRUCTURE:
[
  {{
    "sequence_no": {{"text": "Sequence number"}},
    "sequence_name": {{"text": "Sequence title"}},
    "equipment_asset": {{"text": "Equipment/asset name if any"}},
    "maintainable_item": [{{"text": "Item description"}}],
    "lmi": [{{"text": "LMI information if any"}}],
    "steps": [
      {{
        "step_no": {{"text": "Step number", "image": ""}},
        "step_description": [{{"text": "Step instructions", "image": ""}}],
        "photo_diagram": [{{"text": "Caption or label", "image": ""}}],
        "notes": [{{"text": "Note text", "image": ""}}],
        "acceptable_limit": [{{"text": "Limit specification", "image": ""}}],
        "question": [{{"text": "Question text", "image": ""}}],
        "corrective_action": [{{"text": "Action to take", "image": ""}}],
        "execution_condition": {{"text": "Condition for execution", "image": ""}},
        "other_content": [{{"text": "Any other relevant content", "image": ""}}]
      }}
    ]
  }}
]

CRITICAL RULES FOR FIELD POPULATION:

1. step_description field:
   - Main text instructions for the step go here
   - Small inline icons/symbols that appear within step text: include each as separate list entry
   - For each icon: extract EMBEDDED TEXT only (e.g., "HOLD POINT" if visible in icon), not description
   - Multiple icons = multiple list entries [{{"text": "main instruction", "image": ""}}, {{"text": "HOLD POINT", "image": "path1"}}, {{"text": "QA", "image": "path2"}}]
   - Do NOT describe icons (wrong: "flame icon", correct: extract text inside icon or "")

2. photo_diagram field:
   - ONLY actual photographs, technical diagrams, or illustrations
   - NOT small icons or symbols from step description
   - Each photo/diagram as separate entry: [{{"text": "diagram caption", "image": "path1"}}, {{"text": "photo caption", "image": "path2"}}]
   - Leave empty [] if no actual photos/diagrams for the step

3. notes field:
   - Extract ALL notes, including those shown as graphical text elements
   - If "Notes" appears as a graphical element, extract the text content from inside it
   - Multiple notes = multiple list entries
   - Do NOT skip notes even if they appear as images

4. Text extraction from graphical elements:
   - Extract TEXT EMBEDDED in the graphical element (what's written inside it)
   - DO NOT describe the graphical element (wrong: "flame icon", correct: "HOLD POINT" if that text is in the icon)
   - If no text inside the graphical element, use empty string ""

5. Multiple elements handling:
   - ALWAYS create separate list entries for distinct elements
   - Do not combine multiple icons/images/text blocks into one entry
   - Each icon, each image, each text block = separate list entry

6. PRESERVATION RULES:
   - EVERY STEP MUST HAVE ALL FIELDS listed in the schema
   - NEVER omit fields even if they are empty in the document
   - Empty fields should use appropriate empty values as shown above
   - But the field MUST STILL BE PRESENT

EXTRACTION STEPS:
1. Look at ALL pages shown ({section_info['start_page']} to {section_info['end_page']})
2. Identify sequences in the section
3. For each step, carefully distinguish:
   - Main text instructions (step_description)
   - Inline icons/symbols with embedded text (step_description as separate entries)
   - Actual photos/diagrams (photo_diagram)
   - Notes including graphical notes (extract text, not describe)
4. Extract embedded text from graphical elements, not descriptions
5. Create separate list entries for multiple elements
6. Ensure ALL fields are present for each step

Return the complete JSON array now (start with [, no markdown):
"""
    
    def _build_extraction_prompt(
        self, 
        section_info: Dict, 
        next_section_name: str = None, 
        image_descriptions: Dict[str, str] = None  # CHANGED: Dict instead of List
    ) -> str:
        """Build extraction prompt with image descriptions."""
        
        # Format image descriptions if available
        image_info = ""
        if image_descriptions:
            image_info = "\\n\\nIMAGES AVAILABLE IN THIS SECTION:\\n"
            for desc, path in image_descriptions.items():
                image_info += f"- {desc}: {path}\\n"
            image_info += """
    \\nWHEN YOU SEE AN IMAGE IN THE DOCUMENT:
    1. Identify what the image shows
    2. Find the matching description from the list above
    3. Use the corresponding path in the "image" field
    4. Format: {"text": "any caption or embedded text", "image": "Media/xxx/pagex_imgx.png"}
    5. Extract TEXT FROM INSIDE the image/icon, not describe it
    """
        
        return f"""Extract all information from this document section into JSON format.

    Section: {section_info['section_name']} ({section_info['section_type']})
    Pages: {section_info['start_page']} to {section_info['end_page']}
    {image_info}

    REQUIRED JSON SCHEMA:
    {json.dumps(self.section_schema, indent=2)}

    CRITICAL SCHEMA PRESERVATION RULES:
    1. YOU MUST INCLUDE EVERY KEY shown in the schema above
    2. YOU MUST PRESERVE THE EXACT STRUCTURE - do not add, remove, or rename ANY keys
    3. For EVERY field in the schema:
    - If you find matching content → populate with the EXACT text
    - If no matching content exists → use the default empty value
    4. NEVER SKIP OR OMIT ANY KEY from the schema, even if empty

    DEFAULT VALUES FOR EMPTY FIELDS:
    - For string fields → use empty string: ""
    - Empty object with "text" property: {{"text": ""}}
    - Empty object with "text" and "image" properties: {{"text": "", "image": ""}}
    - Empty array of objects with "text" property: [{{"text": ""}}]
    - Empty array of objects with "text" and "image" properties: [{{"text": "", "image": ""}}]
    - But the field MUST STILL BE PRESENT
    - NEVER use null, undefined, or omit the field

    CRITICAL REMINDER:
    - Extract ALL text EXACTLY as written below the section "{section_info['section_name']}" but before next section: "{next_section_name}" - do not paraphrase or reword
    - For "image" fields: Use the image paths provided above when you see corresponding images
    - Extract TEXT FROM INSIDE icons/images, do NOT describe them
    - Do not start from the top of the pages only extract information below the section until next section
    - Copy text word-for-word from the section
    - Preserve original order, spelling, capitalization, and punctuation
    - Use empty string "" for missing data, never use "N/A" or placeholder text
    - Return ONLY the JSON, no markdown code blocks or extra text

    EXTRACTION STEPS:
    1. Look at ALL pages shown ({section_info['start_page']} to {section_info['end_page']})
    2. Identify all text in the section {section_info['section_name']} (including text in images) until you reach the next section: {next_section_name}
    3. When you see an image/icon/diagram in the page, use the corresponding image path from the list above
    4. Extract each piece of text EXACTLY as it appears (DO NOT duplicate the information if the section is across two pages)
    5. Structure strictly according to the schema above
    6. Extract text from images if image is available (text INSIDE the image, not description)
    7. All text which you think should go under "other_content" key please put them under "notes" key in the JSON format
    8. Populate all the related values in the above JSON schema based on the section's text, if no related information available for a particular key in the schema then leave it as "".
    9. Populate image fields with the actual paths when images are present
    10. For multiple images/icons, create separate list entries for each

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
            raise

    def _validate_and_fix_schema(self, extracted_json: Any, expected_schema: Any) -> Any:
        """
        Validate extracted JSON against schema and add missing keys.
        This is a safety net in case the LLM still omits keys.
        """
        if isinstance(expected_schema, list) and len(expected_schema) > 0:
            # Schema is an array, get the template item
            template = expected_schema[0]
            
            if isinstance(extracted_json, list):
                # Fix each item in the array
                fixed_items = []
                for item in extracted_json:
                    if isinstance(item, dict) and isinstance(template, dict):
                        fixed_item = self._fix_dict_schema(item, template)
                        fixed_items.append(fixed_item)
                    else:
                        fixed_items.append(item)
                return fixed_items
            else:
                # Expected array but got something else, return with template structure
                return []
        
        elif isinstance(expected_schema, dict) and isinstance(extracted_json, dict):
            return self._fix_dict_schema(extracted_json, expected_schema)
        
        return extracted_json

    def _fix_dict_schema(self, data: Dict, template: Dict) -> Dict:
        """
        Ensure all keys from template exist in data.
        """
        fixed = {}
        
        for key, template_value in template.items():
            if key in data:
                # Key exists, recursively fix if needed
                if isinstance(template_value, dict) and isinstance(data[key], dict):
                    fixed[key] = self._fix_dict_schema(data[key], template_value)
                elif isinstance(template_value, list) and len(template_value) > 0:
                    if isinstance(data[key], list):
                        # Fix each item in the list
                        fixed_items = []
                        item_template = template_value[0]
                        for item in data[key]:
                            if isinstance(item, dict) and isinstance(item_template, dict):
                                fixed_items.append(self._fix_dict_schema(item, item_template))
                            else:
                                fixed_items.append(item)
                        fixed[key] = fixed_items
                    else:
                        fixed[key] = []  # Empty array if wrong type
                else:
                    fixed[key] = data[key]
            else:
                # Key missing, add with empty value based on template
                if isinstance(template_value, dict):
                    # For dict, recursively create empty structure
                    if "text" in template_value and "image" in template_value:
                        fixed[key] = {"text": "", "image": ""}
                    else:
                        fixed[key] = self._fix_dict_schema({}, template_value)
                elif isinstance(template_value, list):
                    fixed[key] = []  # Empty array
                elif isinstance(template_value, str):
                    fixed[key] = ""  # Empty string
                else:
                    fixed[key] = template_value  # Use template value
        
        return fixed