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
from config.schemas_docuporter import get_section_schema
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
- Empty array of objects: []
- Empty dictionary: {{}}
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
    \\nCRITICAL RULES FOR IMAGE PLACEMENT IN APPROPRIATE FIELDS:
    
    STEP TABLE COLUMN MAPPING:
    When you see a table with columns for steps, map images to the correct fields based on the column they appear in:
    
    1. STEP DESCRIPTION column/cell:
       - Main step text and instructions go to step_description field
       - Small inline icons/symbols within step text: add as separate entries in step_description
       - Extract embedded text from icons (e.g., "HOLD POINT"), don't describe them
    
    2. PHOTO/DIAGRAM column/cell:
       - Images in photo/diagram column → photo_diagram field
       - These are typically equipment photos, assembly diagrams, reference images
       - Each image as separate list entry
    
    3. NOTES column/cell:
       - Text and images in notes column → notes field
       - Including graphical notes elements
       - Multiple notes = multiple list entries
    
    4. ACCEPTABLE LIMIT column/cell:
       - Content from this column → acceptable_limit field
    
    5. QUESTION column/cell:
       - Content from this column → question field
    
    6. CORRECTIVE ACTION column/cell:
       - Content from this column → corrective_action field
    
    IMPORTANT: Place content in the field that matches its TABLE COLUMN, not based on content type!
"""
        
        return f"""Extract all information from this TASK ACTIVITIES section into JSON format using a FLAT structure.

    Section: {section_info['section_name']} ({section_info['section_type']})
    Pages: {section_info['start_page']} to {section_info['end_page']}
    {image_info}

    IMPORTANT - FLAT STRUCTURE:
    Task activities use a FLAT structure where EACH STEP is a separate object in the array.
    Sequence information must be REPEATED for each step belonging to that sequence.

    CRITICAL SEQUENCE NUMBERING RULES:
    1. MAIN SEQUENCES (e.g., "1 JOB PREPARATION", "2 OPERATION"):
       - sequence_no: {{"orig_text": "1", "text": "1"}} (just the number, SAME value in both fields)
       - sequence_name: {{"orig_text": "JOB PREPARATION", "text": "JOB PREPARATION"}} (SAME value in both fields)
    
    2. SUB-HEADINGS (e.g., "Tasks to be done under Isolation", "Pre-Isolation Tasks"):
       - These are NOT sequences themselves
       - sequence_no: {{"orig_text": "", "text": ""}} (EMPTY for sub-headings)
       - sequence_name: {{"orig_text": "Tasks to be done under Isolation", "text": "Tasks to be done under Isolation"}}
       - Sub-headings determine execution_condition
    
    3. SUB-ITEMS under sub-headings (e.g., a., b., c., or 1a, 1b, 1c):
       - These ARE the actual sequences
       - Increment sequence number for these items
       - sequence_no: {{"orig_text": "2", "text": "2"}} (next sequence number)
       - sequence_name: {{"orig_text": "Item description", "text": "Item description"}}
    
    4. PREVENTIVE TASK DESCRIPTION SPECIAL RULE:
       - If you see "Preventive Task Description" in the document:
       - The BOLD text immediately under it becomes the sequence_name
       - sequence_no: {{"orig_text": "", "text": ""}} (EMPTY)
       - Items under it (1, 2, 3, etc.) are the steps for this sequence
       - Example:
         * Document shows: "Preventive Task Description" followed by bold "Conveyor Belt Inspection"
         * sequence_no: {{"orig_text": "", "text": ""}}
         * sequence_name: {{"orig_text": "Conveyor Belt Inspection", "text": "Conveyor Belt Inspection"}}
         * Then items 1, 2, 3 below are steps

    CRITICAL MAINTAINABLE ITEMS RULES:
    1. DEFAULT BEHAVIOR - DO NOT POPULATE maintainable_item:
       - By default, leave maintainable_item as empty array: []
       - Only populate when you see the specific pattern below
    
    2. SPECIAL MAINTAINABLE ITEMS TABLE:
       - ONLY populate maintainable_item when you see text like:
         "The following Task or Tasks is applicable to all the maintenance items listed below"
         (or similar wording about tasks applying to listed items)
       - When this pattern appears:
         * sequence_name: {{"orig_text": "", "text": ""}} (EMPTY)
         * Extract all items from the table/list that follows
         * Each item becomes an entry in maintainable_item array
       - Example:
         * Document shows: "The following Task is applicable to all the maintenance items listed below:"
         * Followed by table/list: Feed Chute CV110, Discharge Chute CV110, etc.
         * Result:
           sequence_name: {{"orig_text": "", "text": ""}}
           maintainable_item: [
             {{"orig_text": "Feed Chute CV110", "text": "Feed Chute CV110"}},
             {{"orig_text": "Discharge Chute CV110", "text": "Discharge Chute CV110"}}
           ]
    
    3. WITHOUT THE SPECIAL TEXT:
       - If you don't see text about "applicable to all maintenance items" or similar
       - Keep maintainable_item as empty array: []
       - Do NOT populate it with random equipment names
    
    EXECUTION CONDITION BASED ON SUB-HEADINGS:
    When you encounter sub-headings, set execution_condition for all steps under them:
    - "Tasks to be done under Isolation" → execution_condition: {{"orig_text": "Isolated", "text": "Isolated"}}
    - "Pre-Isolation Tasks" → execution_condition: {{"orig_text": "Pre-Isolation", "text": "Pre-Isolation"}}
    - "Post-Isolation Tasks" → execution_condition: {{"orig_text": "Post-Isolation", "text": "Post-Isolation"}}
    - Normal conditions → execution_condition: {{"orig_text": "", "text": ""}}

    CRITICAL FIELD DUPLICATION RULE:
    ALL fields with "orig_" prefix must have the SAME value as their corresponding field:
    - orig_text = text (exact same value)
    - orig_image = image (exact same value)
    - orig_seq = seq (exact same value)

REQUIRED JSON STRUCTURE (FLAT):
[
  {{
    "equipment_asset": {{"orig_text": "Equipment name", "text": "Equipment name"}},
    "sequence_no": {{"orig_text": "1", "text": "1"}},
    "sequence_name": {{"orig_text": "JOB PREPARATION", "text": "JOB PREPARATION"}},
    "maintainable_item": [],  // Usually empty unless special table present
    "lmi": [{{"orig_text": "LMI info", "text": "LMI info"}}],
    "step_no": {{"orig_text": "1.1", "orig_image": "", "text": "1.1", "image": ""}},
    "step_description": [{{"orig_text": "Step instructions", "orig_image": "", "text": "Step instructions", "image": ""}}],
    "photo_diagram": [{{"orig_text": "", "orig_image": "path", "text": "", "image": "path"}}],
    "notes": [{{"orig_text": "Note text", "orig_image": "", "text": "Note text", "image": ""}}],
    "acceptable_limit": [{{"orig_text": "45 Nm", "orig_image": "", "text": "45 Nm", "image": ""}}],
    "question": [{{"orig_text": "Is it ok?", "orig_image": "", "text": "Is it ok?", "image": ""}}],
    "corrective_action": [{{"orig_text": "Fix it", "orig_image": "", "text": "Fix it", "image": ""}}],
    "execution_condition": {{"orig_text": "", "text": ""}},
    "other_content": [{{"orig_text": "", "orig_image": "", "text": "", "image": ""}}]
  }}
]

CRITICAL EXTRACTION RULES:

1. SEQUENCE HIERARCHY:
   - Main numbered items (1, 2, 3) are sequences (unless under Preventive Task Description)
   - Sub-headings are NOT sequences (empty sequence_no)
   - Sub-items (a, b, c) under sub-headings ARE sequences (get next number)
   - Preventive Task Description: bold text after it = sequence_name with empty sequence_no

2. MAINTAINABLE ITEMS:
   - Default: Keep empty array []
   - Only populate when seeing "applicable to all maintenance items" text pattern
   - When populated, set sequence_name to empty string

3. FIELD VALUE DUPLICATION:
   - ALWAYS duplicate values: orig_text = text, orig_image = image
   - Never leave one empty if the other has value

4. EMPTY FIELD HANDLING:
   - If both text and image are empty → field can be empty array [] or empty dict {{}}
   - Empty arrays: [] instead of [{{"orig_text": "", "text": ""}}]
   - Empty dicts: {{}} instead of {{"orig_text": "", "text": ""}}

5. IMAGE PLACEMENT BY COLUMN:
   - Map images to fields based on table column
   - Don't put all images in one field

6. TEXT EXTRACTION:
   - Extract TEXT from icons, don't describe them
   - Extract exact text, don't paraphrase

EXTRACTION STEPS:
1. Look at ALL pages ({section_info['start_page']} to {section_info['end_page']})
2. Check for "Preventive Task Description" pattern
3. Check for "applicable to all maintenance items" pattern
4. Identify the sequence structure (main items, sub-headings, sub-items)
5. For each step:
   - Determine sequence numbering based on patterns
   - Handle maintainable_item according to rules
   - Map content from each column to appropriate field
   - Duplicate all values to orig_ fields
   - Set execution_condition based on sub-heading context
6. Clean up empty fields (remove if both text and image are empty)

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
    \\nCRITICAL RULES FOR IMAGE PLACEMENT:
    1. IDENTIFY TABLE STRUCTURE:
       - Look for table columns/cells in the document
       - Map images to fields based on their COLUMN POSITION, not content type
    
    2. COLUMN-TO-FIELD MAPPING:
       - Images in a specific column go to the corresponding field
       - If document has columns like "Risk | Description | Controls"
         then images in Risk column → risk field, etc.
    
    3. TEXT EXTRACTION:
       - Extract TEXT FROM INSIDE icons/images (e.g., "HOLD POINT", "QA")
       - Do NOT describe what the icon looks like
       - If no text is embedded, use empty string ""
    
    4. FORMAT:
       - {"orig_text": "caption", "orig_image": "path", "text": "caption", "image": "path"}
       - orig_text = text (SAME value)
       - orig_image = image (SAME value)
    """
        
        return f"""Extract all information from this document section into JSON format.

    Section: {section_info['section_name']} ({section_info['section_type']})
    Pages: {section_info['start_page']} to {section_info['end_page']}
    {image_info}

    REQUIRED JSON SCHEMA:
    {json.dumps(self.section_schema, indent=2)}

    CRITICAL DOCUPORTER FORMAT RULES:
    1. FIELD DUPLICATION:
       - Every "text" field has a corresponding "orig_text" field with SAME value
       - Every "image" field has a corresponding "orig_image" field with SAME value
       - Every "seq" field has a corresponding "orig_seq" field with SAME value
       - NEVER leave one empty if the other has value
    
    2. SCHEMA PRESERVATION:
       - Include EVERY KEY shown in the schema
       - NEVER add, remove, or rename keys
       - Preserve the EXACT structure
    
    3. EMPTY FIELD HANDLING:
       - If both text and image are empty → return empty array [] or empty dict {{}}
       - [{{"orig_text": "", "text": ""}}] → []
       - {{"orig_text": "", "text": ""}} → {{}}
       - But if either has value, keep the structure

    DEFAULT VALUES FOR FIELDS WITH CONTENT:
    - String fields with value: {{"orig_text": "value", "text": "value"}}
    - Image fields with path: {{"orig_image": "path", "image": "path"}}
    - Empty fields: {{}} or []

    CRITICAL IMAGE PLACEMENT RULES:
    - If document has a table structure, place images in fields matching their table columns
    - Don't put all images in one field - distribute based on column context
    - Extract TEXT FROM INSIDE icons/images, do NOT describe them
    - Multiple images in same column = multiple list entries in that field

    CRITICAL EXTRACTION RULES:
    - Extract ALL text EXACTLY as written below the section "{section_info['section_name']}" but before next section: "{next_section_name}"
    - Do not paraphrase or reword
    - Copy text word-for-word from the section
    - Preserve original spelling, capitalization, and punctuation
    - For images: Use the paths provided above
    - Place images in the field that matches their TABLE COLUMN position

    EXTRACTION STEPS:
    1. Look at ALL pages shown ({section_info['start_page']} to {section_info['end_page']})
    2. Identify the table structure and column headers (if any)
    3. Extract all text from section {section_info['section_name']} until next section: {next_section_name}
    4. For each piece of content:
       - Extract text EXACTLY as it appears
       - Duplicate to orig_text field
       - Map images to appropriate fields by column
       - Duplicate image paths to orig_image fields
    5. Structure according to the schema
    6. Clean up empty fields (if both text and image empty, use [] or {{}})

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