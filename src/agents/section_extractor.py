"""
Stage 2: Section Extraction Agent - Exact Text Extraction Version
Extracts structured data from document sections with EXACT text preservation.
FIXED: task_activities now uses proper hierarchical structure (sequence -> steps)
ENHANCED: Better handling of graphical elements and multiple images
MODIFIED: Maintainable items now create a separate sequence with empty sequence_no, sequence_name, and steps
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
                doc_cls_prompt = self._build_doc_classification_prompt(section_info)
                response = invoke_bedrock_multimodal(
                    images=images_b64,
                    prompt=doc_cls_prompt,
                    max_tokens=MODEL_MAX_TOKENS_EXTRACTION
                )
                doc_type = 'WIN' if "win" in response.lower() else "PMI"
                logger.info(f"doc type: {doc_type}")
                logger.info(f"response: {response}")
                prompt = self._build_task_activities_prompt(section_info, image_descriptions, doc_type)
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
            logger.error(f"RESPONSE: {response}")
            raise

    def _build_doc_classification_prompt(
        self,
        section_info: Dict,
    ) -> str:
        """Build doc classification prompt with image descriptions."""
        
        return f"""You are an expert information extraction assistant specialized in processing technical documents including Work Instructions and Preventative Maintenance Instructions.
        Your task is to classify a document correctly according to the provided TASK ACTIVITIES section.

        Section: {section_info['section_name']} ({section_info['section_type']})
        Pages: {section_info['start_page']} to {section_info['end_page']}


        FIRST STEP (DOCUMENT CLASSIFICATION):
            The TASK ACTIVITIES are from either a Preventative Maintenance Instructions and PRT Work Instructions (PMI) or Work Instructions (WIN) document.
            Carefully look at the Pages: {section_info['start_page']} to {section_info['end_page']} then decide document type: either "WIN" or "PMI" based on the following RULES:

            RULES:
            "WIN": For Work Instructions, pay special attention to:
                    - Look for "Tasks to be Done Under Running Conditions" or "Tasks to be done under Isolation"
            "PMI": For Preventative Maintenance Instructions and PRT Work Instructions, pay special attention to:
                    - Look for "Preventive Task Description" in headers/tables or document is about maintainable items
                    - Look for TEXT similar to "The following Tasks are applicable to all the maintenance items listed below"
            "Unknow": If you can not see either "WIN" or "PMI" indicators. 

        SECOND STEP:
        Once decided Return either "PMI", "WIN", or "Unknown". Do not generate any extra text.
        """
    
    def _build_task_activities_prompt(
        self,
        section_info: Dict,
        image_descriptions: Dict[str, str] = None,  # CHANGED: Dict instead of List
        doc_type: str = "PMI"
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
                
        prompt =  f"""Extract all information from this TASK ACTIVITIES section into JSON format using a FLAT structure.

            Section: {section_info['section_name']} ({section_info['section_type']})
            Pages: {section_info['start_page']} to {section_info['end_page']}
            {image_info}

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

            IMPORTANT - FLAT STRUCTURE:
            Task activities use a FLAT structure where EACH STEP is a separate object in the array.
            Sequence information must be REPEATED for each step belonging to that sequence.
        """
    
        if doc_type == "WIN":
            prompt += f"""
            FIRST STEP (SEQUENCE EXTRACTION or POPULATION):
                1. FIND SEQUENCES:
                - The squences are [BOLD] titles sometimes numerated (e.g., "1 JOB PREPARATION", "2 OPERATION") and sometimes NOT (e.g. "Pneumatic Maintenance Unit CV203"), these are sequence dividers.
                - If the sequences are numerated, the steps are likely 1.1, 1.2 ... or a., b., ...
                - If the sequences are not numerated, then the steps are likely 1., 2., ... or a., b., ... 
                
                2. LOOK FOR SUB-HEADINGS (If available):
                - The sub-headings are about execution conditions e.g., "Tasks to be done under Isolation", "Pre-Isolation Tasks", "Post-Isolation .."
                - These are NOT sequences themselves but they determine execution_condition in the next available sequence.
                - When you encounter sub-headings, infer the execution_condition for all sequences under them until you reach the next sub-heading.
                - "Tasks to be done under Isolation" → execution_condition: {{"orig_text": "Isolated", "text": "Isolated"}}
                - "Pre-Isolation Tasks" → execution_condition: {{"orig_text": "Pre-Isolation", "text": "Pre-Isolation"}}
                - "Post-Isolation Tasks" → execution_condition: {{"orig_text": "Post-Isolation", "text": "Post-Isolation"}}
                - "De-Isolation Tasks" → execution_condition: {{"orig_text": "De-Isolation", "text": "De-Isolation"}}
                - Normal conditions → execution_condition: {{"orig_text": "", "text": ""}}
                
                    2.1. SUB-ITEMS under sub-headings (e.g., a., b., c., or 1a, 1b, 1c):
                    - These usually come right after the sub-headings and before the next sequence
                    - These are steps to be assgined to the next available sequence.

                Example 1 (Numerated Sequences):
                    "Task to be done under Isolation" --> [INFER execution_condition]
                    a. step --> [SUB-ITEM]
                    b. step --> [SUB-ITEM]
                    [BOLD] 1. First TITLE --> [FIRST BOUNDARY]
                    1.1 Step one
                    1.2 Step two
                    [BOLD] 2. Second TITLE --> [SECOND BOUNDARY]
                    1.1 Step one
                    1.2 Step two

                    Result:
                    - First sequence: sequence_name: "First TITLE" and sequence_no: "1", steps a., b., 1.1, 1.2, execution_condition: "Isolated"
                    - Second sequence: sequence_name: "Second TITLE" and sequence_no: "2", steps 2.1, 2.2, execution_condition: "Isolated"

                Example 2 (NOT Numerated Sequences):
                    "Pre-Isolation Tasks (or similar sentence)" --> [INFER execution_condition]
                    a. step --> [SUB-ITEM]
                    b. step --> [SUB-ITEM]
                    [BOLD] First TITLE --> [FIRST BOUNDARY]
                    1 Step one
                    2 Step two
                    [BOLD] Second TITLE --> [SECOND BOUNDARY]
                    3 Step one
                    4 Step two

                    Result:
                    - First sequence: sequence_name: "First TITLE" and sequence_no: "", steps a., b., 1.1, 1.2, execution_condition: "Pre-Isolation"
                    - Second sequence: sequence_name: "Second TITLE" and sequence_no: "", steps 2.1, 2.2, execution_condition: "Pre-Isolation"

                3. If you see "Pre-Task Activities" title then create a sequence with sequence_name: "Pre-Task Activities" where all other fields MUST be empty.
                    Example:
                    [BOLD] PRE-TASK ACTIVITIES --> [FIRST BOUNDARY]
                    [BOLD] Title --> [SECOND BOUNDARY]
                    1. Step one
                    2. Step two

                    Result:
                    - First sequence: sequence_name: "PRE-TASK ACTIVITIES" and sequence_no is empty, NO steps (empty step fields), No other fields
                    - Second sequence: sequence_name: "Title" and sequence_no is empty, Steps 1 and 2 (ONLY in this sequence)

                4. Count how many sequences you will create before starting extraction

                """
        else:
            prompt += f"""
            FIRST STEP (SEQUENCE EXTRACTION or POPULATION):
            ⚠️ CRITICAL MAINTAINABLE ITEMS RULES - PREVENTIVE TASK DESCRIPTION SCENARIOS:
            When you see "Preventive Task Description" in headers/tables or document is about maintainable items:
            1. Scan for all "The following Tasks are applicable to all the maintenance items listed below" and [BOLD] titles, these are sequence dividers.
            2. If found, there are the following two scenarios:
            - SCENARIO 1: You see "The following Tasks are applicable to all the maintenance items listed below" alone without a [BOLD] title right AFTER it then it is a sequence
            without both sequence_no and sequence_name.
                Example 1:
                    [BOLD] Equipment Name --> [FIRST BOUNDARY]
                    [No steps here]
                    "The following Tasks..." --> [SECOND BOUNDARY]
                    1. Step one
                    2. Step two
                    [Table with maintainable items]
                    
                    Result:
                    - First sequence: sequence_name: "Equipment Name" and sequence_no is empty, NO steps (empty step fields), maintainable item: "Equipment Name"
                    - Second sequence: Steps 1 and 2 (ONLY in this sequence) and [Table with maintainable items]

                Example 2:
                    "The following Tasks..." --> [FIRST BOUNDARY]
                    [Table with maintainable items]
                    1. Step one
                    2. Step two
                    [BOLD] Equipment Name --> [SECOND BOUNDARY]
                    [No steps here]
                    [Table with maintainable items]
                    
                    Result:
                    - First sequence: both sequence_name and sequence_no are empty, Steps 1 and 2 (ONLY in this sequence) and [Table with maintainable items]
                    - Second sequence: sequence_name: "Equipment Name" and sequence_no is empty, NO steps (empty step fields) and [Table with maintainable items]

                - SCENARIO 2: You see "The following Tasks are applicable to all the maintenance items listed below" with a [BOLD] title right AFTER it then it is a sequence
            with a sequence_name as the [BOLD] title.
                Example 1:
                    [BOLD] Equipment Name --> [FIRST BOUNDARY]
                    [No steps here]
                    "The following Tasks..." --> [SECOND BOUNDARY]
                    [BOLD] Equipment Name 2
                    [Table with maintainable items]
                    1. Step one
                    2. Step two
                    "The following Tasks..." --> [THIRD BOUNDARY]
                    3. Step one
                    4. Step two
                    
                    Result:
                    - First sequence: sequence_name: "Equipment Name" and sequence_no is empty, NO steps (empty step fields), maintainable item: "Equipment Name"
                    - Second sequence: sequence_name: "Equipment Name 2" and sequence_no is empty, Steps 1 and 2 (ONLY in this sequence) and [Table with maintainable items]
                    - Third sequence: both sequence_name and sequence_no are empty, Steps 3 and 4 (ONLY in this sequence) and NO maintainable items

                - SCENARIO 3: If you see "Pre-Task Activities" title then create a sequence with sequence_name: "Pre-Task Activities" and all other sequence fields MUST be empty.
                    Example:
                    [BOLD] PRE-TASK ACTIVITIES --> [FIRST BOUNDARY]
                    [BOLD] Title --> [SECOND BOUNDARY]
                    1. Step one
                    2. Step two

                    Result:
                    - First sequence: sequence_name: "PRE-TASK ACTIVITIES" and sequence_no is empty, NO steps (empty step fields), No other fields
                    - Second sequence: sequence_name: "Title" and sequence_no is empty, Steps 1 and 2 (ONLY in this sequence)

            3. If the sequence has sequence_name but not maintainable items then DUPLICATE sequence_name to maintainable_item like example below and above:
            
            Example - CORRECT extraction:
            Document structure:
            [BOLD] TLO Hydraulic Power Pack
            [Diagram]
            "The following Tasks are applicable to all the maintenance items listed below"
            [Table with maintainable items]
            [Steps]
            
            Result for FIRST sequence (ends at "The following Tasks..."):
            {{
            "sequence_no": {{"orig_text": "", "text": ""}},
            "sequence_name": {{"orig_text": "TLO Hydraulic Power Pack", "text": "TLO Hydraulic Power Pack"}},
            "maintainable_item": [
                {{"orig_text": "TLO Hydraulic Power Pack", "text": "TLO Hydraulic Power Pack"}}  // DUPLICATED - no table with maintainable items
            ],
            "step_no": {{"orig_text": "", "text": "", "orig_image": "", "image": ""}},  // EMPTY - no steps
            "step_description": [],  // EMPTY - no steps
            "photo_diagram": [],  // Diagram should go here if needed
            "notes": [],
            "acceptable_limit": [],  // EMPTY 
            "question": [],
            "corrective_action": [],
            "execution_condition": {{"orig_text": "", "text": ""}},
            "other_content": []
            }}
            
            Example - CORRECT extraction:
            Continuing from above example, SECOND sequence (starts after "The following Tasks..."):
            {{
            "sequence_no": {{"orig_text": "", "text": ""}},
            "sequence_name": {{"orig_text": "", "text": ""}},  // EMPTY - no bold title
            "maintainable_item": [
                {{"orig_text": "Hydraulic Pump PS01 Suction Valve Open Limit Switch ZS1001", "text": "Hydraulic Pump PS01 Suction Valve Open Limit Switch ZS1001"}},
                {{"orig_text": "Hydraulic Pump PS01 Inlet Strainer Blocked Pressure Switch PSH1002", "text": "Hydraulic Pump PS01 Inlet Strainer Blocked Pressure Switch PSH1002"}},
                // ... all other items from the table
            ],
            "step_no": {{"orig_text": "1", "text": "1"}},
            "step_description": [{{"orig_text": "Visually inspect condition of field device", "text": "Visually inspect condition of field device"}}],
            // ... steps
            }}

            4. NEVER mix content, including maintainable items, across these boundaries
            5. Count how many sequences you will create before starting extraction
            
            """
        prompt += f"""
        -----------------------------------------------
        CRITIAL GENERAL RULES:
        -----------------------------------------------
        ⚠️ CRITICAL PARAGRAPH RULE:
        - If you see the text has continued to the next paragraph (or new line) then create a new text field object
        Example:
        [Paragraph 1]

        [Paragraph 2]
        Result:
        [
            {{"orig_text": "Paragraph 1", "text": "Paragraph 1"}},
            {{"orig_text": "Paragraph 2", "text": "Paragraph 2"}}
        ]

        ⚠️ CRITICAL FIELD DUPLICATION RULE:
        ALL fields with "orig_" prefix must have the SAME value as their corresponding field:
        - orig_text = text (exact same value)
        - orig_image = image (exact same value)
        - orig_seq = seq (exact same value)

        ⚠️ CRITICAL SEQUENCE ASSIGNMET:
        - In flat structure: If sequence has 2 steps, create 2 objects for that sequence with duplicated fields
        - If sequence has 0 steps, create 1 object with empty step fields

        ⚠️ CRITICAL STEP ASSIGNMENT: 
        - Steps belong ONLY to the sequence where they physically appear in the document
        - If no steps exist between the current sequence and next sequence..." → the current sequence has NO steps
        - DO NOT copy steps after next sequence into any other sequences
        - Each step appears in EXACTLY ONE sequence

        --------------------------
        CRITICAL EXTRACTION RULES:
        --------------------------
        1. DOCUMENT TYPE DETECTION:
        - Check for document type "WIN" or "PMI"
        - If found or document is a "WIN" document the use "RULES FOR "WIN" DOCUMENTS only" rules above
        - Otherwise → use use "RULES FOR "PMI" DOCUMENTS only" rules above

        2. FIELD VALUE DUPLICATION:
        - ALWAYS duplicate values: orig_text = text, orig_image = image
        - Never leave one empty if the other has value

        3. EMPTY FIELD HANDLING:
        - If both text and image are empty → field can be empty array [] or empty dict {{}}
        - Empty arrays: [] instead of [{{"orig_text": "", "text": ""}}]
        - Empty dicts: {{}} instead of {{"orig_text": "", "text": ""}}

        4. IMAGE PLACEMENT BY COLUMN:
        - Map images to fields based on table column
        - Don't put all images in one field

        5. TEXT EXTRACTION:
        - Extract TEXT from icons, don't describe them
        - Extract exact text, don't paraphrase or generate random words

        -----------------
        EXTRACTION STEPS:
        -----------------
        1. Look at ALL pages ({section_info['start_page']} to {section_info['end_page']}) and decide on the document type
        """
        if doc_type == "WIN":
            prompt += f"""
            1.1. For "WIN" documents:
            - Follow standard numbering (1, 1.1, 1.2,.. or 1., 2, 3...)
            - Check for special patterns (sub-headings, sub-items)
            - Set execution_condition based on sub-heading context
            - Assign sub-heading items to the next available sequence
            """
        else:
            prompt += f"""
            1.1. For "PMI" documents:
            - Look for "Preventive Task Description" in headers/tables
            - Look for maintainable items patterns
            - Look for bold equipment/component names

                1.1.1. CRITICAL CHECK for "The following Tasks are applicable to all the maintenance items listed below" and [BOLD] titles as sequence dividers:
                - IMMEDIATELY plan to create MULTIPLE sequences
                - DO NOT mix content across these boundaries

                1.1.2. For maintainable items document:
                PATTERN: Bold title → Content → "The following Tasks..." → Table → Steps
                RESULT: TWO sequences
                - Sequence 1: Bold title, duplicated maintainable_item, any steps BEFORE divider
                - Sequence 2: Empty name, table items, steps AFTER divider

                1.1.3. If maintainable items document detected, follow SCENARIO 1 & 2 in "PMI" documents rules:
                - Bold titles = sequences with empty sequence_no
                - Extract steps under each sequence UNTIL hitting a boundary
                - "The following Tasks..." is a HARD BOUNDARY - end current sequence
                - If maintainable table found BEFORE next sequence → populate maintainable_item
                - If no table BEFORE next sequence but have sequence_name → duplicate to maintainable_item
                - After "The following Tasks..." → start NEW sequence with its own rules
                - Content AFTER "The following Tasks..." NEVER belongs to previous sequences
            """
        prompt += f""" 
        2. Make sure:
        - Map content from each column to appropriate field
        - Duplicate all values to orig_ fields
        - Carefully check images and make sure images are populated in the correct fields in the JSON
        
        3. Clean up empty fields (remove if both text and image are empty)

        Return the complete JSON array now (start with [, no markdown):
        """
        return prompt
    
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