"""
Stage 1: Section Detection Agent - Batch Processing Version
Processes ALL pages in batches of 20, no sampling.
FIXED: Distinguishes between SECTIONS (document-level headings) and SEQUENCES (numbered task steps)
"""
import json
from typing import List, Dict
from strands import Agent

from config.schemas_docuporter import SECTION_DEFINITIONS
from config.settings import MODEL_MAX_TOKENS_DETECTION, MODEL
from src.tools import invoke_bedrock_multimodal, prepare_images_for_bedrock
from src.utils import setup_logger, StorageManager

logger = setup_logger('section_detector')


class SectionDetectionAgent:
    """Agent to identify logical sections in PDF documents."""
    
    MAX_IMAGES_PER_CALL = 20
    
    def __init__(self):
        """Initialize section detection agent."""
        self.section_definitions = SECTION_DEFINITIONS
        self.storage = StorageManager()
        
        self.agent = Agent(
            system_prompt=self._build_system_prompt(),
            tools=[invoke_bedrock_multimodal],
            model=MODEL
        )
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for section detection."""
        return f"""You are an expert document analyzer specializing in 
                identifying logical sections in documents.

                Expected section types:
                {json.dumps(self.section_definitions, indent=2)}

                Your task is to:
                1. Analyze document pages
                2. Identify where each section begins and ends
                3. Classify each section by type from the list above
                4. Extract the EXACT section name/title as it appears in the document
                5. Return a structured JSON array of sections

                CRITICAL RULES FOR SECTION NAMES:
                - section_name MUST be the EXACT text of the section heading/title from the document
                - Exception: If you see either "Pre-Task Activities" or "Post-Task Activities" sections then consider them as the "Task Activities" section
                - Examples of CORRECT extraction:
                  * Document shows: "Pre-Task Activities" → section_name: "Task Activities"
                  * Document shows: "Post-Task Activities" → section_name: "Task Activities"
                  * Document shows: "2.1 Material Risks and Controls" → section_name: "2.1 Material Risks and Controls"
                  * Document shows: "SAFETY" → section_name: "SAFETY"
                  * Document shows: "Pre-Task Activities" → section_name: "Task Activities"
                - DO NOT create descriptive names or paraphrase
                - DO NOT generate names based on content
                - Copy the section heading EXACTLY as written (preserving capitalization, punctuation, numbers, etc.)
                - Examples of CORRECT extraction:
                  * Document shows: "2.1 Material Risks and Controls" → section_name: "2.1 Material Risks and Controls"
                  * Document shows: "SAFETY" → section_name: "SAFETY"
                  * Document shows: "Task Activities" → section_name: "Task Activities"
                  * Document shows: "3. Additional PPE Required" → section_name: "3. Additional PPE Required"
                - Examples of INCORRECT extraction:
                  * Document shows: "Material Risks and Controls" → section_name: "Risk Management Section" ❌
                  * Document shows: "SAFETY" → section_name: "Safety Information" ❌
                - If a section has no visible heading/title, use "Untitled Section [page X]" where X is the start page
                
                Guidelines:
                - Be precise with page boundaries
                - Pay attention to headings, visual separators, content changes
                - Look for section numbers, titles, and headers in the document
                - If unsure about section type, use 'general'
                - Sections should not overlap
                - All pages must be covered by sections

                Output format:
                Return ONLY a JSON array, no additional text or markdown.
                """
    
    def detect_sections(
        self, 
        pages_data: List[Dict],
        document_id: str
    ) -> List[Dict]:
        """
        Detect sections from PDF pages using batch processing.
        Processes ALL pages in batches of 20.
        
        Args:
            pages_data: List of page data dicts
            document_id: Unique document identifier
        
        Returns:
            List of sections with start_page, end_page, section_type
        """
        total_pages = len(pages_data)
        logger.info(
            f"[{document_id}] Detecting sections in {total_pages} pages "
            f"(batch processing, max {self.MAX_IMAGES_PER_CALL} per batch)"
        )
        
        try:
            # if total_pages <= self.MAX_IMAGES_PER_CALL:
            #     # Single batch
            #     sections = self._detect_single_batch(pages_data, document_id)
            # else:
            #     # Multiple batches
            sections = self._detect_multi_batch(pages_data, document_id)
            
            # Save intermediate result
            self.storage.save_detection_result(document_id, sections)
            
            logger.info(f"[{document_id}] Detected {len(sections)} sections")
            return sections
            
        except Exception as e:
            logger.error(f"[{document_id}] Section detection failed: {e}")
            return None
    
    def _detect_single_batch(
        self,
        pages_data: List[Dict],
        document_id: str
    ) -> List[Dict]:
        """Process document in single batch (≤20 pages)."""
        logger.info(f"[{document_id}] Single batch: all {len(pages_data)} pages")
        
        images_b64 = prepare_images_for_bedrock(pages_data)
        prompt = self._build_detection_prompt(pages_data, pages_data)
        
        response = invoke_bedrock_multimodal(
            images=images_b64,
            prompt=prompt,
            max_tokens=MODEL_MAX_TOKENS_DETECTION
        )
        
        sections = self._parse_detection_response(response)
        sections = self._validate_sections(sections, len(pages_data))
        
        return sections
    
    def _detect_multi_batch(
        self,
        pages_data: List[Dict],
        document_id: str
    ) -> List[Dict]:
        """
        Process document in multiple batches of 20 pages.
        Analyzes ALL pages across multiple API calls.
        """
        total_pages = len(pages_data)
        num_batches = (total_pages + self.MAX_IMAGES_PER_CALL - 1) // self.MAX_IMAGES_PER_CALL
        
        logger.info(
            f"[{document_id}] Multi-batch processing: " f"{total_pages} pages -> {num_batches} batches"
        )
        
        # Process each batch and collect partial sections
        batch_sections = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.MAX_IMAGES_PER_CALL
            end_idx = min(start_idx + self.MAX_IMAGES_PER_CALL, total_pages)
            
            batch_pages = pages_data[start_idx:end_idx]
            start_page = start_idx + 1
            end_page = end_idx
            
            logger.info(
                f"[{document_id}] Processing batch {batch_idx + 1}/{num_batches}: pages {start_page}-{end_page}"
            )
            
            # Get sections for this batch
            partial_sections = self._detect_batch_sections(
                batch_pages,
                start_page,
                end_page,
                total_pages,
                document_id
            )
            
            batch_sections.append({
                'batch_idx': batch_idx,
                'start_page': start_page,
                'end_page': end_page,
                'sections': partial_sections
            })
        
        # Merge sections from all batches
        merged_sections = self._merge_batch_sections(
            batch_sections,
            total_pages,
            document_id
        )
        
        return merged_sections
    
    def _detect_batch_sections(
        self,
        batch_pages: List[Dict],
        start_page: int,
        end_page: int,
        total_pages: int,
        document_id: str
    ) -> List[Dict]:
        """Detect sections within a single batch."""
        images_b64 = prepare_images_for_bedrock(batch_pages)
        
        prompt = self._build_batch_detection_prompt(
            batch_pages,
            start_page,
            end_page,
            total_pages
        )
        
        response = invoke_bedrock_multimodal(
            images=images_b64,
            prompt=prompt,
            max_tokens=MODEL_MAX_TOKENS_DETECTION
        )
        
        sections = self._parse_detection_response(response)
        
        # Adjust page numbers to global document context
        for section in sections:
            # Ensure page numbers are within batch range
            if section['start_page'] < start_page:
                section['start_page'] = start_page
            if section['end_page'] > end_page:
                section['end_page'] = end_page
        
        return sections
    
    def _merge_batch_sections(
        self,
        batch_sections: List[Dict],
        total_pages: int,
        document_id: str
    ) -> List[Dict]:
        """
        Merge sections from multiple batches into coherent document structure.
        
        Strategy:
        1. Collect all sections from batches
        2. Merge sections that span batch boundaries (same type AND same name)
        3. Validate and fix gaps/overlaps
        """
        logger.info(f"[{document_id}] Merging {len(batch_sections)} batches")
        
        all_sections = []
        
        for batch_info in batch_sections:
            all_sections.extend(batch_info['sections'])
        
        if not all_sections:
            return None
        
        # Sort by start page
        all_sections = sorted(all_sections, key=lambda s: s['start_page'])
        
        # Merge adjacent sections of same type AND same name
        merged = []
        current_section = all_sections[0]
        
        for next_section in all_sections[1:]:
            # Check if we should merge with current section
            # FIXED: Only merge if both type AND name match exactly
            if (current_section['section_name'] == next_section['section_name'] and
                current_section['section_type'] == next_section['section_type'] and
                current_section['end_page'] >= next_section['start_page'] - 1):
                # Merge: extend current section
                logger.debug(
                    f"Merging sections: {current_section['section_name']} "
                    f"({current_section['start_page']}-{current_section['end_page']}) + "
                    f"({next_section['start_page']}-{next_section['end_page']})"
                )
                current_section['end_page'] = max(
                    current_section['end_page'],
                    next_section['end_page']
                )
                # Average confidence
                current_section['confidence'] = (
                    current_section.get('confidence', 0.8) +
                    next_section.get('confidence', 0.8)
                ) / 2
            else:
                # Don't merge: save current and move to next
                merged.append(current_section)
                current_section = next_section
        
        # Don't forget the last section
        merged.append(current_section)
        
        logger.info(
            f"[{document_id}] Merged {len(all_sections)} sections "
            f"→ {len(merged)} sections"
        )
        
        return merged
    
    def _build_batch_detection_prompt(
        self,
        batch_pages: List[Dict],
        start_page: int,
        end_page: int,
        total_pages: int
    ) -> str:
        """Build prompt for batch detection."""
        num_pages_in_batch = len(batch_pages)
        
        return f"""Analyze this PORTION of a larger document and identify DOCUMENT SECTIONS within it.

CONTEXT:
- Total document pages: {total_pages}
- This batch shows: pages {start_page} to {end_page} ({num_pages_in_batch} pages)
- You are seeing pages {start_page}-{end_page} of the complete document

IMPORTANT - UNDERSTAND THE DIFFERENCE:
1. DOCUMENT SECTIONS (what you should identify):
   - These are major document divisions with descriptive headings
   - Examples: "SAFETY", "Material Risks and Controls", "Additional PPE Required"
   - These headings are usually styled differently (bold, larger font, different color)
   - They divide the document into major logical parts

2. NUMBERED SEQUENCES (what you should IGNORE):
   - These are numbered steps or tasks WITHIN a section
   - Examples: "1 JOB PREPARATION", "2 INSPECTION", "3 POST-WORK", "G. HOUSEKEEPING"
   - These appear INSIDE the "Task Activities" section
   - DO NOT treat these as separate sections

YOUR TASK:
Identify DOCUMENT SECTIONS that START and/or END within pages {start_page}-{end_page}.

"""+"""
Return ONLY a JSON array with this exact structure:
[
    {
        "section_type": "one of: """+f"""{', '.join(self.section_definitions.keys())}"""+""",
        "section_name": "EXACT section heading text from document (not numbered sequences)",
        "start_page": number ("""+f"""{start_page}-{end_page}"""+"""),
        "end_page": number ("""+f"""{start_page}-{end_page}"""+"""),
        "confidence": number (0.0-1.0)
    }
]
"""+f"""
--------------
CRITICAL:
--------------
1- Section type MUST be one of the following:
{self.section_definitions}

2- Extract the EXACT section headings/titles as they appear in the document.
3- IGNORE numbered sequences - they are not sections.

Examples:
- Document shows section heading or page content is similar to "Material Risks and Controls" → section_name: "Material Risks and Controls" ✓
- Document shows section heading or page content is similar to "Task Activities" → section_name: "Task Activities" ✓
- Document shows numbered item "1 JOB PREPARATION" → IGNORE IT, it's not a section ✗
- Document shows numbered item "2 OPERATION" → IGNORE IT, it's not a section ✗
- Exception: If you see either "Pre-Task Activities", "Post-Task Activities", "HANDOVER TASKS", "HOUSEKEEPING TASKS" or "HOUSE KEEPING" sections or similar, then they MUST be concat to the "Task Activities" section with section_type: "task_activities".
- Example of document having "Pre-Task Activities", "Post-Task Activities", "HANDOVER TASKS", "HOUSEKEEPING TASKS" or "HOUSE KEEPING" sections or similar:
    [BOLD] Pre-Task Activities --> ["Task Activities" boundary starts here]

    [BOLD] Task Activities

    [BOLD] Post-Task Activities

    [BOLD] Next Section --> ["Task Activities" boundary ends here]
  NOTE: Do not report the "Pre-Task Activities", "Post-Task Activities", "HANDOVER TASKS", "HOUSEKEEPING TASKS" or "HOUSE KEEPING" separately, include them in the task activities with section_type: "task_activities".
- Do NOT Add "Pre-Task Activities", "Post-Task Activities", "HANDOVER TASKS", "HOUSEKEEPING TASKS" or "HOUSE KEEPING" sections to "unhandled_content"

RULES:
- Page numbers must be between {start_page} and {end_page}
- If a section starts before {start_page}, use {start_page} as start_page
- If a section ends after {end_page}, use {end_page} as end_page
- All pages {start_page}-{end_page} must be covered
- section_name MUST be the exact SECTION heading text, not numbered sequences
- Use ONLY ASCII characters (0-127) - replace unicode (- → -, • → *) with ASCII equivalents
- Numbered sequences (1 JOB PREP, 2 OPERATION, etc.) are NOT sections

Return the JSON array now, no other text:
"""
    
    def _parse_detection_response(self, response: str) -> List[Dict]:
        """Parse LLM response to extract sections."""
        response = response.strip()
        
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        
        response = response.strip()
        
        try:
            sections = json.loads(response)
            
            if not isinstance(sections, list):
                raise ValueError("Response is not a JSON array")
            
            return sections
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse detection response: {e}")
            logger.debug(f"Response was: {response[:500]}")
            raise
    
    def _validate_sections(
        self,
        sections: List[Dict],
        total_pages: int
    ) -> List[Dict]:
        """Validate and adjust section boundaries."""
        if not sections:
            return None
        
        sections = sorted(sections, key=lambda s: s['start_page'])
        
        # Ensure first section starts at page 1
        if sections[0]['start_page'] != 1:
            logger.warning(
                f"Adjusting first section from page {sections[0]['start_page']} to page 1"
            )
            sections[0]['start_page'] = 1
        
        # Ensure last section ends at last page
        if sections[-1]['end_page'] != total_pages:
            logger.warning(
                f"Adjusting last section from page {sections[-1]['end_page']} "
                f"to page {total_pages}"
            )
            sections[-1]['end_page'] = total_pages
        
        # Fix gaps and overlaps
        for i in range(len(sections) - 1):
            current_section = sections[i]
            next_section = sections[i + 1]
            
            # Fix gap
            if current_section['end_page'] < next_section['start_page'] - 1:
                logger.warning(
                    f"Gap detected between sections, extending "
                    f"{current_section['section_name']}"
                )
                current_section['end_page'] = next_section['start_page'] - 1
            
            # Fix overlap
            if current_section['end_page'] >= next_section['start_page']:
                logger.warning(
                    f"Overlap detected, adjusting boundary between "
                    f"{current_section['section_name']} and "
                    f"{next_section['section_name']}"
                )
                current_section['end_page'] = next_section['start_page'] - 1
        
        return sections
    
    def _create_default_section(self, total_pages: int) -> List[Dict]:
        """Create a single default section covering all pages."""
        return [{
            'section_type': 'general',
            'section_name': f'Untitled Section [page 1]',
            'start_page': 1,
            'end_page': total_pages,
            'description': 'Complete document (fallback section)',
            'confidence': 0.5
        }]