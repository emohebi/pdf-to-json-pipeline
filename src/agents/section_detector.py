"""
Stage 1: Section Detection Agent
Identifies logical sections in PDF documents.
"""
import json
from typing import List, Dict
from strands import Agent

from config.schemas import SECTION_DEFINITIONS
from config.settings import MODEL_MAX_TOKENS_DETECTION
from src.tools import invoke_bedrock_multimodal, prepare_images_for_bedrock
from src.utils import setup_logger, StorageManager

logger = setup_logger('section_detector')


class SectionDetectionAgent:
    """Agent to identify logical sections in PDF documents."""
    
    def __init__(self):
        """Initialize section detection agent."""
        self.section_definitions = SECTION_DEFINITIONS
        self.storage = StorageManager()
        
        # Create Strands agent
        self.agent = Agent(
            system_prompt=self._build_system_prompt(),
            tools=[invoke_bedrock_multimodal],
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"
        )
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for section detection."""
        return f"""You are an expert document analyzer specializing in 
identifying logical sections in documents.

Expected section types:
{json.dumps(self.section_definitions, indent=2)}

Your task is to:
1. Analyze document pages (samples will be provided)
2. Identify where each section begins and ends
3. Classify each section by type
4. Return a structured JSON array of sections

Guidelines:
- Be precise with page boundaries
- Pay attention to:
  * Headings and titles
  * Visual separators (lines, spacing)
  * Content changes
  * Layout shifts
  * Font size and style changes
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
        Detect sections from PDF pages.
        
        Args:
            pages_data: List of page data dicts
            document_id: Unique document identifier
        
        Returns:
            List of sections with start_page, end_page, section_type
        """
        logger.info(
            f"[{document_id}] Detecting sections in {len(pages_data)} pages"
        )
        
        try:
            # Sample pages for analysis (to reduce API costs)
            sample_pages = self._sample_pages(pages_data)
            
            logger.info(
                f"[{document_id}] Analyzing {len(sample_pages)} sample pages"
            )
            
            # Prepare images
            images_b64 = prepare_images_for_bedrock(sample_pages)
            
            # Build prompt
            prompt = self._build_detection_prompt(
                pages_data, 
                sample_pages
            )
            
            # Invoke agent
            response = invoke_bedrock_multimodal(
                images=images_b64,
                prompt=prompt,
                max_tokens=MODEL_MAX_TOKENS_DETECTION
            )
            
            # Parse response
            sections = self._parse_detection_response(response)
            
            # Validate and adjust sections
            sections = self._validate_sections(sections, len(pages_data))
            
            logger.info(
                f"[{document_id}] Detected {len(sections)} sections"
            )
            
            # Save intermediate result
            self.storage.save_detection_result(document_id, sections)
            
            return sections
            
        except Exception as e:
            logger.error(
                f"[{document_id}] Section detection failed: {e}"
            )
            # Fallback to simple detection
            return self._fallback_section_detection(pages_data)
    
    def _sample_pages(self, pages_data: List[Dict]) -> List[Dict]:
        """
        Sample pages for analysis.
        Uses first page, last page, and evenly distributed pages.
        """
        total_pages = len(pages_data)
        
        if total_pages <= 10:
            # Use all pages for small documents
            return pages_data
        
        # Sample strategy:
        # - First page (likely header/title)
        # - Last page (likely references/conclusion)
        # - Every 5th page in between
        
        sample_indices = {0, total_pages - 1}  # First and last
        
        # Add every 5th page
        for i in range(4, total_pages - 1, 5):
            sample_indices.add(i)
        
        sample_indices = sorted(sample_indices)
        
        return [pages_data[i] for i in sample_indices]
    
    def _build_detection_prompt(
        self,
        all_pages: List[Dict],
        sample_pages: List[Dict]
    ) -> str:
        """Build prompt for section detection."""
        sample_page_numbers = [p['page_number'] for p in sample_pages]
        
        return f"""Analyze these document pages and identify logical sections.

Total pages in document: {len(all_pages)}
Sample pages shown: {sample_page_numbers}

Based on these samples, infer the complete section structure.

Return ONLY a JSON array with this exact structure:
[
    {{
        "section_type": "one of: {', '.join(self.section_definitions.keys())}",
        "section_name": "descriptive name",
        "start_page": number (1-indexed),
        "end_page": number (1-indexed),
        "description": "brief description",
        "confidence": number (0.0-1.0)
    }}
]

Requirements:
- Sections must not overlap
- All pages (1 to {len(all_pages)}) must be covered
- Use proper section types
- Be precise with page numbers

Return the JSON array now, no other text:
"""
    
    def _parse_detection_response(self, response: str) -> List[Dict]:
        """Parse LLM response to extract sections."""
        # Clean response
        response = response.strip()
        
        # Remove markdown code blocks if present
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
        """
        Validate and adjust section boundaries.
        Ensures no gaps or overlaps.
        """
        if not sections:
            return self._create_default_section(total_pages)
        
        # Sort by start_page
        sections = sorted(sections, key=lambda s: s['start_page'])
        
        # Ensure first section starts at page 1
        if sections[0]['start_page'] != 1:
            sections[0]['start_page'] = 1
        
        # Ensure last section ends at last page
        if sections[-1]['end_page'] != total_pages:
            sections[-1]['end_page'] = total_pages
        
        # Fix gaps and overlaps
        for i in range(len(sections) - 1):
            current_section = sections[i]
            next_section = sections[i + 1]
            
            # If there's a gap, extend current section
            if current_section['end_page'] < next_section['start_page'] - 1:
                current_section['end_page'] = next_section['start_page'] - 1
            
            # If there's overlap, adjust boundary
            if current_section['end_page'] >= next_section['start_page']:
                current_section['end_page'] = next_section['start_page'] - 1
        
        return sections
    
    def _create_default_section(self, total_pages: int) -> List[Dict]:
        """Create a single default section covering all pages."""
        return [{
            'section_type': 'general',
            'section_name': 'Document Content',
            'start_page': 1,
            'end_page': total_pages,
            'description': 'Complete document (fallback section)',
            'confidence': 0.5
        }]
    
    def _fallback_section_detection(
        self,
        pages_data: List[Dict]
    ) -> List[Dict]:
        """
        Simple fallback section detection.
        Splits document into equal-sized sections.
        """
        total_pages = len(pages_data)
        
        if total_pages <= 5:
            return self._create_default_section(total_pages)
        
        # Split into sections of ~5 pages
        pages_per_section = 5
        sections = []
        
        for i in range(0, total_pages, pages_per_section):
            start_page = i + 1
            end_page = min(i + pages_per_section, total_pages)
            
            sections.append({
                'section_type': 'general',
                'section_name': f'Section {len(sections) + 1}',
                'start_page': start_page,
                'end_page': end_page,
                'description': 'Auto-detected section',
                'confidence': 0.6
            })
        
        logger.warning(
            f"Using fallback detection: {len(sections)} sections"
        )
        
        return sections
