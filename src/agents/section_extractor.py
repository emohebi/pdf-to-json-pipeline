"""
Stage 2: Section Extraction Agent
Extracts structured data from document sections.
"""
import json
from typing import Dict, List
from strands import Agent

from config.settings import MODEL_MAX_TOKENS_EXTRACTION
from config.schemas import get_section_schema
from src.tools import invoke_bedrock_multimodal, prepare_images_for_bedrock
from src.tools.validation import calculate_confidence_score
from src.utils import setup_logger, StorageManager

logger = setup_logger('section_extractor')


class SectionExtractionAgent:
    """Agent to extract structured data from document sections."""
    
    def __init__(self, section_schema: Dict):
        self.section_schema = section_schema
        self.storage = StorageManager()
        
        self.agent = Agent(
            system_prompt=self._build_system_prompt(),
            tools=[invoke_bedrock_multimodal],
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"
        )
    
    def _build_system_prompt(self) -> str:
        return f"""You are an expert data extraction agent.

Your task:
1. Analyze all provided document pages/images
2. Extract ALL text, including text embedded in images
3. Extract data from tables, charts, and diagrams
4. Structure the data according to the provided JSON schema
5. Be thorough - don't miss any information

CRITICAL INSTRUCTIONS:
- Extract text from ALL images in the section
- Preserve tables, charts, and diagram data
- Follow the schema exactly
- Return ONLY valid JSON, no markdown or additional text
- If a field is not present, omit it or use null
- Never use placeholder text like "N/A" or "Unknown"

Schema to follow:
{json.dumps(self.section_schema, indent=2)}
"""
    
    def extract_section(
        self,
        section_pages: List[Dict],
        section_info: Dict,
        document_id: str
    ) -> Dict:
        logger.info(
            f"[{document_id}] Extracting: {section_info['section_name']} "
            f"(pages {section_info['start_page']}-{section_info['end_page']})"
        )
        
        try:
            images_b64 = prepare_images_for_bedrock(section_pages)
            prompt = self._build_extraction_prompt(section_info)
            
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
            
            # Add metadata
            section_json['_metadata'] = {
                'section_type': section_info['section_type'],
                'section_name': section_info['section_name'],
                'page_range': [section_info['start_page'], section_info['end_page']],
                'confidence': confidence,
                'quality_issues': issues
            }
            
            # Save intermediate result
            self.storage.save_section_json(
                document_id,
                section_info['section_name'],
                section_json,
                confidence
            )
            
            logger.info(
                f"[{document_id}] Extracted {section_info['section_name']} "
                f"(confidence: {confidence:.2f})"
            )
            
            return section_json
            
        except Exception as e:
            logger.error(
                f"[{document_id}] Failed to extract {section_info['section_name']}: {e}"
            )
            raise
    
    def _build_extraction_prompt(self, section_info: Dict) -> str:
        return f"""Extract all information from this document section into JSON.

Section: {section_info['section_name']} ({section_info['section_type']})
Pages: {section_info['start_page']} to {section_info['end_page']}

REQUIRED JSON SCHEMA:
{json.dumps(self.section_schema, indent=2)}

INSTRUCTIONS:
1. Analyze ALL pages in this section carefully
2. Extract ALL text including text in images, charts, tables
3. Structure according to the schema above
4. Include page numbers where data was found if relevant
5. Be thorough and accurate

Return the complete JSON object now (no markdown, just JSON):
"""
    
    def _parse_extraction_response(self, response: str) -> Dict:
        response = response.strip()
        
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction response: {e}")
            raise
