#!/usr/bin/env python
# coding: utf-8

"""
Document Scraping Solution - Restructured
Parses and extracts information from PDF/DOCX documents using AWS Bedrock
"""

import os
import json
import yaml
import logging
import re
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import base64

import mammoth
from bs4 import BeautifulSoup
import pymupdf
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as boto_Config


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

class Config:
    """Application configuration"""
    
    # Working directories
    CURRENT_PATH = os.path.join(os.getcwd(), "old_src")
    INPUT_DIR = os.path.join(CURRENT_PATH, "input")
    OUTPUT_DIR = os.path.join(CURRENT_PATH, "output")
    TEMPLATE_DIR = os.path.join(CURRENT_PATH, "templates")
    PROMPTS_DIR = os.path.join(CURRENT_PATH, "prompts")
    
    # AWS Bedrock
    BEDROCK_NAMESPACE = "swscraping"
    MODEL_ID_37 = "apac.anthropic.claude-3-7-sonnet-20250219-v1:0"
    MODEL_ID_40 = "apac.anthropic.claude-sonnet-4-20250514-v1:0"
    
    # Section exclusions
    SECTION_EXCLUSION_LIST = [
        "FEEDBACK",
        "FEEDBACK (To support content improvement)",
        "ADDITIONAL WORK IDENTIFIED",
        "PMI FEEDBACK",
        "PMI FEEDBACK (To support content improvement)",
        "WIN FEEDBACK",
        "WIN FEEDBACK (To support content improvement)",
        "WORK REQUEST SIGN-OFF",
        "WORK REQUEST SIGN-OFF (Person(s) Who Completed Work)",
        "WORK ORDER SIGN-OFF",
        "WORK ORDER SIGN-OFF (Person(s) Who Completed Work)",
        "WORK ORDER SIGN OFF (Person(s) Who Completed Work)",
        "PARTS MANAGEMENT",
        "SIGN-OFF WORK COMPLETED",
        "SIGN – OFF WORK COMPLETED",
        "QUALITY OF THIS DOCUMENT",
        "COMPLETION OF WORK",
        "COMPLETION INFORMATION",
        "TASK DESCRIPTION",
        "TASK ACTIVITIES",
        "WORK DESCRIPTION",
        "SIGN – OFF WORK COMPLETED (Person(s) Who Completed Work)",
        "WORK DESCRIPTION",
        "ACTIVITY EXECUTION",
        "FEEDBACK (To support improvement)",
        "SCOPE",
        "Record Keeping Responsibilities"
    ]
    
    # Section category mapping
    SECTION_CATEGORY_MAPPING = {
        "safety": {
            "headings": ["SAFETY"],
            "json_property": "safety"
        },
        "material_risks": {
            "headings": [
                "MATERIAL RISKS",
                "MATERIAL RISKS AND CONTROLS", 
                "MATERIAL RISKS AND MAJOR HAZARDS",
                "MATERIAL RISKS, HAZARDS & CONTROLS",
                "CATASTROPHIC RISKS",
                "CATASTROPHIC RISKS AND MAJOR HAZARDS",
                "ISOLATIONS, PERMITS AND SPECIAL CONTROLS REQUIRED"
            ],
            "json_property": "material_risks_and_controls"
        },
        "controls": {
            "headings": ["ADDITIONAL CONTROLS REQUIRED"],
            "json_property": "additional_controls_required"
        },
        "ppe": {
            "headings": [
                "ADDITIONAL PPE REQUIRED", 
                "ADDITIONAL PPE REQUIRED (Adhere to signage requesting specific PPE)", 
                "PPE REQUIRED (Over and above Standard PPE)"
            ],
            "json_property": "additional_ppe_required"
        },
        "competencies": {
            "headings": [
                "SPECIFIC COMPETENCIES, KNOWLEDGE AND SKILLS REQUIRED",
                "RESOURCES, SPECIFIC COMPETENCIES, KNOWLEDGE AND SKILLS REQUIRED"
            ],
            "json_property": "specific_competencies_knowledge_and_skills"
        },
        "tooling": {
            "headings": [
                "TOOLING AND EQUIPMENT REQUIRED",
                "TOOLING, EQUIPMENT REQUIRED",
                "EQUIPMENT, SPECIAL TOOLING AND CONSUMABLES REQUIRED"
            ],
            "json_property": "tooling_equipment_required"
        },
        "reference_documentation": {
            "headings": ["REFERENCE DOCUMENTATION"],
            "json_property": "reference_documentation"
        },
        "reference_drawings": {
            "headings": ["REFERENCE DRAWINGS"],
            "json_property": "reference_drawings"
        },
        "attached_images": {
            "headings": ["ATTACHED PICTURES, DRAWINGS OR DIAGRAMS"],
            "json_property": "attached_images"
        }
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class FileSystemUtils:
    """File system operations"""
    
    @staticmethod
    def create_directory(dir_path: str) -> None:
        """Creates a directory at the specified path"""
        try:
            os.makedirs(dir_path)
            print(f"Directory '{dir_path}' created successfully.")
        except FileExistsError:
            print(f"Directory '{dir_path}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create directory '{dir_path}'.")
        except OSError as oer:
            print(f"Error: {oer.strerror}")
    
    @staticmethod
    def load_json(file_path: str) -> Dict:
        """Load JSON from file"""
        try:
            with open(file_path, 'r', encoding='utf8') as json_file:
                return json.load(json_file)
        except OSError as oer:
            print(oer)
            raise
    
    @staticmethod
    def store_json(data: Dict, file_path: str) -> None:
        """Store data as JSON file"""
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        try:
            with open(file_path, "w", encoding='utf8') as outfile:
                outfile.write(json_str)
            print(f'{file_path} file created successfully.')
        except OSError as oer:
            print(oer)


# =============================================================================
# PDF PROCESSING
# =============================================================================

class PDFExtractor:
    """Extract text content from PDF files"""
    
    @staticmethod
    def extract_text(pdf_path: str, start_page: int = 0, max_pages: int = 50) -> List[str]:
        """Extract text from PDF pages"""
        pdf_contents_list = []
        try:
            pdf_doc = pymupdf.open(pdf_path)
            print(f'Total pages in the PDF: {pdf_doc.page_count}')
            
            end_page = min(max_pages, pdf_doc.page_count) if pdf_doc.page_count < 150 else max_pages
            print(f'Text extraction is performed from {start_page} to {end_page} pages.')
            
            for page in pdf_doc[start_page:end_page]:
                pdf_page = page.get_text()
                pdf_contents_list.append(pdf_page)
                
        except OSError as oer:
            print(oer)
            raise
            
        return pdf_contents_list


class PDFtoHTMLConverter:
    """Convert PDF to HTML with image extraction"""
    
    def __init__(self, pdf_path: str, image_dir: str, html_path: str):
        self.pdf_path = pdf_path
        self.image_dir = image_dir
        self.html_path = html_path
        
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
    
    @staticmethod
    def expand_bbox(bbox: Tuple, margin: float) -> Tuple:
        """Expand bounding box by margin"""
        x0, y0, x1, y1 = bbox
        return (x0-margin, y0-margin, x1+margin, y1+margin)
    
    @staticmethod
    def bboxes_overlap(bbox1: Tuple, bbox2: Tuple, threshold: float = 0.0) -> bool:
        """Check if two bounding boxes overlap"""
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2
        
        dx = min(x1_1, x1_2) - max(x0_1, x0_2)
        dy = min(y1_1, y1_2) - max(y0_1, y0_2)
        
        if dx >= 0 and dy >= 0:
            intersection = dx * dy
            area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
            area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
            iou = intersection / float(area1 + area2 - intersection)
            return iou > threshold
        return False
    
    def convert(self) -> str:
        """Convert PDF to HTML"""
        try:
            doc = pymupdf.open(self.pdf_path)
            all_pages_html = []
            
            for page_num, page in enumerate(doc):
                page_html = self._process_page(page, page_num)
                all_pages_html.append(page_html)
            
            full_html = self._generate_html_document(all_pages_html)
            
            with open(self.html_path, "w", encoding="utf-8") as f:
                f.write(full_html)
            
            print(f"Done! Output HTML written to: {self.html_path}")
            print(f"Extracted images are in the folder: {self.image_dir}")
            return full_html
            
        except Exception as e:
            print(f"Error during PDF to HTML conversion: {e}")
            return False
    
    def _process_page(self, page, page_num: int) -> str:
        """Process a single PDF page"""
        blocks = page.get_text("dict")["blocks"]
        elements = []
        img_count = 0
        image_bboxes = []
        
        # Extract images
        for block in blocks:
            if block["type"] == 1:
                x0, y0, x1, y1 = block["bbox"]
                if y0 >= 0 and y1 <= 90:
                    continue
                
                rect = pymupdf.Rect(x0, y0, x1, y1)
                mat = pymupdf.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat, clip=rect, alpha=True)
                img_count += 1
                img_filename = f"page{page_num+1}_img{img_count}_rendered.png"
                img_filepath = os.path.join(self.image_dir, img_filename)
                pix.save(img_filepath)
                
                elements.append({
                    "type": "image",
                    "y": y0,
                    "x": x0,
                    "filename": img_filename,
                    "bbox": (x0, y0, x1, y1)
                })
                image_bboxes.append((x0, y0, x1, y1))
        
        # Define header/footer regions
        page_width = int(page.rect.width)
        page_height = int(page.rect.height)
        header_region = (0, 0, page_width, 90)
        footer_region = (0, page_height - 90, page_width, page_height)
        
        # Extract text
        for block in blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        span_bbox = span["bbox"]
                        x0, y0, x1, y1 = span_bbox
                        
                        overlap = any(self.bboxes_overlap(span_bbox, img_bbox) 
                                    for img_bbox in image_bboxes)
                        in_header = self.bboxes_overlap(span_bbox, header_region)
                        in_footer = self.bboxes_overlap(span_bbox, footer_region)
                        
                        if not overlap and not in_header and not in_footer:
                            elements.append({
                                "type": "text",
                                "y": y0,
                                "x": x0,
                                "text": span["text"],
                                "bbox": span_bbox,
                                "font": span.get("font", "Arial"),
                                "size": span.get("size", 12)
                            })
        
        elements.sort(key=lambda e: (e['y'], e['x']))
        
        return self._generate_page_html(elements, page_width, page_height)
    
    def _generate_page_html(self, elements: List[Dict], width: int, height: int) -> str:
        """Generate HTML for a single page"""
        page_html = (
            f'<div style="position:relative; width:{width}px; height:{height}px; '
            'border:1px solid #ccc; margin-bottom:30px;">\n'
        )
        
        for el in elements:
            left, top, right, bottom = el['bbox']
            el_width = right - left
            el_height = bottom - top
            style = (
                f"position:absolute; left:{left}px; top:{top}px; width:{el_width}px; "
                f"height:{el_height}px; overflow:hidden;"
            )
            
            if el["type"] == "text":
                page_html += (
                    f'<div style="{style} font-size:{el["size"]}px; font-family:{el["font"]};'
                    ' color:#111; background:rgba(255,255,255,0.0);">'
                    f'{el["text"]}</div>\n'
                )
            else:
                img_src = os.path.join("Media", el["filename"])
                page_html += f'<img src="{img_src}" style="{style} object-fit:contain;" />\n'
        
        page_html += "</div>\n"
        return page_html
    
    def _generate_html_document(self, pages_html: List[str]) -> str:
        """Generate complete HTML document"""
        return (
            "<!DOCTYPE html>\n"
            "<html>\n<head>\n"
            '    <meta charset="utf-8">\n'
            "    <title>PDF to HTML (No double text on images)</title>\n"
            "</head>\n<body>\n"
            f"{''.join(pages_html)}"
            "</body>\n</html>\n"
        )


# =============================================================================
# AWS BEDROCK CLIENT
# =============================================================================

class BedrockClient:
    """AWS Bedrock client management"""
    
    def __init__(self, namespace: str = Config.BEDROCK_NAMESPACE):
        self.namespace = namespace
    
    def create_client(self, consumer_bool: bool = True, runtime_client: bool = True):
        """Create and configure Bedrock client"""
        sts_client = boto3.client("sts")
        
        role_type = "bedrock-consumer" if consumer_bool else "bedrock-developer"
        bedrock_account = sts_client.assume_role(
            RoleArn=f"arn:aws:iam::533267133246:role/{self.namespace}-{role_type}",
            RoleSessionName=role_type,
        )
        
        credentials = bedrock_account["Credentials"]
        config = boto_Config(
            retries={"total_max_attempts": 20, "mode": "standard"}, 
            read_timeout=1000
        )
        
        service_name = "bedrock-runtime" if runtime_client else "bedrock"
        endpoint = f"https://{service_name}.ap-southeast-2.amazonaws.com"
        
        return boto3.client(
            service_name=service_name,
            region_name="ap-southeast-2",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            config=config,
            endpoint_url=endpoint,
        )


# =============================================================================
# LLM CONVERSATION HANDLER
# =============================================================================

class ConversationHandler:
    """Handle LLM conversations with continuation support"""
    
    def __init__(self, bedrock_client):
        self.bedrock_client = bedrock_client
    
    def generate_conversation(
        self, 
        model_id: str, 
        system_prompts: List[Dict], 
        messages: List[Dict],
        task_type: str = "general", 
        enable_continuation: bool = True, 
        max_attempts: int = 5
    ) -> Dict:
        """Generate conversation with optional continuation"""
        logger.info("Generating message with model %s for task: %s", model_id, task_type)
        
        if not enable_continuation:
            return self._original_generate_conversation(
                model_id, system_prompts, messages
            )
        
        return self._generate_with_continuation(
            model_id, system_prompts, messages, task_type, max_attempts
        )
    
    def _original_generate_conversation(
        self, 
        model_id: str, 
        system_prompts: List[Dict], 
        messages: List[Dict]
    ) -> Dict:
        """Original conversation generation"""
        if model_id == Config.MODEL_ID_40:
            inference_config = {"temperature": 0, "maxTokens": 16384, "topP": 1}
            additional_model_fields = {"top_k": 250}
        else:
            inference_config = {"temperature": 0, "maxTokens": 8192}
            additional_model_fields = {"top_k": 200}
        
        response = self.bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields
        )
        
        token_usage = response['usage']
        logger.info("Input tokens: %s", token_usage['inputTokens'])
        logger.info("Output tokens: %s", token_usage['outputTokens'])
        logger.info("Total tokens: %s", token_usage['totalTokens'])
        logger.info("Stop reason: %s", response['stopReason'])
        
        return response
    
    def _generate_with_continuation(
        self,
        model_id: str,
        system_prompts: List[Dict],
        messages: List[Dict],
        task_type: str,
        max_attempts: int
    ) -> Dict:
        """Generate with continuation support"""
        full_response_text = ""
        conversation_messages = messages.copy()
        total_input_tokens = 0
        total_output_tokens = 0
        continuation_count = 0
        
        for attempt in range(max_attempts):
            response = self._original_generate_conversation(
                model_id, system_prompts, conversation_messages
            )
            
            content = response['output']['message']['content'][0]['text']
            full_response_text += content
            logger.debug(
                f"Attempt {attempt + 1}, Stop reason: {response['stopReason']}, "
                f"Content length: {len(content)}"
            )
            
            if 'usage' in response:
                total_input_tokens += response['usage']['inputTokens']
                total_output_tokens += response['usage']['outputTokens']
            
            if response['stopReason'] in ['end_turn', 'stop_sequence']:
                break
            elif response['stopReason'] == 'max_tokens':
                continuation_count += 1
                continuation_prompt = ContinuationPromptGenerator.get_prompt(
                    task_type, full_response_text, conversation_messages
                )
                
                conversation_messages.append({
                    "role": "assistant",
                    "content": [{"text": content}]
                })
                conversation_messages.append({
                    "role": "user",
                    "content": [{"text": continuation_prompt}]
                })
            else:
                break
        
        final_response = ResponseValidator.validate_and_clean(
            full_response_text, task_type
        )
        
        response['output']['message']['content'][0]['text'] = final_response
        response['usage']['inputTokens'] = total_input_tokens
        response['usage']['outputTokens'] = total_output_tokens
        response['usage']['totalTokens'] = total_input_tokens + total_output_tokens
        response['continuationCount'] = continuation_count
        
        return response


# =============================================================================
# CONTINUATION PROMPT GENERATOR
# =============================================================================

class ContinuationPromptGenerator:
    """Generate task-specific continuation prompts"""
    
    @staticmethod
    def get_prompt(
        task_type: str, 
        current_response: str, 
        conversation_messages: List[Dict]
    ) -> str:
        """Get continuation prompt based on task type"""
        if task_type == "html_chunking":
            return ContinuationPromptGenerator._html_chunking_prompt(
                current_response, conversation_messages
            )
        elif task_type == "json_extraction":
            return ContinuationPromptGenerator._json_extraction_prompt()
        else:
            return ContinuationPromptGenerator._generic_prompt()
    
    @staticmethod
    def _html_chunking_prompt(
        current_response: str, 
        conversation_messages: List[Dict]
    ) -> str:
        """Continuation prompt for HTML chunking"""
        original_html = ""
        for msg in conversation_messages:
            if msg["role"] == "user":
                content = msg["content"][0]["text"]
                if "<html" in content.lower() or "<!doctype" in content.lower():
                    original_html = content
                    break
        
        processed_sections = ResponseValidator._extract_processed_sections(current_response)
        
        return f"""CONTINUATION FOR HTML CHUNKING TASK:

CRITICAL RULES:
1. Continue processing the remaining HTML sections that haven't been chunked yet
2. Already processed sections: {processed_sections}
3. DO NOT repeat any previously chunked sections
4. Continue with the next unprocessed section in the HTML
5. Maintain the same JSON format for new chunks
6. Ensure all remaining sections are processed

Original HTML length: {len(original_html)} characters
Current response length: {len(current_response)} characters

** DO NOT provide any explanatory or preamble text or information **
** Continue exactly where you left off with the next HTML section **"""
    
    @staticmethod
    def _json_extraction_prompt() -> str:
        """Continuation prompt for JSON extraction"""
        return """CRITICAL RULES FOR CONTINUATION:
1. If your response gets cut off, ensure the JSON structure is valid up to that point
2. When continuing, **DO NOT repeat any previous content**
3. When continuing, pick up exactly where you left off - if the last content ended with an incomplete field like '"text', complete that field first
4. **Never duplicate the entire JSON structure**
5. Always maintain proper JSON syntax

**DO NOT provide any explanatory or preamble text or information**
**Ensure the JSON is syntactically correct and ready for immediate parsing**
**Continue the JSON structure seamlessly from the cutoff point**"""
    
    @staticmethod
    def _generic_prompt() -> str:
        """Generic continuation prompt"""
        return """CRITICAL RULES FOR CONTINUATION:
1. If your response gets cut off, ensure the JSON structure is valid up to that point
2. When continuing, ** DO NOT repeat any previous content **
3. When continuing, pick up exactly where you left off
4. ** Never duplicate the entire JSON structure **
5. Always maintain proper JSON syntax
** DO NOT provide any explanatory or preamble text or information **
** Ensure the JSON is syntactically correct and ready for immediate parsing **"""


# =============================================================================
# RESPONSE VALIDATOR
# =============================================================================

class ResponseValidator:
    """Validate and clean LLM responses"""
    
    @staticmethod
    def validate_and_clean(response: str, task_type: str) -> str:
        """Validate and clean response based on task type"""
        if task_type == "json_extraction":
            return ResponseValidator._clean_json_response(response)
        elif task_type == "html_chunking":
            return ResponseValidator._clean_html_chunking_response(response)
        else:
            return response.strip()
    
    @staticmethod
    def _clean_json_response(response: str) -> str:
        """Clean and validate JSON response"""
        try:
            response = response.strip()
            
            start_idx = response.find('{')
            if start_idx == -1:
                start_idx = response.find('[')
            
            if start_idx != -1:
                bracket_count = 0
                end_idx = -1
                start_char = response[start_idx]
                end_char = '}' if start_char == '{' else ']'
                
                for i in range(start_idx, len(response)):
                    if response[i] == start_char:
                        bracket_count += 1
                    elif response[i] == end_char:
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    try:
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError:
                        return ResponseValidator._attempt_json_repair(json_str)
            
            return response
        except Exception as e:
            logger.warning(f"Error cleaning JSON: {e}")
            return response
    
    @staticmethod
    def _clean_html_chunking_response(response: str) -> str:
        """Clean HTML chunking response"""
        return ResponseValidator._clean_json_response(response)
    
    @staticmethod
    def _attempt_json_repair(json_str: str) -> str:
        """Attempt basic JSON repair"""
        try:
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
            json.loads(json_str)
            return json_str
        except:
            return json_str
    
    @staticmethod
    def _extract_processed_sections(response: str) -> List[str]:
        """Extract processed section names"""
        sections = []
        try:
            if response.strip().startswith('{') or response.strip().startswith('['):
                section_patterns = [
                    r'"section":\s*"([^"]+)"',
                    r'"name":\s*"([^"]+)"',
                    r'"id":\s*"([^"]+)"',
                    r'"title":\s*"([^"]+)"',
                    r'"section_name":\s*"([^"]+)"'
                ]
                for pattern in section_patterns:
                    matches = re.findall(pattern, response, re.IGNORECASE)
                    sections.extend(matches)
        except:
            pass
        return list(set(sections))


# =============================================================================
# PROMPT LOADER
# =============================================================================

class PromptLoader:
    """Load prompts from YAML files"""
    
    @staticmethod
    def load_prompts(file_path: str, prompt_key: str) -> Dict[str, str]:
        """Load YAML with comprehensive error handling"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"YAML file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            
            if not isinstance(data, dict):
                raise ValueError("YAML file must contain a dictionary at root level")
            
            if prompt_key not in data:
                available = list(data.keys())
                raise KeyError(f"Prompt '{prompt_key}' not found. Available: {available}")
            
            prompt_config = data[prompt_key]
            required_keys = ['system_prompt', 'message_prompt']
            missing_keys = [key for key in required_keys if key not in prompt_config]
            
            if missing_keys:
                raise ValueError(f"Missing required keys in prompt config: {missing_keys}")
            
            return {
                'system_prompt': prompt_config['system_prompt'],
                'message_prompt': prompt_config['message_prompt']
            }
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading prompts: {e}")


# =============================================================================
# SECTION CATEGORIZER
# =============================================================================

class SectionCategorizer:
    """Categorize document sections"""
    
    @staticmethod
    def get_category(section_title: str) -> Tuple[str, str]:
        """Get category and JSON property for a section"""
        section_upper = section_title.upper().strip()
        
        for category, config in Config.SECTION_CATEGORY_MAPPING.items():
            for heading in config["headings"]:
                if heading.upper() == section_upper or heading.upper() in section_upper:
                    return category, config["json_property"]
        
        return "task_activities", "task_activities"
    
    @staticmethod
    def extract_schema_for_section(
        section_title: str, 
        json_template_schema: Dict
    ) -> Optional[Tuple[str, Dict]]:
        """Extract relevant JSON schema for a section"""
        category, json_property = SectionCategorizer.get_category(section_title)
        print(f"Section: {section_title} -> Category: {category}, Property: {json_property}")
        
        if json_property not in json_template_schema:
            print(f"Warning: JSON property '{json_property}' not found in schema for section: {section_title}")
            return None
        
        return category, {json_property: json_template_schema[json_property]}
    
    @staticmethod
    def filter_sections(sections: List[str]) -> List[str]:
        """Filter out excluded sections"""
        return [
            title for title in sections 
            if title.strip().lower() not in 
            [h.strip().lower() for h in Config.SECTION_EXCLUSION_LIST]
        ]
    
    @staticmethod
    def has_number(input_string: str) -> bool:
        """Check if string contains a digit"""
        return any(char.isdigit() for char in input_string)


# =============================================================================
# JSON SCHEMA GENERATORS
# =============================================================================

class JSONSchemaGenerator:
    """Generate JSON schemas for different section types"""
    
    @staticmethod
    def task_activity_schema(section: str) -> str:
        """Generate schema for task activity sections"""
        section_list = section.split('.')
        section_number = section_list[0]
        section_items = "".join(section_list[1:])
        section_name = section_items.strip()
        
        json_sample_dict = {
            "equipment_asset": {},
            "sequence_no": {"text": f"{section_number}"},
            "sequence_name": {"text": f"{section_name}"},
            "maintainable_item": [],
            "lmi": [],
            "step_no": {"text": f"{section_number}.1", "image": ""},
            "step_description": [{"text":"", "image": ""}],
            "photo_diagram": [{"text":"", "image": ""}],
            "notes": [{"text":"", "image": ""}],
            "acceptable_limit": [],
            "question": [],
            "corrective_action": [],
            "execution_condition": {"text":"", "image": ""},
            "other_content": []
        }
        
        return f"{json_sample_dict} \n (so on extract all the task steps information in the step_no, step_description, photo_diagram, notes. Till end of this {section})"
    
    @staticmethod
    def task_execution_schema(section: str) -> str:
        """Generate schema for task execution sections"""
        json_sample_dict = {
            "equipment_asset": {},
            "sequence_no": {"text": ""},
            "sequence_name": {"text": f"{section}"},
            "maintainable_item": [],
            "lmi": [],
            "step_no": {"text": "", "image": ""},
            "step_description": [{"text":"", "image": ""}],
            "photo_diagram": [{"text":"", "image": ""}],
            "notes": [{"text":"", "image": ""}],
            "acceptable_limit": [],
            "question": [],
            "corrective_action": [],
            "execution_condition": {"text":"", "image": ""},
            "other_content": []
        }
        
        return f"{json_sample_dict} \n ** Put all the content in step_description **"


# =============================================================================
# JSON TRANSFORMER
# =============================================================================

class JSONTransformer:
    """Transform JSON structures to required format"""
    
    FIELD_MAPPINGS = {
        "text": ("orig_text", "text"),
        "image": ("orig_image", "image"),
        "seq": ("orig_seq", "seq")
    }
    
    SECTION_RULES = {
        "task_activities": {
            "type": "task_activities",
            "group_by": "equipment_asset"
        },
        "reference_documentation": {
            "pairs": [("document_reference_number", "document_description")],
            "type": "paired"
        },
        "reference_drawings": {
            "pairs": [("drawing_reference_number", "drawing_description")],
            "type": "paired"
        },
        "material_risks_and_controls": {
            "pairs": [("risk", "risk_description", "critical_controls")],
            "type": "grouped"
        },
        "additional_controls_required": {
            "pairs": [("control_type", "reason_for_control")],
            "type": "grouped"
        },
        "tooling_equipment_required": {
            "pairs": [("tool_set", "tools")],
            "type": "grouped"
        },
        "safety": {
            "type": "object"
        },
        "additional_ppe_required": {
            "type": "simple_list"
        },
        "specific_competencies_knowledge_and_skills": {
            "type": "simple_list"
        },
        "attached_images": {
            "type": "simple_list"
        }
    }
    
    @classmethod
    def transform_item(cls, item):
        """Transform individual item"""
        if isinstance(item, dict):
            transformed = {}
            for key, value in item.items():
                if key in cls.FIELD_MAPPINGS:
                    orig_key, new_key = cls.FIELD_MAPPINGS[key]
                    transformed[orig_key] = value
                    transformed[new_key] = value
                elif isinstance(value, (list, dict)):
                    transformed[key] = cls.transform_item(value)
                else:
                    transformed[key] = value
            return transformed
        elif isinstance(item, list):
            return [cls.transform_item(sub_item) for sub_item in item]
        else:
            return item
    
    @classmethod
    def transform_structure(cls, input_json: Dict) -> Any:
        """Transform JSON structure based on section type"""
        main_key = list(input_json["value"].keys())[0]
        main_data = input_json["value"][main_key]
        
        section_rule = cls.SECTION_RULES.get(main_key, {"type": "default"})
        section_type = section_rule.get("type", "default")
        
        if section_type == "task_activities":
            return cls._transform_task_activities(main_data, section_rule)
        elif section_type == "object":
            return cls._transform_object(main_data)
        elif section_type == "simple_list":
            return cls._transform_simple_list(main_data)
        elif section_type == "paired":
            return cls._transform_paired(main_data, section_rule)
        elif section_type == "grouped":
            return cls._transform_grouped(main_data, section_rule)
        else:
            return cls._transform_default(main_data)
    
    @classmethod
    def _transform_task_activities(cls, data: List, rule: Dict) -> List:
        """Transform task activities"""
        group_by_field = rule.get("group_by", "equipment_asset")
        current_task = {}
        grouped_tasks = []
        
        for item in data:
            for section_key, section_value in item.items():
                if section_key == group_by_field and current_task:
                    grouped_tasks.append(current_task)
                    current_task = {}
                
                current_task[section_key] = cls.transform_item(section_value)
        
        if current_task:
            grouped_tasks.append(current_task)
        
        return grouped_tasks
    
    @classmethod
    def _transform_object(cls, data: List) -> Dict:
        """Transform to object"""
        result = {}
        for item in data:
            for section_key, section_value in item.items():
                result[section_key] = cls.transform_item(section_value)
        return result
    
    @classmethod
    def _transform_simple_list(cls, data: List) -> List:
        """Transform to simple list"""
        return [cls.transform_item(item) for item in data]
    
    @classmethod
    def _transform_paired(cls, data: List, rule: Dict) -> List:
        """Transform paired fields"""
        pairs = rule.get("pairs", [])
        result = []
        temp_item = {}
        
        for item in data:
            for section_key, section_value in item.items():
                transformed_value = cls.transform_item(section_value)
                
                for pair in pairs:
                    if section_key in pair:
                        pair_id = "_".join(pair)
                        if pair_id not in temp_item:
                            temp_item[pair_id] = {}
                        
                        temp_item[pair_id][section_key] = transformed_value
                        
                        if all(p_key in temp_item[pair_id] for p_key in pair):
                            result.append(dict(temp_item[pair_id]))
                            del temp_item[pair_id]
                        break
                else:
                    result.append({section_key: transformed_value})
        
        for pair_id, remaining in temp_item.items():
            if remaining:
                result.append(remaining)
        
        return result
    
    @classmethod
    def _transform_grouped(cls, data: List, rule: Dict) -> List:
        """Transform grouped fields"""
        pairs = rule.get("pairs", [])
        result = []
        
        for pair in pairs:
            group_items = []
            for item in data:
                for section_key, section_value in item.items():
                    if section_key in pair:
                        found = False
                        for group in group_items:
                            if section_key not in group:
                                group[section_key] = cls.transform_item(section_value)
                                found = True
                                break
                        
                        if not found:
                            group_items.append({section_key: cls.transform_item(section_value)})
            
            result.extend(group_items)
        
        return result
    
    @classmethod
    def _transform_default(cls, data: List) -> List:
        """Default transformation"""
        result = []
        for item in data:
            for section_key, section_value in item.items():
                result.append({section_key: cls.transform_item(section_value)})
        return result
    
    @classmethod
    def transform_document(cls, doc_contents: Dict, json_data: List) -> Dict:
        """Transform entire document"""
        header = doc_contents['document_header']
        
        document_header = {
            "document_source": {"orig_text": header['document_source'], "text": header['document_source']},
            "document_type": {"orig_text": header['document_type'], "text": header['document_type']},
            "document_number": {"orig_text": header['document_number'], "text": header['document_number']},
            "document_version_number": {"orig_text": header['document_version_number'], "text": header['document_version_number']},
            "work_description": {"orig_text": header['work_description'], "text": header['work_description']},
            "purpose": {"orig_text": header['purpose'], "text": header['purpose']},
            "sections": header['sections']
        }
        
        output = {
            "document_id": header['document_number'],
            "document_header": document_header,
            "safety": {},
            "material_risks_and_controls": [],
            "additional_controls_required": [],
            "additional_ppe_required": [],
            "specific_competencies_knowledge_and_skills": [],
            "tooling_equipment_required": [],
            "reference_documentation": [],
            "reference_drawings": [],
            "attached_images": [],
            "task_activities": [],
            "unhandled_content": []
        }
        
        for section in json_data:
            key = section["key"]
            category, mapped_key = SectionCategorizer.get_category(key)
            updated_json = cls.transform_structure(section)
            
            if category == "task_activities":
                output[category].extend(updated_json)
            else:
                if mapped_key == "safety":
                    output[mapped_key] = updated_json
                else:
                    if isinstance(updated_json, list):
                        output[mapped_key].extend(updated_json)
                    else:
                        print(f"Warning: {mapped_key} returned non-list: {type(updated_json)}")
                        output[mapped_key] = updated_json
        
        return output


# =============================================================================
# DOCUMENT PROCESSOR (MAIN ORCHESTRATOR)
# =============================================================================

class DocumentProcessor:
    """Main document processing orchestrator"""
    
    def __init__(self, doc_name: str):
        self.doc_name = doc_name
        self.setup_paths()
        self.bedrock_client = BedrockClient()
        self.conversation_handler = None
        
    def setup_paths(self):
        """Setup all necessary paths"""
        # Input paths
        waio_win_dir = os.path.join(Config.INPUT_DIR, "WAIO WINs")
        self.word_document_path = os.path.join(waio_win_dir, f"{self.doc_name}.docx")
        self.pdf_document_path = os.path.join(waio_win_dir, f"{self.doc_name}.pdf")
        
        # Output paths
        self.output_dir = os.path.join(Config.OUTPUT_DIR, self.doc_name)
        self.media_dir = os.path.join(self.output_dir, "Media")
        self.html_path = os.path.join(self.output_dir, f"{self.doc_name}.html")
        
        # Create directories
        FileSystemUtils.create_directory(self.output_dir)
        FileSystemUtils.create_directory(self.media_dir)
        
        # Prompt paths
        self.header_prompts_path = os.path.join(Config.PROMPTS_DIR, "win_gen_header_prompts.yml")
        self.split_html_prompts_path = os.path.join(Config.PROMPTS_DIR, "win_split_html_prompts.yml")
    
    def process(self):
        """Main processing pipeline"""
        print("=" * 80)
        print(f"Processing document: {self.doc_name}")
        print("=" * 80)
        
        # Step 1: Convert PDF to HTML
        print("\n[1/7] Converting PDF to HTML...")
        html_content = self.convert_pdf_to_html()
        
        # Step 2: Setup Bedrock clients
        print("\n[2/7] Setting up AWS Bedrock clients...")
        self.setup_bedrock_clients()
        
        # Step 3: Extract document headers
        print("\n[3/7] Extracting document section headers...")
        doc_contents = self.extract_document_headers()
        
        # Step 4: Filter sections
        print("\n[4/7] Filtering sections...")
        updated_sections = self.filter_sections(doc_contents)
        
        # Step 5: Split HTML by sections
        print("\n[5/7] Splitting HTML by sections...")
        section_wise_html = self.split_html_by_sections(html_content, updated_sections)
        
        # Step 6: Extract structured data
        print("\n[6/7] Extracting structured data from sections...")
        json_data = self.extract_structured_data(
            doc_contents, updated_sections, section_wise_html
        )
        
        # Step 7: Transform and save final result
        print("\n[7/7] Transforming and saving final result...")
        final_result = self.transform_and_save(doc_contents, json_data)
        
        print("\n" + "=" * 80)
        print("Processing complete!")
        print("=" * 80)
        
        return final_result
    
    def convert_pdf_to_html(self) -> str:
        """Convert PDF to HTML"""
        converter = PDFtoHTMLConverter(
            self.pdf_document_path, 
            self.media_dir, 
            self.html_path
        )
        return converter.convert()
    
    def setup_bedrock_clients(self):
        """Setup Bedrock clients"""
        runtime_client = self.bedrock_client.create_client(
            consumer_bool=True, 
            runtime_client=True
        )
        self.conversation_handler = ConversationHandler(runtime_client)
    
    def extract_document_headers(self) -> Dict:
        """Extract document headers and sections"""
        # Extract text from PDF
        pdf_text = PDFExtractor.extract_text(self.pdf_document_path)
        
        if not pdf_text:
            raise ValueError(f'PDF extraction returned empty list for: {self.pdf_document_path}')
        
        # Load prompts
        prompts = PromptLoader.load_prompts(self.header_prompts_path, "document_section_extraction")
        system_prompts = [{"text": prompts['system_prompt']}]
        formatted_message = prompts['message_prompt'].format(pdf_extracted_text=pdf_text)
        message_prompt = {"role": "user", "content": [{"text": formatted_message}]}
        
        # Generate response
        try:
            response = self.conversation_handler.generate_conversation(
                model_id=Config.MODEL_ID_37,
                system_prompts=system_prompts,
                messages=[message_prompt],
                task_type="general"
            )
            
            output = response['output']['message']['content'][0]['text']
            doc_contents = json.loads(output)
            
            # Save header JSON
            header_json_path = os.path.join(self.output_dir, f'{self.doc_name}_header.json')
            FileSystemUtils.store_json(doc_contents, header_json_path)
            
            return doc_contents
            
        except ClientError as err:
            logger.error("A client error occurred: %s", err.response['Error']['Message'])
            raise
    
    def filter_sections(self, doc_contents: Dict) -> List[str]:
        """Filter sections based on exclusion list"""
        sections = doc_contents['document_header']['sections']
        updated_sections = SectionCategorizer.filter_sections(sections)
        doc_contents['document_header']['sections'] = updated_sections
        return updated_sections
    
    def load_split_prompts(self, yaml_file_path: str) -> Dict:
        """Load prompts from YAML file."""
        with open(yaml_file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def split_html_by_sections(self, html_content: str, sections: List[str]) -> Dict:
        """Split HTML content by sections"""
        section_wise_html = {}
        split_prompts = self.load_split_prompts(
            self.split_html_prompts_path, 
        )["markdown_section_split"]
        
        for i, section_name in enumerate(sections):
            print(f"  Processing section: {section_name}")
            
            # Prepare prompts
            system_prompt = [{
                "text": split_prompts["system_prompt"].format(html_content=html_content)
            }]
            
            if i == len(sections) - 1:
                message_text = split_prompts["last_section"].format(section_name=section_name)
            else:
                next_section = sections[i + 1]
                message_text = split_prompts["with_next_section"].format(
                    section_name=section_name, 
                    next_section=next_section
                )
            
            message_prompt = {"role": "user", "content": [{"text": message_text}]}
            
            try:
                response = self.conversation_handler.generate_conversation(
                    model_id=Config.MODEL_ID_40,
                    system_prompts=system_prompt,
                    messages=[message_prompt],
                    task_type="html_chunking"
                )
                
                output = response['output']['message']['content'][0]['text']
                section_wise_html[section_name] = {"html_content": output}
                
            except ClientError as err:
                logger.error("A client error occurred: %s", err.response['Error']['Message'])
                raise
        
        # Save section-wise HTML
        section_json_path = os.path.join(
            self.output_dir, 
            f'{self.doc_name}_section_wise_html_content.json'
        )
        FileSystemUtils.store_json(section_wise_html, section_json_path)
        
        return section_wise_html
    
    def extract_structured_data(
        self, 
        doc_contents: Dict, 
        sections: List[str], 
        section_wise_html: Dict
    ) -> List[Dict]:
        """Extract structured data from HTML sections"""
        # Determine document type and load appropriate templates/prompts
        doc_type = doc_contents['document_header']['document_type']
        
        if doc_type in ["Work Instruction", "Maintenance Work Instruction"]:
            print("  Loading WIN prompts and templates...")
            template_path = os.path.join(Config.TEMPLATE_DIR, "WIN_Template.json")
            prompts_path = os.path.join(Config.PROMPTS_DIR, "win_prompts.yml")
        elif doc_type in [
            "Preventive Maintenance Instruction",
            "Preventative Maint Instruction",
            "PRT Work Instruction",
            "Mechanical Inspection",
            "Mechanical Inspection Procedure",
            "2W Mech Insp Conveyor P10",
            "Preventative Maintenance Inspection"
        ]:
            print("  Loading PMI prompts and templates...")
            template_path = os.path.join(Config.TEMPLATE_DIR, "PMI_Template.json")
            prompts_path = os.path.join(Config.PROMPTS_DIR, "pmi_prompts.yml")
        else:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        json_template = FileSystemUtils.load_json(template_path)
        prompts = PromptLoader.load_prompts(prompts_path, 'section_wise_prompts')
        system_prompts = [{"text": prompts['system_prompt']}]
        
        json_data = []
        section_wise_responses = {}
        
        for section_name in sections:
            print(f"  Extracting data from: {section_name}")
            
            section_html = section_wise_html[section_name]['html_content']
            result = SectionCategorizer.extract_schema_for_section(section_name, json_template)
            
            if not result:
                print(f"  Warning: No schema found for section: {section_name}")
                continue
            
            section_category, json_schema = result
            
            # Generate appropriate JSON example based on category
            if section_category == "task_activities" and SectionCategorizer.has_number(section_name):
                json_example = JSONSchemaGenerator.task_activity_schema(section_name)
            elif section_category == "task_activities":
                json_example = JSONSchemaGenerator.task_execution_schema(section_name)
            else:
                json_example = ""
            
            formatted_message = prompts['message_prompt'].format(
                section_name=section_name,
                section_html_content=section_html,
                json_schema=json_schema,
                json_sample_dict=json_example
            )
            message_prompt = {"role": "user", "content": [{"text": formatted_message}]}
            
            try:
                response = self.conversation_handler.generate_conversation(
                    model_id=Config.MODEL_ID_37,
                    system_prompts=system_prompts,
                    messages=[message_prompt],
                    task_type="json_extraction"
                )
                
                output = response['output']['message']['content'][0]['text']
                section_wise_responses[section_name] = {"llm_response": output}
                
                json_contents = json.loads(output)
                json_data.append({"key": section_name, "value": json_contents})
                
            except ClientError as err:
                logger.error("A client error occurred: %s", err.response['Error']['Message'])
                raise
        
        # Save responses
        response_json_path = os.path.join(
            self.output_dir, 
            f'{self.doc_name}_section_wise_llm_response.json'
        )
        FileSystemUtils.store_json(section_wise_responses, response_json_path)
        
        raw_json_path = os.path.join(self.output_dir, f'{self.doc_name}.json')
        FileSystemUtils.store_json(json_data, raw_json_path)
        
        return json_data
    
    def transform_and_save(self, doc_contents: Dict, json_data: List[Dict]) -> Dict:
        """Transform JSON data and save final result"""
        final_result = JSONTransformer.transform_document(doc_contents, json_data)
        
        result_path = os.path.join(self.output_dir, f'result_{self.doc_name}.json')
        FileSystemUtils.store_json(final_result, result_path)
        
        print(f"\nFinal result saved to: {result_path}")
        return final_result


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    # Document to process
    doc_name_list = ["0008496", "0159380", "0134622", "0129376", "0121071"]#, 0008496
    
    # Create processor and run
    for doc_name in doc_name_list:
        processor = DocumentProcessor(doc_name)
        result = processor.process()
        
        print("\nProcessing Summary:")
        print(f"  Document ID: {result['document_id']}")
        print(f"  Document Type: {result['document_header']['document_type']['text']}")
        print(f"  Number of task activities: {len(result['task_activities'])}")
        print(f"  Number of sections: {len(result['document_header']['sections'])}")


if __name__ == "__main__":
    main()