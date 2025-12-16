"""
Stage 3.5: Review Agent
Reviews each section with its page images and validates for:
- Sentence completeness
- Information duplication
- Logical section ordering
"""
import json
from typing import Dict, List, Any

from config.settings import MODEL_MAX_TOKENS_VALIDATION, MODEL
from src.tools import invoke_bedrock_multimodal, prepare_images_for_bedrock
from src.utils import setup_logger, StorageManager

logger = setup_logger('review_agent')


class ReviewAgent:
    """Agent to review and validate extracted section content."""
    
    def __init__(self):
        self.storage = StorageManager()
    
    def review_document(
        self,
        section_jsons: List[Dict],
        document_id: str,
        pages_data: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Review extracted document sections for quality issues.
        
        Args:
            section_jsons: List of extracted section JSONs
            document_id: Unique document identifier
            pages_data: List of page data with images (optional, for section-by-section review)
        
        Returns:
            Dictionary containing review findings (empty if no issues)
        """
        logger.info(f"[{document_id}] Starting document review")
        
        try:
            # Aggregate results from all sections
            aggregated_results = {
                'WORD_ACCURACY': [],
                'MISSING_INFORMATION_ACCURACY': [],
                'DUPLICATION_ACCURACY': []
                # 'order_issues': [],
                # 'missing_information': []
            }
            
            # Step 1: Review each section individually with its page images
            logger.info(f"[{document_id}] Reviewing {len(section_jsons)} sections...")
            
            for idx, section in enumerate(section_jsons, 1):
                section_name = section.get('section_name', f'Section {idx}')
                logger.info(f"  [{idx}/{len(section_jsons)}] Reviewing: {section_name}")
                
                # Convert section JSON to plain text
                section_plain_text = self._section_json_to_plain_text(section, section_name)
                
                # Get page images for this section if available
                section_images = None
                if pages_data:
                    page_range = section.get('page_range', [])
                    if page_range and len(page_range) == 2:
                        start_idx = page_range[0] - 1
                        end_idx = page_range[1]
                        section_pages = pages_data[start_idx:end_idx]
                        section_images = prepare_images_for_bedrock(section_pages)
                
                # Review this section
                section_review = self._validate_section_content(
                    section_plain_text,
                    section_name,
                    section_images,
                    document_id
                )
                
                # Aggregate results
                for key in aggregated_results.keys():
                    if key in section_review and section_review[key]:
                        aggregated_results[key].extend(section_review[key])
            
            # # Step 2: Check for cross-section duplications
            # logger.info(f"[{document_id}] Checking for cross-section duplications...")
            # cross_section_duplications = self._check_cross_section_duplications(
            #     section_jsons,
            #     document_id
            # )
            # aggregated_results['duplications'].extend(cross_section_duplications)
            
            # Step 3: Save aggregated results
            self.storage.save_review_results(document_id, aggregated_results)
            
            # Save plain text for reference (all sections combined)
            full_plain_text = self._all_sections_to_plain_text(section_jsons)
            self.storage.save_plain_text(document_id, full_plain_text)
            
            # Log summary
            total_issues = sum(len(issues) for issues in aggregated_results.values())
            if total_issues == 0:
                logger.info(f"[{document_id}] Review passed - no issues found")
            else:
                logger.warning(
                    f"[{document_id}] Review found {total_issues} issue(s): "
                    f"{len(aggregated_results.get('incomplete_sentences', []))} incomplete, "
                    f"{len(aggregated_results.get('duplications', []))} duplications, "
                    f"{len(aggregated_results.get('order_issues', []))} order issues, "
                    f"{len(aggregated_results.get('missing_information', []))} missing info"
                )
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"[{document_id}] Review failed: {e}")
            raise
    
    def _section_json_to_plain_text(
        self,
        section: Dict,
        section_name: str
    ) -> str:
        """
        Convert a single section JSON to plain text.
        
        Args:
            section: Section JSON object
            section_name: Name of the section
        
        Returns:
            Plain text representation of the section
        """
        try:
            # Extract just the data portion of the section
            section_data = section.get('data', section)
            
            plain_text = f"=== {section_name} ===\n\n"
            plain_text += self._extract_text_from_json(section_data)
            
            return plain_text.strip()
            
        except Exception as e:
            logger.error(f"Failed to convert section {section_name} to plain text: {e}")
            return f"=== {section_name} ===\n\n[Error converting section to text]"
    
    def _extract_text_from_json(self, data: Any, indent: int = 0) -> str:
        """
        Recursively extract text content from JSON structure.
        
        Args:
            data: JSON data (can be dict, list, or primitive)
            indent: Current indentation level
        
        Returns:
            Extracted text content
        """
        text_parts = []
        indent_str = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                # Skip internal metadata and image fields
                if key.startswith('_') or key == 'image' or '_orig' in key:
                    continue
                
                # Extract text from text fields
                if key == 'text' and isinstance(value, str) and value.strip():
                    text_parts.append(f"{indent_str}{value}")
                else:
                    # Recursively process nested structures
                    nested_text = self._extract_text_from_json(value, indent)
                    if nested_text:
                        text_parts.append(nested_text)
        
        elif isinstance(data, list):
            for item in data:
                nested_text = self._extract_text_from_json(item, indent)
                if nested_text:
                    text_parts.append(nested_text)
        
        elif isinstance(data, str) and data.strip():
            text_parts.append(f"{indent_str}{data}")
        
        return "\n".join(text_parts)
    
    def _all_sections_to_plain_text(self, section_jsons: List[Dict]) -> str:
        """
        Convert all sections to plain text for reference.
        
        Args:
            section_jsons: List of all section JSONs
        
        Returns:
            Combined plain text of all sections
        """
        all_text = []
        
        for section in section_jsons:
            section_name = section.get('section_name', 'Unknown Section')
            section_text = self._section_json_to_plain_text(section, section_name)
            all_text.append(section_text)
        
        return "\n\n".join(all_text)
    
    def _validate_section_content(
        self,
        section_plain_text: str,
        section_name: str,
        section_images: List[str] = None,
        document_id: str = None
    ) -> Dict[str, List[Dict]]:
        """
        Validate a single section for completeness, duplication, and logical flow.
        
        Args:
            section_plain_text: Plain text version of the section
            section_name: Name of the section
            section_images: Base64 encoded images for this section (optional)
            document_id: Document identifier
        
        Returns:
            Dictionary with three keys:
            - incomplete_sentences: List of incomplete sentence issues
            - duplications: List of duplication issues (within section)
            - order_issues: List of section order issues
        """
        # Build different prompts depending on whether images are available
        if section_images:
            prompt = f"""You are a document quality reviewer. I am providing you with:
1. PDF PAGE IMAGES showing the actual document pages for section "{section_name}"
2. EXTRACTED TEXT that was pulled from these pages

CRITICAL: The PAGE IMAGES are the GROUND TRUTH. Your job is to validate the extracted text against what you see in the images.

SECTION NAME: {section_name}

EXTRACTED TEXT TO VALIDATE:
{section_plain_text}

YOUR TASK:
Carefully review the PAGE IMAGES first, then compare the extracted text against what you see in the images. Identify the following issue_types:

1. WORD ACCURACY:
   - Compare each extracted word to the corresponding word in the image
   - Identify words in the extracted text that do not match the word visible in the image
   - Example: If a sentence is "Remove bolts A, B, and C carefully" but extracted text says "Remove belts A, B, and" - Belts is incorrect word and has WORD ACCURACY issue.

   CRITICAL:
   - DO NOT report here if a WORD was not extracted (that's missing word, not WORD ACCURACY)

  CALCULATION WORD ACCURACY:
    - Count the total number of words in the section as E
    - Count the total number of mismatched words in the section as W
    - Calculate the accuracy as (1 - E/W)*100 
    - If 
        the accuracy>=95%, report as high with accuracy (1 - E/W)*100
        the accuracy<95% and accuracy>=85%, report as medium with accuracy (1 - E/W)*100
        the accuracy<85%, report as low with accuracy (1 - E/W)*100

2. MISSING INFORMATION ACCURACY:
   - Compare the extracted TEXT to the corresponding TEXT in the images
   - Find WORDS in the IMAGES that don't EXIST in the extracted TEXT
   - Example: If a sentence is "close the bleed down valves" in the text while in the image it is "close the bleed down valves by rotating them clockwise.", words "by', "rotating", "them", and "clockwise" are missing words.  

   CRITICAL:
   - DO NOT report here if a WORD was extracted INCORRECTLY (that's WORD ACCURACY, not MISSING INFORMATION)

  CALCULATION MISSING INFORMATION ACCURACY:
    - Count the total number of words in the section as E
    - Count the total number of missing words in the section as W
    - Calculate the accuracy as (1 - E/W)*100 
    - If 
        the accuracy>=95%, report as high with accuracy (1 - E/W)*100
        the accuracy<95% and accuracy>=85%, report as with accuracy (1 - E/W)*100
        the accuracy<85%, report as low with accuracy (1 - E/W)*100

3. DUPLICATION ACCURACY:
   - Compare extracted TEXT to the corresponding image
   - Identify sections that are duplicated
   - Focus only on entire sections that are repeated, not on individual sentences, words, or partial fragments
   
   CRITICAL:
   - DO NOT report here if individual sentences, words, or partial fragments are duplicated

  CALCULATION WORD ACCURACY:
    - Count the total number of duplicated sections
    - If 
        the count>=1, report as low
        otherwise high

        
VALIDATION METHODOLOGY:
1. LOOK AT THE IMAGES FIRST - These are the original PDF pages, the source of truth
2. Identify where section "{section_name}" appears in the images (look for the heading/title)
3. Read through the section content as it appears visually in the images
4. Compare what you see in images with the extracted text provided
5. Report any WORD ACCURACY issue

CRITICAL RULES:
- The IMAGES are the TRUTH - validate extracted text against them
- If there are no issues in a issue_type, that issue_type should be mentioned
- Consider the technical nature of the document (safety procedures)

OUTPUT FORMAT:
Return ONLY a valid JSON object with this exact structure (no markdown, no code blocks):
{{
  "WORD_ACCURACY":[
    {{
      "section_name": "{section_name}",  
      "issue_severity": "Report the accuracy here"
    }}
    ],
    "MISSING_INFORMATION_ACCURACY":[
    {{
      "section_name": "{section_name}", 
      "issue_severity": "Report the accuracy here"
    }}
    ], 
    "DUPLICATION_ACCURACY":[
    {{
      "section_name": "{section_name}", 
      "issue_severity": "Report the accuracy here"
    }}
    ]
}}

IMPORTANT: 
- If a category has NO issues, FOLLOW the EXACT STRUCTURE and report it based on high accuracy. Dont return blank array in this case.
- Return valid JSON only, no additional text
- Start with {{ and end with }}
- Base your analysis on what you SEE in the images

Look at the images and validate the extracted text now:
"""
        
        loop = True
        counter = 3
        while loop:
            loop = False
            try:
                # Use multimodal if images are available, otherwise text-only
                if section_images:
                    response = invoke_bedrock_multimodal(
                        images=section_images,
                        prompt=prompt,
                        max_tokens=MODEL_MAX_TOKENS_VALIDATION
                    )
                else:
                    # Fall back to text-only invocation
                    from src.tools import invoke_bedrock_text
                    response = invoke_bedrock_text(
                        prompt=prompt,
                        max_tokens=MODEL_MAX_TOKENS_VALIDATION
                    )
                
                # Clean and parse the response
                import re
                cleaned = re.sub(r'```(?:json)?\s*', '', response)
                cleaned = re.sub(r'```\s*$', '', cleaned)
                
                # Find JSON object
                json_pattern = r'\{[\s\S]*\}'
                match = re.search(json_pattern, cleaned)
                
                if not match:
                    raise ValueError("No JSON object found in response")
                
                json_str = match.group(0)
                section_review = json.loads(json_str)
                
                # Validate structure
                if not isinstance(section_review, dict):
                    raise ValueError("Response is not a JSON object")
                
                # Ensure all required keys exist
                for key in ['WORD_ACCURACY', 'MISSING_INFORMATION_ACCURACY', 'DUPLICATION_ACCURACY']:
                    if key not in section_review:
                        section_review[key] = []
                
                logger.debug(
                    f"[{document_id}] Section '{section_name}' validation: "
                    f"{len(section_review['WORD_ACCURACY'])} WORD ACCURACY, "
                    f"{len(section_review['MISSING_INFORMATION_ACCURACY'])} MISSING ACCURACY "
                )
                
                return section_review
                
            except json.JSONDecodeError as e:
                logger.error(f"[{document_id}] Failed to parse validation response for {section_name}: {e}")
                logger.debug(f"Response was: {response[:500]}")
                # Return empty results on parse failure
                if counter > 0:
                    logger.info(f"Trying {counter} more times .. ")
                    counter -= 1
                    loop = True
                else:
                    return {
                        'MISSING_INFORMATION_ACCURACY': [],
                        'WORD_ACCURACY': [],
                        'DUPLICATION_ACCURACY': []
                    }
            except Exception as e:
                logger.error(f"[{document_id}] Section validation failed for {section_name}: {e}")
                if counter > 0:
                    logger.info(f"Trying {counter} more times .. ")
                    counter -= 1
                    loop = True
                else:
                    return {
                        'MISSING_INFORMATION_ACCURACY': [],
                        'WORD_ACCURACY': [],
                        'DUPLICATION_ACCURACY': []
                    }
    
    def _check_cross_section_duplications(
        self,
        section_jsons: List[Dict],
        document_id: str
    ) -> List[Dict]:
        """
        Check for duplications across multiple sections.
        
        Args:
            section_jsons: List of all section JSONs
            document_id: Document identifier
        
        Returns:
            List of duplication issues found across sections
        """
        duplications = []
        
        # Build a simplified text representation for each section
        section_texts = []
        for section in section_jsons:
            section_name = section.get('section_name', 'Unknown')
            section_text = self._section_json_to_plain_text(section, section_name)
            section_texts.append({
                'name': section_name,
                'text': section_text
            })
        
        # Compare sections pairwise for duplications
        for i in range(len(section_texts)):
            for j in range(i + 1, len(section_texts)):
                section_a = section_texts[i]
                section_b = section_texts[j]
                
                # Simple heuristic: check if significant portions are duplicated
                # Split into sentences and check for matching sequences
                sentences_a = [s.strip() for s in section_a['text'].split('.') if s.strip()]
                sentences_b = [s.strip() for s in section_b['text'].split('.') if s.strip()]
                
                # Find common sentences
                common = set(sentences_a) & set(sentences_b)
                
                # If more than 2 sentences match and they're substantial (>20 chars each)
                substantial_matches = [s for s in common if len(s) > 20]
                
                if len(substantial_matches) >= 2:
                    duplications.append({
                        'content': substantial_matches[0][:100] + '...',
                        'sections': [section_a['name'], section_b['name']],
                        'severity': 'major' if len(substantial_matches) > 4 else 'moderate',
                        'explanation': f'Found {len(substantial_matches)} matching sentences between sections'
                    })
        
        if duplications:
            logger.warning(f"[{document_id}] Found {len(duplications)} cross-section duplications")
        
        return duplications