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
                'incomplete_sentences': [],
                'duplications': [],
                'order_issues': [],
                'missing_information': []
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
            
            # Step 2: Check for cross-section duplications
            logger.info(f"[{document_id}] Checking for cross-section duplications...")
            cross_section_duplications = self._check_cross_section_duplications(
                section_jsons,
                document_id
            )
            aggregated_results['duplications'].extend(cross_section_duplications)
            
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
                if key.startswith('_') or key == 'image':
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
Carefully look at the PAGE IMAGES first, then compare the extracted text against what you see in the images. Identify:

1. INCOMPLETE SENTENCES:
   - Look at the images: Are there sentences that were cut off mid-extraction?
   - Compare the extracted text to what you see in the images
   - Find sentences in the extracted text that end abruptly or don't match the complete sentence visible in the image
   - Report sentences where the extraction stopped but the image shows more text
   - Example: If image shows "Remove bolts A, B, and C carefully" but extracted text says "Remove bolts A, B, and" - this is incomplete
   
   CRITICAL - Only report if ACTUALLY incomplete:
   - DO NOT report if the extracted sentence is complete and correct (even if worded differently than expected)
   - DO NOT report spelling errors or grammar issues - only report if text is truly cut off
   - DO NOT report if the sentence makes sense and is complete, even if it could be more detailed
   - ONLY report if the sentence clearly stops mid-thought with missing words that are visible in the image
   
   Examples of ACTUAL incomplete sentences to report:
   - ✓ "Remove bolts A, B, and" (image shows "Remove bolts A, B, and C carefully")
   - ✓ "Ensure the safety" (image shows "Ensure the safety valve is closed")
   - ✓ "Inspect for damage or" (image shows "Inspect for damage or wear")
   
   Examples of complete sentences NOT to report:
   - ✗ "You must complete a risk assessment" (complete sentence, even if image has more detail)
   - ✗ "Check oil level" (complete instruction, even if brief)
   - ✗ Sentences with spelling errors but complete meaning

2. DUPLICATIONS (within this section):
   - Look at the images: Is any content repeated in the visual layout?
   - Check if the extracted text has duplicated paragraphs, sentences, or content blocks
   - Compare with images to verify if duplication is real or an extraction error
   - Report actual duplications you can see both in images and extracted text

3. ORDER ISSUES (within this section):
   - Look at the images: What is the reading order on the page (top to bottom, left to right)?
   - Compare the sequence of content in the extracted text vs. the visual order in the images
   - Identify if paragraphs, steps, or items appear in wrong order in extracted text
   - For example: if image shows "Step 1, Step 2, Step 3" but extracted text shows "Step 1, Step 3, Step 2"
   - Check if subsections like "Task Steps" and "Notes" are mixed up compared to their visual positions

4. MISSING INFORMATION (within this section):
   - Look at the images: What information is visible in the section?
   - Compare what you see in images with what appears in extracted text
   - Identify ENTIRE pieces of content that are completely absent from the extraction
   - Examples: entire steps that don't appear at all, whole paragraphs missing, bullet points completely omitted
   
   CRITICAL - Avoid overlap with incomplete sentences:
   - DO NOT report here if part of a sentence was extracted (that's an incomplete sentence, not missing information)
   - ONLY report if an ENTIRE piece of content (step, paragraph, bullet point) is completely missing
   - Focus on structural omissions (missing steps, missing sections, missing list items)
   - If the sentence/step EXISTS in extracted text but is incomplete, that belongs in "incomplete_sentences", NOT here
   
   IMPORTANT - What NOT to report as missing:
   - DO NOT report missing table headers or table structure formatting (e.g., "Part No.", "Qty", "Description")
   - DO NOT report text that appears only inside icons, symbols, or graphical elements (e.g., text in flame icons, warning triangles)
   - DO NOT report missing visual formatting or layout structures
   - DO NOT report partial extraction of sentences (use incomplete_sentences for that)
   - ONLY report missing substantive content (entire steps, paragraphs, bullet points, data values that are COMPLETELY absent)
   
   Examples of what TO report as missing information:
   - ✓ "Step 2.3: Check fuel level" - the ENTIRE step is missing from extracted text
   - ✓ A complete paragraph of safety instructions that doesn't appear at all
   - ✓ An entire bullet point about PPE requirements that's not in the extracted text
   - ✓ A data row from a table that's completely omitted (the actual data, not the headers)
   
   Examples of what NOT to report as missing (belongs elsewhere):
   - ✗ "Remove bolts A, B, and" (incomplete - belongs in incomplete_sentences)
   - ✗ Table headers like "Part No.", "Resource Description", "Qty"
   - ✗ Text inside icons (e.g., "Hold Point" text in a flame icon)
   - ✗ Text inside warning symbols or graphical elements
   - ✗ Column headers, table formatting, or visual structure elements
   
   Rule of thumb: 
   - If it was extracted but incomplete → incomplete_sentences
   - If it wasn't extracted at all → missing_information

VALIDATION METHODOLOGY:
1. LOOK AT THE IMAGES FIRST - These are the original PDF pages, the source of truth
2. Identify where section "{section_name}" appears in the images (look for the heading/title)
3. Read through the section content as it appears visually in the images
4. Compare what you see in images with the extracted text provided
5. Report any discrepancies, incompleteness, duplications, or ordering issues

CRITICAL RULES:
- The IMAGES are the TRUTH - validate extracted text against them
- Only report issues you can VERIFY by looking at the images
- Quote EXACT text from the extracted text when reporting issues
- Reference what you see in the images when explaining issues
- If the images show complete text but extraction is incomplete, report it in incomplete_sentences
- If images show different order than extracted text, report it in order_issues
- If entire content blocks are missing (not extracted at all), report in missing_information
- Only report ACTUAL issues you can see by comparing images to text
- If there are no issues in a category, that category should be empty
- Consider the technical nature of the document (safety procedures)

AVOID FALSE POSITIVES:
- Incomplete sentences: Only report if sentence is ACTUALLY cut off mid-word or mid-phrase (not just brief)
- Missing information: Only report if ENTIRE content block is absent (not if partially extracted)
- Do NOT overlap: If something was partially extracted → incomplete_sentences. If not extracted at all → missing_information
- Do NOT report complete sentences as incomplete, even if they could be more detailed

OUTPUT FORMAT:
Return ONLY a valid JSON object with this exact structure (no markdown, no code blocks):
{{
  "incomplete_sentences": [
    {{
      "section": "{section_name}",
      "sentence": "The exact incomplete sentence from extracted text",
      "issue": "What you see in the image that's missing from extracted text",
      "location": "Page number or context from image"
    }}
  ],
  "duplications": [
    {{
      "content": "The duplicated text (first 100 chars if long)",
      "sections": ["{section_name}"],
      "severity": "minor|moderate|major",
      "explanation": "Verified in images - content appears multiple times"
    }}
  ],
  "order_issues": [
    {{
      "issue": "Description of the ordering problem",
      "section": "{section_name}",
      "current_order": ["Order in extracted text"],
      "suggested_order": ["Order as shown in images"],
      "reasoning": "Visual order in images differs from extracted text order"
    }}
  ],
  "missing_information": [
    {{
      "section": "{section_name}",
      "missing_content": "Description of what is missing (e.g., 'Step 2.3', 'Safety warning paragraph')",
      "issue": "What you see in the image that is completely absent from extracted text",
      "location": "Page number or context from image where it appears"
    }}
  ]
}}

IMPORTANT: 
- If a category has NO issues, use an empty array: []
- Return valid JSON only, no additional text
- Start with {{ and end with }}
- Base your analysis on what you SEE in the images

Look at the images and validate the extracted text now:
"""
        else:
            # Fallback prompt when no images available (text-only validation)
            prompt = f"""You are a document quality reviewer. Analyze the following section text and identify issues.

SECTION NAME: {section_name}

SECTION TEXT:
{section_plain_text}

YOUR TASK:
Carefully analyze this section and identify:

1. INCOMPLETE SENTENCES:
   - Find sentences that are cut off or incomplete
   - Find sentences that abruptly end mid-thought
   - Provide the exact incomplete sentence and its location (section name)

2. DUPLICATIONS (within this section):
   - Find duplicated content within this section
   - Provide the duplicated text

3. ORDER ISSUES (within this section):
   - Identify text in subsections that seems out of logical order
   - For example, paragraphs under "Task Steps" and "Notes" that appear mixed up
   - Check if the reading flow makes sense
   - Suggest better ordering if needed

4. MISSING INFORMATION (within this section):
   - Note: Without images, this check is limited to logical gaps
   - Identify if there are references to content that doesn't appear (e.g., "See Step 2.3" but no Step 2.3)
   - Check for numbering gaps (Step 1, Step 3 - where is Step 2?)
   - Report any logical inconsistencies suggesting missing content

CRITICAL RULES:
- Be thorough and precise
- Quote EXACT text from the document when reporting issues
- Only report ACTUAL issues, don't invent problems
- If there are no issues in a category, that category should be empty
- Consider the technical nature of the document (safety procedures)

OUTPUT FORMAT:
Return ONLY a valid JSON object with this exact structure (no markdown, no code blocks):
{{
  "incomplete_sentences": [
    {{
      "section": "{section_name}",
      "sentence": "The exact incomplete sentence",
      "issue": "Brief description of why it's incomplete",
      "location": "Context information if available"
    }}
  ],
  "duplications": [
    {{
      "content": "The duplicated text (first 100 chars if long)",
      "sections": ["{section_name}"],
      "severity": "minor|moderate|major",
      "explanation": "Why this is considered duplication"
    }}
  ],
  "order_issues": [
    {{
      "issue": "Description of the ordering problem",
      "section": "{section_name}",
      "current_order": ["Item A", "Item B"],
      "suggested_order": ["Item B", "Item A"],
      "reasoning": "Why the suggested order is better"
    }}
  ],
  "missing_information": [
    {{
      "section": "{section_name}",
      "missing_content": "Description of what appears to be missing",
      "issue": "Evidence of missing content (e.g., numbering gaps, broken references)",
      "location": "Context information if available"
    }}
  ]
}}

IMPORTANT: 
- If a category has NO issues, use an empty array: []
- Return valid JSON only, no additional text
- Start with {{ and end with }}

Analyze the section and return the JSON now:
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
                for key in ['incomplete_sentences', 'duplications', 'order_issues', 'missing_information']:
                    if key not in section_review:
                        section_review[key] = []
                
                logger.debug(
                    f"[{document_id}] Section '{section_name}' validation: "
                    f"{len(section_review['incomplete_sentences'])} incomplete, "
                    f"{len(section_review['duplications'])} duplications, "
                    f"{len(section_review['order_issues'])} order issues, "
                    f"{len(section_review['missing_information'])} missing info"
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
                        'incomplete_sentences': [],
                        'duplications': [],
                        'order_issues': [],
                        'error': f'Failed to parse validation response: {str(e)}'
                    }
            except Exception as e:
                logger.error(f"[{document_id}] Section validation failed for {section_name}: {e}")
                if counter > 0:
                    logger.info(f"Trying {counter} more times .. ")
                    counter -= 1
                    loop = True
                else:
                    return {
                        'incomplete_sentences': [],
                        'duplications': [],
                        'order_issues': [],
                        'error': str(e)
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