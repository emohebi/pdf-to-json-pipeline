"""
Stage 3.5: Review Agent
Converts JSON sections to plain text and validates for:
- Sentence completeness
- Information duplication
- Logical section ordering
"""
import json
from typing import Dict, List, Any

from config.settings import MODEL_MAX_TOKENS_VALIDATION, MODEL
from src.tools import invoke_bedrock_text
from src.utils import setup_logger, StorageManager

logger = setup_logger('review_agent')


class ReviewAgent:
    """Agent to review and validate extracted section content."""
    
    def __init__(self):
        self.storage = StorageManager()
    
    def review_document(
        self,
        section_jsons: List[Dict],
        document_id: str
    ) -> Dict[str, Any]:
        """
        Review extracted document sections for quality issues.
        
        Args:
            section_jsons: List of extracted section JSONs
            document_id: Unique document identifier
        
        Returns:
            Dictionary containing review findings (empty if no issues)
        """
        logger.info(f"[{document_id}] Starting document review")
        
        try:
            # Step 1: Convert all sections to plain text
            logger.info(f"[{document_id}] Converting JSON to plain text...")
            plain_text = self._json_to_plain_text(section_jsons, document_id)
            
            # Save plain text for reference
            self.storage.save_plain_text(document_id, plain_text)
            
            # Step 2: Validate the plain text
            logger.info(f"[{document_id}] Validating content...")
            review_results = self._validate_content(
                plain_text,
                section_jsons,
                document_id
            )
            
            # Step 3: Save review results
            self.storage.save_review_results(document_id, review_results)
            
            # Log summary
            total_issues = sum(len(issues) for issues in review_results.values())
            if total_issues == 0:
                logger.info(f"[{document_id}] Review passed - no issues found")
            else:
                logger.warning(
                    f"[{document_id}] Review found {total_issues} issue(s): "
                    f"{len(review_results.get('incomplete_sentences', []))} incomplete, "
                    f"{len(review_results.get('duplications', []))} duplications, "
                    f"{len(review_results.get('order_issues', []))} order issues"
                )
            
            return review_results
            
        except Exception as e:
            logger.error(f"[{document_id}] Review failed: {e}")
            raise
    
    def _json_to_plain_text(
        self,
        section_jsons: List[Dict],
        document_id: str
    ) -> str:
        """
        Convert JSON sections to plain text using Bedrock while preserving exact wording.
        
        Args:
            section_jsons: List of section JSON objects
            document_id: Document identifier
        
        Returns:
            Plain text representation of all sections
        """
        prompt = f"""You are a precise text converter. Convert the following JSON document sections into plain text.

CRITICAL RULES:
1. Extract ALL text content from the JSON
2. Use the EXACT wording from the JSON - DO NOT paraphrase or reword
3. Preserve the original text word-for-word
4. Organize by sections in the order they appear
5. Format as readable plain text with clear section boundaries
6. Include section names as headers
7. Do not add any commentary or explanation
8. Empty fields should be omitted (don't write "N/A" or "empty")

INPUT JSON:
{json.dumps(section_jsons, indent=2)}

OUTPUT FORMAT:
=== [Section Name] ===
[Exact text content from that section]

=== [Next Section Name] ===
[Exact text content from that section]

Convert to plain text now (preserve exact wording):
"""
        
        try:
            plain_text = invoke_bedrock_text(
                prompt=prompt,
                max_tokens=MODEL_MAX_TOKENS_VALIDATION
            )
            
            logger.debug(
                f"[{document_id}] Converted to plain text: "
                f"{len(plain_text)} characters"
            )
            
            return plain_text.strip()
            
        except Exception as e:
            logger.error(f"[{document_id}] JSON to text conversion failed: {e}")
            raise
    
    def _validate_content(
        self,
        plain_text: str,
        section_jsons: List[Dict],
        document_id: str
    ) -> Dict[str, List[Dict]]:
        """
        Validate plain text for completeness, duplication, and logical flow.
        
        Args:
            plain_text: Plain text version of the document
            section_jsons: Original section JSONs for context
            document_id: Document identifier
        
        Returns:
            Dictionary with three keys:
            - incomplete_sentences: List of incomplete sentence issues
            - duplications: List of duplication issues
            - order_issues: List of section order issues
        """
        # Build section names list for context
        section_names = [
            s.get('section_name', 'Unknown Section')
            for s in section_jsons
        ]
        
        prompt = f"""You are a document quality reviewer. Analyze the following document text and identify issues.

DOCUMENT SECTIONS (in order):
{json.dumps(section_names, indent=2)}

DOCUMENT TEXT:
{plain_text}

YOUR TASK:
Carefully analyze the document and identify:

1. INCOMPLETE SENTENCES:
   - Find sentences that are cut off or incomplete. 
   - Find sentences that abruptly end mid-thought
   - Provide the exact incomplete sentence and its location (section name)

2. DUPLICATIONS:
   - Find duplicated sections
   - Provide the duplicated section name

3. ORDER ISSUES:
   - Identify sections in which text in subsections seems out of logical order. For example, the order of the paragraphs under "Task Steps" and "Notes" appears to be out of order and mix up. 
   - Check if the reading flow makes sense
   - Suggest better ordering if needed

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
      "section": "Section name where found",
      "sentence": "The exact incomplete sentence",
      "issue": "Brief description of why it's incomplete",
      "location": "Page or context information if available"
    }}
  ],
  "duplications": [
    {{
      "content": "The duplicated text (first 100 chars if long)",
      "sections": ["Section 1", "Section 2"],
      "severity": "minor|moderate|major",
      "explanation": "Why this is considered duplication"
    }}
  ],
  "order_issues": [
    {{
      "issue": "Description of the ordering problem",
      "current_order": ["Section A", "Section B"],
      "suggested_order": ["Section B", "Section A"],
      "reasoning": "Why the suggested order is better"
    }}
  ]
}}

IMPORTANT: 
- If a category has NO issues, use an empty array: []
- Return valid JSON only, no additional text
- Start with {{ and end with }}

Analyze the document and return the JSON now:
"""
        
        try:
            response = invoke_bedrock_text(
                prompt=prompt,
                max_tokens=MODEL_MAX_TOKENS_VALIDATION
            )
            
            # Clean and parse the response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            
            response = response.strip()
            
            # Parse JSON
            review_results = json.loads(response)
            
            # Validate structure
            if not isinstance(review_results, dict):
                raise ValueError("Response is not a JSON object")
            
            # Ensure all required keys exist
            for key in ['incomplete_sentences', 'duplications', 'order_issues']:
                if key not in review_results:
                    review_results[key] = []
            
            logger.debug(
                f"[{document_id}] Validation complete: "
                f"{len(review_results['incomplete_sentences'])} incomplete, "
                f"{len(review_results['duplications'])} duplications, "
                f"{len(review_results['order_issues'])} order issues"
            )
            
            return review_results
            
        except json.JSONDecodeError as e:
            logger.error(f"[{document_id}] Failed to parse validation response: {e}")
            logger.debug(f"Response was: {response[:500]}")
            # Return empty results on parse failure
            return {
                'incomplete_sentences': [],
                'duplications': [],
                'order_issues': [],
                'error': f'Failed to parse validation response: {str(e)}'
            }
        except Exception as e:
            logger.error(f"[{document_id}] Content validation failed: {e}")
            raise
