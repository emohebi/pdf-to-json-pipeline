"""
Image description generator for PDF to JSON pipeline.
Uses Bedrock multimodal to generate short descriptions of images for accurate mapping.
Save this as: src/utils/image_descriptor.py
"""
import base64
from typing import List, Dict, Tuple
from pathlib import Path
import json

from src.utils.logger import setup_logger

logger = setup_logger('image_descriptor')


class ImageDescriptor:
    """Generate descriptions for images to enable accurate mapping in JSON."""
    
    def __init__(self):
        """Initialize the image descriptor."""
        self.cache = {}  # Cache descriptions to avoid redundant API calls
    
    def generate_image_descriptions(
        self, 
        images: List[Dict],
        document_id: str = None
    ) -> Dict[str, str]:
        """
        Generate descriptions for all images and return as a dictionary.
        
        Args:
            images: List of image dictionaries with 'image_path' and image data
            document_id: Optional document ID for logging
        
        Returns:
            Dictionary mapping description to image path
            Example: {"Safety warning icon with flame symbol": "Media/doc/page1_img1.png"}
        """
        if not images:
            return {}
        
        logger.info(f"[{document_id}] Generating descriptions for {len(images)} images...")
        
        descriptions_dict = {}
        
        for idx, img in enumerate(images, 1):
            try:
                # Check cache first
                cache_key = f"{document_id}_{img['page_number']}_{idx}"
                if cache_key in self.cache:
                    description = self.cache[cache_key]
                else:
                    # Generate description using Bedrock
                    description = self._generate_single_description(
                        img, 
                        idx, 
                        document_id
                    )
                    self.cache[cache_key] = description
                
                # Add to dictionary
                # Ensure unique keys by appending index if needed
                if description in descriptions_dict:
                    description = f"{description} (image {idx})"
                
                descriptions_dict[description] = img['image_path']
                
                logger.debug(
                    f"[{document_id}] Image {idx}: '{description}' -> {img['image_path']}"
                )
                
            except Exception as e:
                logger.error(f"[{document_id}] Failed to describe image {idx}: {e}")
                # Fallback to generic description
                fallback_desc = f"Image on page {img['page_number']} position {idx}"
                descriptions_dict[fallback_desc] = img['image_path']
        
        logger.info(f"[{document_id}] Generated {len(descriptions_dict)} descriptions")
        return descriptions_dict
    
    def generate_section_image_descriptions(
        self,
        section_images: List[Dict],
        section_name: str,
        section_type: str,
        document_id: str = None
    ) -> Dict[str, str]:
        """
        Generate descriptions for images in a specific section with context.
        
        Args:
            section_images: List of images in this section
            section_name: Name of the section
            section_type: Type of section (task_activities, safety, etc.)
            document_id: Document ID for logging
        
        Returns:
            Dictionary mapping description to image path
        """
        if not section_images:
            return {}
        
        logger.info(
            f"[{document_id}] Generating descriptions for {len(section_images)} images "
            f"in section '{section_name}' ({section_type})"
        )
        
        descriptions_dict = {}
        len_images = len(section_images)
        for idx, img in enumerate(section_images, 1):
            try:
                # Generate context-aware description based on section type
                description = self._generate_contextual_description(
                    img,
                    idx,
                    section_type,
                    section_name,
                    description_len = 30 if len_images <=30 else 10,
                    document_id = document_id
                )
                
                # Ensure unique keys
                if description in descriptions_dict:
                    description = f"{description} (item {idx})"
                
                descriptions_dict[description] = img['image_path']
                
            except Exception as e:
                logger.error(
                    f"[{document_id}] Failed to describe image {idx} in {section_name}: {e}"
                )
                fallback_desc = f"{section_type} image {idx} on page {img['page_number']}"
                descriptions_dict[fallback_desc] = img['image_path']
        
        return descriptions_dict
    
    def _generate_single_description(
        self, 
        img: Dict, 
        index: int,
        document_id: str = None
    ) -> str:
        """
        Generate a description for a single image using Bedrock.
        
        Args:
            img: Image dictionary with path and data
            index: Image index
            document_id: Document ID for context
        
        Returns:
            Short descriptive text for the image
        """
        from src.tools.bedrock_vision import invoke_bedrock_vision
        
        # Load image file
        image_path = img.get('local_path')
        if not image_path:
            # If local_path not available, construct from image_path
            from config.settings import OUTPUT_DIR
            image_path = OUTPUT_DIR / img['image_path']
        
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Encode to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create prompt for description
            prompt = """Analyze this image and provide a SHORT, SPECIFIC description (max 10 words).

Focus on:
- What type of image it is (photo, diagram, icon, chart, etc.)
- The main subject or content
- Any visible text or labels
- For icons: describe the symbol (e.g., "flame hazard warning icon")
- For diagrams: main components shown (e.g., "valve assembly diagram with parts labeled")
- For photos: what's depicted (e.g., "maintenance worker inspecting equipment")

Examples of good descriptions:
- "Safety helmet PPE icon"
- "Step 3 valve installation diagram"
- "Warning triangle with electrical hazard symbol"
- "Pump assembly cross-section showing components"
- "Before and after comparison photos"

Return ONLY the short description, nothing else:"""
            
            # Get description from Bedrock
            description = invoke_bedrock_vision(
                image_data=image_b64,
                prompt=prompt,
                max_tokens=100
            )
            
            # Clean and truncate description
            description = description.strip()
            if len(description) > 100:
                description = description[:97] + "..."
            
            return description
            
        except Exception as e:
            logger.error(f"Error generating description for image {index}: {e}")
            return f"Image {index} on page {img['page_number']}"
    
    def _generate_contextual_description(
        self,
        img: Dict,
        index: int,
        section_type: str,
        section_name: str,
        description_len = 30,
        document_id: str = None
    ) -> str:
        """
        Generate description with section context for better accuracy.
        
        Args:
            img: Image dictionary
            index: Image index in section
            section_type: Type of section
            section_name: Name of section
            document_id: Document ID
        
        Returns:
            Contextual description
        """
        from src.tools.bedrock_vision import invoke_bedrock_vision
        
        # Load image
        image_path = img.get('local_path')
        if not image_path:
            from config.settings import OUTPUT_DIR
            image_path = OUTPUT_DIR / img['image_path']
        
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Context-specific prompts
            context_prompts = {
                'task_activities': """This image is from a task/activity section. 
Describe it focusing on: step number if visible, equipment shown, action depicted, or diagram type.""",
                
                'safety': """This image is from a safety section.
Describe safety icons, warning symbols, or hazard indicators.""",
                
                'additional_ppe_required': """This image shows required PPE equipment.
Describe the specific PPE item shown.""",
                
                'material_risks_and_controls': """This image relates to risks and controls.
Describe hazard symbols or control measures shown.""",
                
                'attached_images': """This is an attached reference image.
Describe what it shows or depicts."""
            }
            
            context = context_prompts.get(section_type, "Describe this image briefly.")
            
            prompt = f"""Analyze this image from the '{section_name}' section.

{context}

Provide a detailed, SPECIFIC description (Maximum {description_len} words) which clearly describes the image for a computer vision model. 
If you see any text or object's colors in the image please explain them for better matching accuracy.
Return ONLY the description, which MUST be less than {description_len} words long:"""
            
            description = invoke_bedrock_vision(
                image_data=image_b64,
                prompt=prompt,
                max_tokens=1000
            )
            
            return description.strip()
            
        except Exception as e:
            logger.error(f"Error generating contextual description: {e}")
            return f"{section_type} image {index}"


def create_image_descriptions_for_document(
    extracted_images: List[Dict],
    document_id: str,
    save_to_file: bool = True
) -> Dict[str, str]:
    """
    Convenience function to generate descriptions for all document images.
    
    Args:
        extracted_images: List of extracted images from PDF
        document_id: Document ID
        save_to_file: Whether to save descriptions to a JSON file
    
    Returns:
        Dictionary of descriptions to paths
    """
    descriptor = ImageDescriptor()
    descriptions = descriptor.generate_image_descriptions(
        extracted_images,
        document_id
    )
    
    if save_to_file:
        from config.settings import INTERMEDIATE_DIR
        output_file = INTERMEDIATE_DIR / f"{document_id}_image_descriptions.json"
        with open(output_file, 'w') as f:
            json.dump(descriptions, f, indent=2)
        logger.info(f"Saved image descriptions to {output_file}")
    
    return descriptions


def create_section_image_descriptions(
    section_images: List[Dict],
    section_name: str,
    section_type: str,
    document_id: str = None
) -> Dict[str, str]:
    """
    Convenience function for section-specific image descriptions.
    
    Args:
        section_images: Images in this section
        section_name: Section name
        section_type: Section type
        document_id: Document ID
    
    Returns:
        Dictionary of descriptions to paths
    """
    descriptor = ImageDescriptor()
    return descriptor.generate_section_image_descriptions(
        section_images,
        section_name,
        section_type,
        document_id
    )
