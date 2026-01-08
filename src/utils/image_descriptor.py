"""
Image description generator for PDF to JSON pipeline.
ENHANCED: Uses position-based mapping for more accurate image placement.
Uses Bedrock multimodal to generate short descriptions of images for accurate mapping.
"""
import base64
from typing import List, Dict, Tuple
from pathlib import Path
import json

from src.utils.logger import setup_logger

logger = setup_logger('image_descriptor')


# ============================================================================
# POSITION HELPER FUNCTIONS
# ============================================================================

def get_position_grid(y_percent: float, x_percent: float) -> str:
    """
    Convert position percentages to a human-readable grid location.
    
    Args:
        y_percent: Vertical position as percentage (0-100, top to bottom)
        x_percent: Horizontal position as percentage (0-100, left to right)
    
    Returns:
        Grid position string like "top-left", "middle-center", etc.
    """
    # Vertical position
    if y_percent < 33:
        v_pos = "top"
    elif y_percent < 66:
        v_pos = "middle"
    else:
        v_pos = "bottom"
    
    # Horizontal position
    if x_percent < 33:
        h_pos = "left"
    elif x_percent < 66:
        h_pos = "center"
    else:
        h_pos = "right"
    
    return f"{v_pos}-{h_pos}"


def get_row_column(y_percent: float, x_percent: float) -> Tuple[int, int]:
    """
    Convert position to row/column indices (1-indexed).
    Divides page into 3x3 grid.
    """
    row = min(3, max(1, int(y_percent / 33.33) + 1))
    col = min(3, max(1, int(x_percent / 33.33) + 1))
    return row, col


# ============================================================================
# IMAGE DESCRIPTOR CLASS
# ============================================================================

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
        document_id: str = None,
        save_to_file: bool = True
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
                    description_len = 50 if len_images <=70 else 70,
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
        if save_to_file:
            from config.settings import IMG_DESC_DIR
            output_file = IMG_DESC_DIR / f"{section_name.replace(' ', '_')}_img_desc.json"
            with open(output_file, 'w') as f:
                json.dump(descriptions_dict, f, indent=2)
            logger.info(f"Saved image descriptions to {output_file}")
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

Provide a detailed, SPECIFIC description (Maximum {description_len} words) which clearly describes the image for a computer vision model matching. 
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


# ============================================================================
# POSITIONAL IMAGE MAPPING (NEW - MORE ROBUST APPROACH)
# ============================================================================

def create_positional_image_mapping(
    section_images: List[Dict],
    section_name: str,
    section_type: str,
    document_id: str = None,
    page_dimensions: Dict[int, Tuple[float, float]] = None
) -> List[Dict]:
    """
    Create position-based image mapping for more accurate matching.
    
    This approach provides:
    1. Page number (exact)
    2. Grid position (top-left, middle-center, etc.)
    3. Y-position percentage (for ordering)
    4. Brief description (for disambiguation)
    
    Args:
        section_images: Images in this section
        section_name: Section name
        section_type: Section type
        document_id: Document ID
        page_dimensions: Dict mapping page_num to (width, height)
    
    Returns:
        List of image info dicts with positional data
    """
    if not section_images:
        return []
    
    logger.info(
        f"[{document_id}] Creating positional mapping for {len(section_images)} images "
        f"in section '{section_name}'"
    )
    
    image_mappings = []
    descriptor = ImageDescriptor()
    
    for idx, img in enumerate(section_images, 1):
        try:
            page_num = img['page_number']
            y_pos = img.get('y_position', 0)
            x_pos = img.get('x_position', 0)
            y_bottom = img.get('y_bottom', y_pos)
            
            # Get page dimensions (default to standard page if not provided)
            if page_dimensions and page_num in page_dimensions:
                page_width, page_height = page_dimensions[page_num]
            else:
                # Default A4-ish dimensions at 150 DPI
                page_width = 595  # ~8.5 inches
                page_height = 842  # ~11 inches
            
            # Calculate position percentages
            y_percent = (y_pos / page_height * 100) if page_height > 0 else 50
            x_percent = (x_pos / page_width * 100) if page_width > 0 else 50
            
            # Get grid position
            grid_pos = get_position_grid(y_percent, x_percent)
            row, col = get_row_column(y_percent, x_percent)
            
            # Generate a brief description (shorter than before for disambiguation only)
            brief_desc = descriptor._generate_contextual_description(
                img, idx, section_type, section_name,
                description_len=12,  # Shorter description
                document_id=document_id
            )
            
            image_info = {
                'index': idx,
                'page': page_num,
                'grid': grid_pos,
                'row': row,
                'col': col,
                'y_percent': round(y_percent, 1),
                'x_percent': round(x_percent, 1),
                'description': brief_desc,
                'path': img['image_path'],
                'width': img.get('width', 0),
                'height': img.get('height', 0)
            }
            
            image_mappings.append(image_info)
            
            logger.debug(
                f"[{document_id}] Image {idx}: page {page_num}, {grid_pos}, "
                f"Y={y_percent:.1f}% -> {img['image_path']}"
            )
            
        except Exception as e:
            logger.error(f"[{document_id}] Failed to map image {idx}: {e}")
            # Fallback mapping
            image_mappings.append({
                'index': idx,
                'page': img.get('page_number', 0),
                'grid': 'unknown',
                'row': 2,
                'col': 2,
                'y_percent': 50,
                'x_percent': 50,
                'description': f"Image {idx}",
                'path': img['image_path'],
                'width': img.get('width', 0),
                'height': img.get('height', 0)
            })
    
    # Sort by page then by Y position for consistent ordering
    image_mappings.sort(key=lambda x: (x['page'], x['y_percent']))
    
    # Re-index after sorting
    for i, mapping in enumerate(image_mappings, 1):
        mapping['sorted_index'] = i
    
    logger.info(f"[{document_id}] Created {len(image_mappings)} positional mappings")
    return image_mappings


def format_image_mapping_for_prompt(image_mappings: List[Dict]) -> str:
    """
    Format image mappings into a clear, structured prompt section.
    
    Args:
        image_mappings: List of positional image mappings
    
    Returns:
        Formatted string for inclusion in extraction prompt
    """
    if not image_mappings:
        return ""
    
    lines = ["\n\nIMAGES IN THIS SECTION (ordered by page and vertical position):"]
    lines.append("=" * 80)
    
    current_page = None
    for img in image_mappings:
        if img['page'] != current_page:
            current_page = img['page']
            lines.append(f"\n--- PAGE {current_page} ---")
        
        desc_short = img['description'][:35] + "..." if len(img['description']) > 38 else img['description']
        lines.append(
            f"  [{img['sorted_index']:2d}] Position: {img['grid']:13s} (Y:{img['y_percent']:4.0f}%) | {desc_short}"
        )
        lines.append(f"       PATH: {img['path']}")
    
    lines.append("\n" + "=" * 80)
    lines.append("""
IMAGE MATCHING INSTRUCTIONS:
============================
To correctly place images in the JSON, follow this process:

1. IDENTIFY BY PAGE: First, determine which page the image appears on
2. IDENTIFY BY POSITION: Look at where the image sits on the page:
   - top-left, top-center, top-right (upper third)
   - middle-left, middle-center, middle-right (middle third)  
   - bottom-left, bottom-center, bottom-right (lower third)
3. USE Y% FOR ORDERING: Images with lower Y% appear higher on the page
4. VERIFY WITH DESCRIPTION: Use the description to confirm the match
5. COPY THE EXACT PATH: Use the PATH shown above exactly as written

EXAMPLE:
- You see a diagram in the middle-right area of page 5
- Find in list above: Page 5, Position: middle-right
- Verify description matches what you see
- Copy that exact PATH to the "image" field

CRITICAL RULES:
- NEVER invent or modify paths - use ONLY paths from the list above
- If you cannot find a matching image, leave the image field empty
- Multiple images on same page are distinguished by Y% (vertical position)
""")
    
    return "\n".join(lines)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

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
    LEGACY: Returns dict format for backward compatibility.
    
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


def create_section_image_mapping(
    section_images: List[Dict],
    section_name: str,
    section_type: str,
    document_id: str = None,
    pages_data: List[Dict] = None
) -> Tuple[List[Dict], str]:
    """
    NEW: Create positional image mapping with formatted prompt text.
    This is the recommended function for accurate image placement.
    
    Args:
        section_images: Images in this section
        section_name: Section name
        section_type: Section type
        document_id: Document ID
        pages_data: List of page data (to get dimensions)
    
    Returns:
        Tuple of (image_mappings list, formatted_prompt_text)
    """
    # Extract page dimensions if available
    page_dimensions = {}
    if pages_data:
        for page in pages_data:
            page_num = page.get('page_number', 0)
            page_dimensions[page_num] = (
                page.get('original_width', 595),
                page.get('original_height', 842)
            )
    
    # Create positional mappings
    mappings = create_positional_image_mapping(
        section_images,
        section_name,
        section_type,
        document_id,
        page_dimensions
    )
    
    # Format for prompt
    prompt_text = format_image_mapping_for_prompt(mappings)
    
    return mappings, prompt_text