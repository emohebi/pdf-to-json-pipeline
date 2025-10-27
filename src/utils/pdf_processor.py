"""
PDF processing utilities.
"""
import pymupdf
from typing import List, Dict
from pathlib import Path
import base64

from config.settings import DPI
from src.utils.logger import setup_logger

logger = setup_logger('pdf_processor')


class PDFProcessor:
    """Handles PDF extraction and conversion to images."""
    
    def __init__(self, dpi: int = DPI):
        """
        Initialize PDF processor.
        
        Args:
            dpi: Resolution for image extraction (default from settings)
        """
        self.dpi = dpi
    
    def pdf_to_images(self, pdf_path: str, extract_with_bedrock: bool = False) -> List[Dict]:
        """
        Convert PDF pages to high-quality images with metadata.
        Also extracts text from each page using:
        1. PyMuPDF for machine-readable text
        2. Bedrock multimodal for text including text in images (optional)
        
        Args:
            pdf_path: Path to PDF file
            extract_with_bedrock: If True, also extract text using Bedrock vision model
        
        Returns:
            List of dicts containing page images and metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        logger.info(f"Bedrock text extraction: {'Enabled' if extract_with_bedrock else 'Disabled'}")
        
        # Create text output directories
        from config.settings import INTERMEDIATE_DIR
        text_output_dir = INTERMEDIATE_DIR / 'page_texts' / pdf_path.stem
        pymupdf_dir = text_output_dir / 'pymupdf'
        bedrock_dir = text_output_dir / 'bedrock'
        
        pymupdf_dir.mkdir(parents=True, exist_ok=True)
        if extract_with_bedrock:
            bedrock_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            doc = pymupdf.open(str(pdf_path))
            pages_data = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image with specified DPI
                mat = pymupdf.Matrix(self.dpi/72, self.dpi/72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Extract text with layout preservation (blocks method preserves order)
                text_blocks = page.get_text("blocks")
                # Sort blocks by vertical position (y0), then horizontal (x0) to preserve reading order
                sorted_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))
                # Extract only text content from blocks
                pymupdf_text = "\n".join(block[4] for block in sorted_blocks if len(block) > 4 and block[4].strip())
                
                # Get page dimensions
                rect = page.rect
                
                pages_data.append({
                    'page_number': page_num + 1,
                    'image': img_data,
                    'text': pymupdf_text,
                    'width': pix.width,
                    'height': pix.height,
                    'original_width': rect.width,
                    'original_height': rect.height
                })
                
                # Save PyMuPDF extracted text to file
                pymupdf_file = pymupdf_dir / f"pg{page_num + 1}.txt"
                with open(pymupdf_file, 'w', encoding='utf-8') as f:
                    f.write(pymupdf_text)
                
                logger.debug(f"Processed page {page_num + 1}/{len(doc)} (PyMuPDF) -> {pymupdf_file.name}")
                
                # Extract text using Bedrock multimodal (includes text in images)
                if extract_with_bedrock:
                    try:
                        logger.info(f"Calling Bedrock for page {page_num + 1}...")
                        bedrock_text = self._extract_text_with_bedrock(img_data, page_num + 1)
                        
                        # Log extraction stats
                        bedrock_lines = len(bedrock_text.split('\n'))
                        bedrock_chars = len(bedrock_text)
                        logger.info(
                            f"Bedrock extracted {bedrock_chars} characters "
                            f"({bedrock_lines} lines) for page {page_num + 1}"
                        )
                        
                        # Save Bedrock extracted text to file
                        bedrock_file = bedrock_dir / f"pg{page_num + 1}.txt"
                        with open(bedrock_file, 'w', encoding='utf-8') as f:
                            f.write(bedrock_text)
                        
                        logger.debug(f"Extracted text with Bedrock for page {page_num + 1} -> {bedrock_file.name}")
                    except Exception as e:
                        logger.error(f"Bedrock extraction failed for page {page_num + 1}: {e}")
                        # Create empty file to indicate failure
                        bedrock_file = bedrock_dir / f"pg{page_num + 1}.txt"
                        with open(bedrock_file, 'w', encoding='utf-8') as f:
                            f.write(f"[ERROR: Bedrock extraction failed - {str(e)}]\n")
            
            doc.close()
            logger.info(f"Successfully extracted {len(pages_data)} pages")
            logger.info(f"PyMuPDF texts saved to: {pymupdf_dir}")
            if extract_with_bedrock:
                logger.info(f"Bedrock texts saved to: {bedrock_dir}")
            
            return pages_data
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def _extract_text_with_bedrock(self, img_data: bytes, page_num: int) -> str:
        """
        Extract text from page image using Bedrock multimodal model.
        This captures ALL text including text embedded in images.
        
        Args:
            img_data: Page image as PNG bytes
            page_num: Page number for logging
        
        Returns:
            Extracted text in reading order
        """
        from src.tools.bedrock_vision import invoke_bedrock_vision
        
        # Encode image to base64
        img_b64 = base64.b64encode(img_data).decode('utf-8')
        
        logger.debug(f"Page {page_num}: Image size = {len(img_data)} bytes, Base64 size = {len(img_b64)} chars")
        
        # Enhanced prompt that explicitly tells the model to analyze the image
        prompt = """I am showing you an image of a document page. Your ONLY job is to extract ALL visible text from this image.

        LOOK AT THE IMAGE CAREFULLY. This is a visual image that contains:
        - Regular printed text
        - Text inside icons, symbols, and graphical elements
        - Text in diagrams, charts, and technical drawings
        - Text labels on images
        - Any handwritten or stylized text
        - Text in tables, forms, and structured layouts

        YOUR TASK:
        Read the image from TOP to BOTTOM, LEFT to RIGHT, and transcribe EVERY piece of text you see.

        CRITICAL RULES:
        1. LOOK AT THE IMAGE - do not ignore any visual elements
        2. Extract text EXACTLY as written - character by character
        3. Include text that appears IN images, icons, warning signs, diagrams
        4. Include text that is part of graphical elements
        5. Include labels, annotations, measurements in diagrams
        6. Preserve the reading order (top→bottom, left→right)
        7. DO NOT add any explanations or descriptions
        8. DO NOT skip text just because it's in an image or icon
        9. DO NOT summarize - copy the exact words
        10. If you see a warning icon with text "DANGER", write "DANGER"
        11. If you see a diagram with labels "Part A", "12mm", write those exact labels

        WHAT TO INCLUDE:
        ✓ Main document text
        ✓ Text inside warning triangles, caution signs, safety icons
        ✓ Labels on diagrams and technical drawings
        ✓ Measurements and dimensions shown in images
        ✓ Text in photo captions
        ✓ Text overlaid on any images
        ✓ Headers, footers, page numbers
        ✓ Text in tables and forms
        ✓ Any visible text regardless of where it appears

        OUTPUT FORMAT:
        Return ONLY the extracted text. No explanations. No descriptions. No markdown.
        Start immediately with the first text you see when reading the image from top to bottom.

        Begin extracting now:"""
        
        try:
            logger.debug(f"Page {page_num}: Invoking Bedrock with prompt length = {len(prompt)} chars")
            
            # Invoke Bedrock with vision capability
            response = invoke_bedrock_vision(
                image_data=img_b64,
                prompt=prompt,
                max_tokens=16000
            )
            
            logger.debug(f"Page {page_num}: Bedrock response length = {len(response)} chars")
            
            # Log first 200 chars of response for debugging
            preview = response[:200].replace('\n', '\\n')
            logger.debug(f"Page {page_num}: Response preview: {preview}...")
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Bedrock vision API error for page {page_num}: {e}")
            raise
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract embedded images from PDF.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of extracted images with metadata
        """
        pdf_path = Path(pdf_path)
        doc = pymupdf.open(str(pdf_path))
        
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                
                images.append({
                    'page_number': page_num + 1,
                    'image_index': img_index,
                    'image_data': base_image['image'],
                    'extension': base_image['ext'],
                    'width': base_image['width'],
                    'height': base_image['height']
                })
        
        doc.close()
        logger.info(f"Extracted {len(images)} embedded images from {pdf_path.name}")
        
        return images
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict:
        """
        Extract metadata from PDF.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dictionary of PDF metadata
        """
        pdf_path = Path(pdf_path)
        doc = pymupdf.open(str(pdf_path))
        
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
            'creation_date': doc.metadata.get('creationDate', ''),
            'mod_date': doc.metadata.get('modDate', ''),
            'page_count': len(doc),
            'file_size': pdf_path.stat().st_size
        }
        
        doc.close()
        
        return metadata