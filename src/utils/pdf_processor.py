"""
ENHANCED pdf_processor.py with precise text block positioning for image context.
This version extracts text WITH coordinates for better image-to-text matching.
"""
import pymupdf
from typing import List, Dict, Tuple
from pathlib import Path
import base64

from config.settings import DPI
from src.utils.logger import setup_logger

logger = setup_logger('pdf_processor')


class PDFProcessor:
    """Handles PDF extraction with precise text positioning."""
    
    def __init__(self, dpi: int = DPI):
        self.dpi = dpi
    
    def pdf_to_images(self, pdf_path: str, extract_with_bedrock: bool = False) -> List[Dict]:
        """
        Convert PDF pages to images with text AND coordinates.
        Enhanced to include text block positions for accurate image context matching.
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        
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
                
                # Convert page to image
                mat = pymupdf.Matrix(self.dpi/72, self.dpi/72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # ENHANCED: Extract text with precise block positions
                text_blocks = page.get_text("dict")["blocks"]
                
                # Build text and position data
                positioned_text = []
                full_text_parts = []
                
                for block in text_blocks:
                    if block.get("type") == 0:  # Text block
                        x0, y0, x1, y1 = block["bbox"]
                        
                        # Extract text from lines
                        block_text = ""
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                block_text += span.get("text", "")
                            block_text += "\n"
                        
                        block_text = block_text.strip()
                        if block_text:
                            positioned_text.append({
                                'text': block_text,
                                'bbox': [x0, y0, x1, y1],
                                'y_center': (y0 + y1) / 2
                            })
                            full_text_parts.append(block_text)
                
                # Sort by Y position for reading order
                positioned_text.sort(key=lambda b: b['y_center'])
                pymupdf_text = "\n".join(full_text_parts)
                
                # Get page dimensions
                rect = page.rect
                
                pages_data.append({
                    'page_number': page_num + 1,
                    'image': img_data,
                    'text': pymupdf_text,
                    'text_blocks': positioned_text,  # NEW: Include positioned text
                    'width': pix.width,
                    'height': pix.height,
                    'original_width': rect.width,
                    'original_height': rect.height
                })
                
                # Save PyMuPDF extracted text
                pymupdf_file = pymupdf_dir / f"pg{page_num + 1}.txt"
                with open(pymupdf_file, 'w', encoding='utf-8') as f:
                    f.write(pymupdf_text)
                
                logger.debug(f"Processed page {page_num + 1}/{len(doc)} with {len(positioned_text)} text blocks")
                
                # Bedrock extraction (if enabled)
                if extract_with_bedrock:
                    try:
                        bedrock_text = self._extract_text_with_bedrock(img_data, page_num + 1)
                        bedrock_file = bedrock_dir / f"pg{page_num + 1}.txt"
                        with open(bedrock_file, 'w', encoding='utf-8') as f:
                            f.write(bedrock_text)
                    except Exception as e:
                        logger.error(f"Bedrock extraction failed for page {page_num + 1}: {e}")
            
            doc.close()
            logger.info(f"Successfully extracted {len(pages_data)} pages with positioned text")
            
            return pages_data
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def _extract_text_with_bedrock(self, img_data: bytes, page_num: int) -> str:
        """Extract text using Bedrock vision model."""
        from src.tools.bedrock_vision import invoke_bedrock_vision
        
        img_b64 = base64.b64encode(img_data).decode('utf-8')
        
        prompt = """Extract ALL visible text from this image exactly as written.
Include text in diagrams, icons, labels, and any graphical elements.
Return only the text in reading order (top to bottom, left to right)."""
        
        try:
            response = invoke_bedrock_vision(
                image_data=img_b64,
                prompt=prompt,
                max_tokens=16000
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Bedrock vision API error for page {page_num}: {e}")
            raise
    
    def extract_images_from_pdf(self, pdf_path: str, output_dir: Path = None) -> List[Dict]:
        """
        Extract embedded images from PDF with precise Y-positions.
        FILTERS OUT logos. Returns images with accurate positioning.
        """
        pdf_path = Path(pdf_path)
        doc = pymupdf.open(str(pdf_path))
        
        if output_dir is None:
            from config.settings import OUTPUT_DIR
            output_dir = OUTPUT_DIR / 'Media' / pdf_path.stem
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # First pass: Collect all images
        all_images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            rect = page.rect
            page_height = rect.height
            
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                
                # Get precise image position
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    y_position = img_rects[0].y0
                    y_bottom = img_rects[0].y1
                    img_height = y_bottom - y_position
                    x_position = img_rects[0].x0
                else:
                    y_position = 0
                    y_bottom = 0
                    img_height = 0
                    x_position = 0
                
                all_images.append({
                    'page_number': page_num + 1,
                    'image_index': img_index,
                    'xref': xref,
                    'base_image': base_image,
                    'y_position': y_position,
                    'y_bottom': y_bottom,
                    'x_position': x_position,
                    'img_height': img_height,
                    'page_height': page_height,
                    'width': base_image['width'],
                    'height': base_image['height']
                })
        
        # Detect and filter logos
        logos = self._detect_logos(all_images, len(doc))
        
        # Second pass: Save non-logo images
        images = []
        saved_count = 0
        skipped_count = 0
        
        for img_data in all_images:
            if self._is_logo_image(img_data, logos):
                skipped_count += 1
                continue
            
            saved_count += 1
            img_filename = f"page{img_data['page_number']}_img{saved_count}.{img_data['base_image']['ext']}"
            img_path = output_dir / img_filename
            
            with open(img_path, 'wb') as f:
                f.write(img_data['base_image']['image'])
            
            relative_path = f"Media/{pdf_path.stem}/{img_filename}"
            
            images.append({
                'page_number': img_data['page_number'],
                'image_index': saved_count,
                'image_path': relative_path,
                'local_path': str(img_path),
                'extension': img_data['base_image']['ext'],
                'width': img_data['width'],
                'height': img_data['height'],
                'y_position': img_data['y_position'],
                'y_bottom': img_data['y_bottom'],
                'x_position': img_data['x_position']  # Include X position too
            })
        
        doc.close()
        logger.info(
            f"Extracted {len(all_images)} total images: "
            f"{saved_count} saved, {skipped_count} logos filtered"
        )
        
        return images
    
    def _detect_logos(self, all_images: List[Dict], total_pages: int) -> List[Dict]:
        """Detect repeating logo patterns in headers/footers."""
        position_groups = {}
        
        for img in all_images:
            rel_pos = img['y_position'] / img['page_height'] if img['page_height'] > 0 else 0
            
            # Only header/footer (top 15% or bottom 15%)
            if not (rel_pos < 0.15 or rel_pos > 0.85):
                continue
            
            pos_key = (
                round(img['y_position'] / 10) * 10,
                round(img['width'] / 10) * 10,
                round(img['height'] / 10) * 10
            )
            
            if pos_key not in position_groups:
                position_groups[pos_key] = []
            position_groups[pos_key].append(img)
        
        logos = []
        threshold = max(3, total_pages * 0.5)
        
        for pos_key, images in position_groups.items():
            if len(images) >= threshold:
                logos.append({
                    'y_position': pos_key[0],
                    'width': pos_key[1],
                    'height': pos_key[2],
                    'count': len(images),
                    'tolerance': 20
                })
        
        return logos
    
    def _is_logo_image(self, img: Dict, logos: List[Dict]) -> bool:
        """Check if image matches logo pattern."""
        for logo in logos:
            pos_match = abs(img['y_position'] - logo['y_position']) < logo['tolerance']
            width_match = abs(img['width'] - logo['width']) < logo['tolerance']
            height_match = abs(img['height'] - logo['height']) < logo['tolerance']
            
            if pos_match and width_match and height_match:
                return True
        return False
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict:
        """Extract PDF metadata."""
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