"""
PDF processing utilities.
"""
import pymupdf
from typing import List, Dict
from pathlib import Path

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
    
    def pdf_to_images(self, pdf_path: str) -> List[Dict]:
        """
        Convert PDF pages to high-quality images with metadata.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of dicts containing page images and metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        try:
            doc = pymupdf.open(str(pdf_path))
            pages_data = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image with specified DPI
                mat = pymupdf.Matrix(self.dpi/72, self.dpi/72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Extract text as fallback
                text = page.get_text()
                
                # Get page dimensions
                rect = page.rect
                
                pages_data.append({
                    'page_number': page_num + 1,
                    'image': img_data,
                    'text': text,
                    'width': pix.width,
                    'height': pix.height,
                    'original_width': rect.width,
                    'original_height': rect.height
                })
                
                logger.debug(f"Processed page {page_num + 1}/{len(doc)}")
            
            doc.close()
            logger.info(f"Successfully extracted {len(pages_data)} pages")
            
            return pages_data
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
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
