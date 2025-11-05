"""
FINAL ENHANCED pipeline.py with PRECISE image-to-text matching using coordinates.
Uses actual text block positions from PyMuPDF for accurate image context extraction.
"""
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time
import re

from config.settings import MAX_WORKERS, MODEL
from config.schemas import SECTION_SCHEMAS, get_section_schema
from src.agents import SectionDetectionAgent, SectionExtractionAgent, ValidationAgent, ReviewAgent
from src.utils import PDFProcessor, StorageManager, setup_logger
from src.utils.image_descriptor import create_section_image_descriptions

logger = setup_logger('pipeline')


class PDFToJSONPipeline:
    """Enhanced orchestrator with PRECISE context-aware image placement."""
    
    IMAGE_FIELDS = {'photo_diagram', 'safety_icon', 'attached_images', 'image'}
    
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.pdf_processor = PDFProcessor()
        self.section_detector = SectionDetectionAgent()
        self.validator = ValidationAgent()
        self.storage = StorageManager()
        self.review_agent = ReviewAgent()
        self.extracted_images = []
        self.pages_data_cache = []

    def call_review_agent(self, section_jsons: str, document_id = None, pages_data: List[Dict] = None):
        review_results = self.review_agent.review_document(
            section_jsons=section_jsons,
            document_id=document_id,
            pages_data=pages_data
        )
        
        total_issues = sum(
            len(issues) 
            for key, issues in review_results.items() 
            if key != 'error' and isinstance(issues, list)
        )
        
        if total_issues == 0:
            logger.info(f"Review PASSED - No issues found")
        else:
            logger.warning(f"Review found {total_issues} issue(s):")
            logger.warning(f"    - Incomplete sentences: {len(review_results.get('incomplete_sentences', []))}")
            logger.warning(f"    - Duplications: {len(review_results.get('duplications', []))}")
            logger.warning(f"    - Order issues: {len(review_results.get('order_issues', []))}")
            logger.warning(f"    - Missing information: {len(review_results.get('missing_information', []))}")
        return review_results, total_issues
    
    def process_single_pdf(self, pdf_path: str, parallel: bool = False, review_json: str = None) -> Dict:
        """Process a single PDF document with image extraction."""
        pdf_path = Path(pdf_path)
        document_id = pdf_path.stem
        start_time = time.time()
        
        logger.info(f"=" * 60)
        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"Document ID: {document_id}")
        logger.info(f"Mode: {'Parallel' if parallel else 'Sequential'}")
        logger.info(f"=" * 60)
        
        try:
            # STAGE 0: Extract images from PDF (excluding logos)
            logger.info("STAGE 0: Extracting embedded images from PDF...")
            self.extracted_images = self.pdf_processor.extract_images_from_pdf(str(pdf_path))
            logger.info(f"Extracted {len(self.extracted_images)} images (logos filtered)")
            
            # STAGE 1: Extract PDF to images
            logger.info("STAGE 1: Extracting PDF pages...")
            pages_data = self.pdf_processor.pdf_to_images(str(pdf_path), extract_with_bedrock=False)
            logger.info(f"Extracted {len(pages_data)} pages")
            
            # STAGE 2: Detect sections
            logger.info("STAGE 2: Detecting sections...")
            sections = self.section_detector.detect_sections(pages_data, document_id)
            logger.info(f"Detected {len(sections)} sections")
            
            # STAGE 3: Extract each section WITH IMAGE INFORMATION
            if parallel:
                logger.info("STAGE 3: Extracting sections with images (PARALLEL)...")
                section_jsons = self._extract_sections_parallel(pages_data, sections, document_id)
            else:
                logger.info("STAGE 3: Extracting sections with images (SEQUENTIAL)...")
                section_jsons = self._extract_sections_non_parallel(pages_data, sections, document_id)
            
            logger.info(f"Extracted {len(section_jsons)} sections")

            # STAGE 3.75: Review extracted content
            logger.info("STAGE 3.75: Reviewing extracted content...")
            review_results, total_issues = self.call_review_agent(
                section_jsons=section_jsons,
                document_id=document_id,
                pages_data=pages_data
            )
            
            # STAGE 4: Validate and combine
            logger.info("STAGE 4: Validating and combining...")
            duration = time.time() - start_time
            
            document_metadata = {
                'document_id': document_id,
                'source_file': str(pdf_path),
                'total_pages': len(pages_data),
                'total_images_extracted': len(self.extracted_images),
                'processing_timestamp': datetime.now().isoformat(),
                'processing_duration': duration,
                'model_used': 'claude-sonnet-4',
                'extraction_mode': 'precise_positioning',
                'review_results': review_results,
                'review_passed': total_issues == 0,
                'review_issues_count': total_issues
            }
            
            final_json = self.validator.validate_and_combine(
                section_jsons, document_metadata, document_id
            )
            
            logger.info(f"Processing complete ({duration:.1f}s)")
            logger.info(f"=" * 60)
            
            return final_json
            
        except Exception as e:
            logger.error(f"Processing failed for {document_id}: {e}")
            raise
    
    
    def _extract_sections_non_parallel(self, pages_data: List[Dict], sections: List[Dict], document_id: str) -> List[Dict]:
        """Extract sections sequentially WITH IMAGE DESCRIPTIONS."""
        section_jsons = []
        
        for idx, section in enumerate(sections, 1):
            section_name = section['section_name']
            next_section_name = "<END OF DOCUMENT>"
            if idx < len(sections):
                next_section_name = sections[idx]['section_name']
            
            counter = 3
            loop = True
            while loop:
                loop = False
                try:
                    logger.info(f"  [{idx}/{len(sections)}] Extracting: {section_name}")
                    
                    start_idx = section['start_page'] - 1
                    end_idx = section['end_page']
                    section_pages = pages_data[start_idx:end_idx]
                    
                    # Find images for this section
                    section_images = [
                        img for img in self.extracted_images
                        if section['start_page'] <= img['page_number'] <= section['end_page']
                    ]
                    
                    # Generate descriptions for section images
                    image_descriptions = {}
                    if section_images:
                        logger.info(f"  [{idx}/{len(sections)}] Generating descriptions for {len(section_images)} image(s)...")
                        image_descriptions = create_section_image_descriptions(
                            section_images,
                            section_name,
                            section['section_type'],
                            document_id
                        )
                        logger.info(f"  [{idx}/{len(sections)}] Generated {len(image_descriptions)} descriptions")
                    
                    section_schema = get_section_schema(section['section_type'])
                    extractor = SectionExtractionAgent(section_schema)
                    
                    # Pass image descriptions to the extractor
                    section_json = extractor.extract_section(
                        section_pages,
                        section,
                        next_section_name,
                        document_id,
                        image_descriptions  # Pass descriptions dict
                    )
                    
                    section_jsons.append(section_json)
                    
                    # Log completion
                    if '_metadata' in section_json:
                        confidence = section_json['_metadata'].get('confidence', 0)
                        logger.info(f"  [{idx}/{len(sections)}] Completed: {section_name} (confidence: {confidence:.2f})")
                    else:
                        logger.info(f"  [{idx}/{len(sections)}] Completed: {section_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to extract {section_name}: {e}")
                    if counter > 0:
                        logger.info(f"Trying {counter} more time ...")
                        counter -= 1
                        loop = True
        
        return section_jsons

    # 3. Modify _extract_sections_parallel similarly:
    def _extract_sections_parallel(self, pages_data: List[Dict], sections: List[Dict], document_id: str) -> List[Dict]:
        """Extract sections in parallel WITH IMAGE DESCRIPTIONS."""
        section_jsons = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for idx, section in enumerate(sections):
                start_idx = section['start_page'] - 1
                end_idx = section['end_page']
                section_pages = pages_data[start_idx:end_idx]
                
                next_section_name = "<END OF DOCUMENT>"
                if idx < len(sections) - 1:
                    next_section_name = sections[idx + 1]['section_name']
                
                # Find images for this section
                section_images = [
                    img for img in self.extracted_images
                    if section['start_page'] <= img['page_number'] <= section['end_page']
                ]
                
                # Generate descriptions
                image_descriptions = {}
                if section_images:
                    logger.info(f"Generating descriptions for {len(section_images)} images in {section['section_name']}")
                    image_descriptions = create_section_image_descriptions(
                        section_images,
                        section['section_name'],
                        section['section_type'],
                        document_id
                    )
                
                section_schema = get_section_schema(section['section_type'])
                extractor = SectionExtractionAgent(section_schema)
                
                future = executor.submit(
                    extractor.extract_section,
                    section_pages,
                    section,
                    next_section_name,
                    document_id,
                    image_descriptions  # Pass descriptions dict
                )
                futures[future] = section['section_name']
            
            for future in as_completed(futures):
                section_name = futures[future]
                try:
                    section_json = future.result()
                    section_jsons.append(section_json)
                except Exception as e:
                    logger.error(f"Failed to extract {section_name}: {e}")
        
        return section_jsons
    
    def process_batch(self, pdf_paths: List[str], resume: bool = False, parallel: bool = True) -> Dict:
        """Process batch of PDFs."""
        logger.info(f"Starting batch: {len(pdf_paths)} documents")
        
        progress = self.storage.load_progress()
        completed = set(progress.get('completed', []))
        failed = progress.get('failed', [])
        
        if resume:
            remaining = [p for p in pdf_paths if Path(p).stem not in completed]
            logger.info(f"Resuming: {len(remaining)} remaining")
        else:
            remaining = pdf_paths
            completed = set()
            failed = []
        
        results = {
            'total': len(pdf_paths),
            'completed': [],
            'failed': [],
            'start_time': datetime.now().isoformat()
        }
        
        for i, pdf_path in enumerate(remaining):
            document_id = Path(pdf_path).stem
            logger.info(f"\n[{i+1}/{len(remaining)}] Processing {document_id}")
            
            try:
                self.process_single_pdf(pdf_path, parallel=parallel)
                completed.add(document_id)
                results['completed'].append(document_id)
            except Exception as e:
                logger.error(f"Failed: {document_id} - {e}")
                failed.append(document_id)
                results['failed'].append({'document_id': document_id, 'error': str(e)})
            
            self.storage.update_progress(list(completed), failed, len(pdf_paths))
        
        results['end_time'] = datetime.now().isoformat()
        results['completed_count'] = len(results['completed'])
        results['failed_count'] = len(results['failed'])
        
        logger.info(f"\nBatch complete: {results['completed_count']} completed, {results['failed_count']} failed")
        
        return results