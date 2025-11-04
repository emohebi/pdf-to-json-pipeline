"""
Main orchestrator for PDF to JSON pipeline with SMART image placement.
FINAL FIX: Only populates images in fields that should contain images (photo_diagram, icons, etc.)
"""
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

from config.settings import MAX_WORKERS, MODEL
from config.schemas import SECTION_SCHEMAS, get_section_schema
from src.agents import SectionDetectionAgent, SectionExtractionAgent, ValidationAgent, ReviewAgent
from src.utils import PDFProcessor, StorageManager, setup_logger

logger = setup_logger('pipeline')


class PDFToJSONPipeline:
    """Main orchestrator for semi-agentic PDF processing with image extraction."""
    
    # Fields that should contain actual images (not text with optional image)
    IMAGE_ONLY_FIELDS = {
        'photo_diagram',      # Task activities diagrams
        'safety_icon',        # Safety icons
        'attached_images',    # Attached images section
    }
    
    # Section types where images are expected
    IMAGE_SECTIONS = {
        'additional_ppe_required',  # PPE icons
        'safety',                   # Safety icons
        'attached_images',          # Image attachments
    }
    
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.pdf_processor = PDFProcessor()
        self.section_detector = SectionDetectionAgent()
        self.validator = ValidationAgent()
        self.storage = StorageManager()
        self.review_agent = ReviewAgent()
        self.extracted_images = []

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
            
            # STAGE 3: Extract each section
            if parallel:
                logger.info("STAGE 3: Extracting sections (PARALLEL)...")
                section_jsons = self._extract_sections_parallel(pages_data, sections, document_id)
            else:
                logger.info("STAGE 3: Extracting sections (SEQUENTIAL)...")
                section_jsons = self._extract_sections_non_parallel(pages_data, sections, document_id)
            
            logger.info(f"Extracted {len(section_jsons)} sections")

            # STAGE 3.5: Add image paths to section JSONs (SMART placement)
            logger.info("STAGE 3.5: Mapping images to sections (smart placement)...")
            section_jsons = self._map_images_to_sections(section_jsons, sections)

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
                'extraction_mode': 'parallel' if parallel else 'sequential',
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
    
    def _map_images_to_sections(self, section_jsons: List[Dict], sections: List[Dict]) -> List[Dict]:
        """
        Map extracted images to their corresponding sections.
        SMART: Only populates images in appropriate fields.
        """
        if not self.extracted_images:
            logger.info("No images to map to sections")
            return section_jsons
        
        logger.info(f"Mapping {len(self.extracted_images)} images to {len(section_jsons)} sections...")
        
        for section_json in section_jsons:
            section_name = section_json.get('section_name', '')
            section_type = section_json.get('_metadata', {}).get('section_type', '')
            page_range = section_json.get('page_range', [])
            
            if not page_range or len(page_range) != 2:
                continue
            
            start_page, end_page = page_range
            
            # Find images in this page range
            section_images = [
                img for img in self.extracted_images
                if start_page <= img['page_number'] <= end_page
            ]
            
            if section_images:
                section_images.sort(key=lambda x: (x['page_number'], x['y_position']))
                logger.info(
                    f"  Section '{section_name}' ({section_type}): "
                    f"Found {len(section_images)} image(s) on pages {start_page}-{end_page}"
                )
                
                # Populate with smart field detection
                populated_count = self._populate_images_smart(
                    section_json, 
                    section_images,
                    section_type
                )
                
                if populated_count > 0:
                    logger.info(f"    → Populated {populated_count} image path(s) in appropriate fields")
                else:
                    logger.warning(f"    → No appropriate image fields found")
            else:
                logger.debug(f"  Section '{section_name}': No images found")
        
        return section_jsons
    
    def _populate_images_smart(
        self, 
        section_json: Dict, 
        images: List[Dict],
        section_type: str
    ) -> int:
        """
        Populate images ONLY in fields that should contain images.
        
        SMART RULES:
        1. photo_diagram fields -> Always get images
        2. safety_icon fields -> Always get images  
        3. additional_ppe_required section -> Images for each item
        4. attached_images section -> Images for each item
        5. Other text fields (step_description, notes, etc.) -> NO images
        """
        data = section_json.get('data')
        
        if data is None or not images:
            return 0
        
        self.current_image_index = 0
        self.available_images = images
        self.populated_count = 0
        self.current_section_type = section_type
        
        # Populate based on section type and field names
        self._populate_smart_recursive(data, depth=0, path="data", parent_field="")
        
        return self.populated_count
    
    def _should_populate_image(self, parent_field: str, section_type: str) -> bool:
        """
        Determine if this field should have an image based on field name and section type.
        
        Returns True for:
        - photo_diagram (always)
        - safety_icon (always)
        - attached_images section items
        - additional_ppe_required section items
        - safety section items (icons/statements)
        
        Returns False for:
        - step_description, step_no, notes, acceptable_limit, question, 
          corrective_action, execution_condition, other_content (these are TEXT fields)
        """
        # Fields that should NEVER have images (they're text descriptions)
        TEXT_ONLY_FIELDS = {
            'step_description', 'step_no', 'notes', 'acceptable_limit',
            'question', 'corrective_action', 'execution_condition', 
            'other_content', 'text', 'seq', 'risk_description',
            'reason_for_control', 'critical_controls', 'sequence_no',
            'sequence_name', 'equipment_asset', 'maintainable_item', 'lmi',
            'tool_set', 'tools', 'document_reference_number', 'document_description',
            'mechanical_drawings', 'structural_civil_drawings'
        }
        
        # Fields that should ALWAYS have images
        IMAGE_FIELDS = {
            'photo_diagram', 'safety_icon', 'safety_statement'
        }
        
        # Check if parent field is explicitly an image field
        if parent_field in IMAGE_FIELDS:
            return True
        
        # Check if parent field is explicitly text-only
        if parent_field in TEXT_ONLY_FIELDS:
            return False
        
        # Section-specific rules
        if section_type in self.IMAGE_SECTIONS:
            # In image sections, root-level items get images
            if parent_field in ['', 'data']:
                return True
        
        return False
    
    def _populate_smart_recursive(
        self, 
        obj: Any, 
        depth: int, 
        path: str,
        parent_field: str
    ) -> None:
        """
        Recursively populate images ONLY in appropriate fields.
        """
        if self.current_image_index >= len(self.available_images):
            return
        
        # Case 1: Dict with text and image (leaf node)
        if isinstance(obj, dict) and 'text' in obj and 'image' in obj:
            # Only populate if this is an appropriate field for images
            if self._should_populate_image(parent_field, self.current_section_type):
                if obj['image'] == "":
                    obj['image'] = self.available_images[self.current_image_index]['image_path']
                    logger.debug(f"      [{depth}] {path} ({parent_field}): Populated image")
                    self.current_image_index += 1
                    self.populated_count += 1
            else:
                logger.debug(f"      [{depth}] {path} ({parent_field}): Skipped (text field)")
            return  # Leaf node
        
        # Case 2: Dict - recurse into values
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if key in ['_metadata', '_internal']:
                    continue
                new_path = f"{path}.{key}"
                # Pass the current key as parent_field for context
                self._populate_smart_recursive(value, depth + 1, new_path, key)
        
        # Case 3: List - recurse into items
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                new_path = f"{path}[{idx}]"
                # Keep the same parent_field for list items
                self._populate_smart_recursive(item, depth + 1, new_path, parent_field)
    
    def _extract_sections_parallel(self, pages_data: List[Dict], sections: List[Dict], document_id: str) -> List[Dict]:
        """Extract sections in parallel."""
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
                
                section_schema = get_section_schema(section['section_type'])
                extractor = SectionExtractionAgent(section_schema)
                
                future = executor.submit(
                    extractor.extract_section,
                    section_pages,
                    section,
                    next_section_name,
                    document_id
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
    
    def _extract_sections_non_parallel(self, pages_data: List[Dict], sections: List[Dict], document_id: str) -> List[Dict]:
        """Extract sections sequentially."""
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
                    
                    section_schema = get_section_schema(section['section_type'])
                    extractor = SectionExtractionAgent(section_schema)
                    
                    section_json = extractor.extract_section(
                        section_pages,
                        section,
                        next_section_name,
                        document_id
                    )
                    
                    section_jsons.append(section_json)
                    
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
    
    def process_batch(self, pdf_paths: List[str], resume: bool = False, parallel: bool = True) -> Dict:
        """Process batch of PDFs."""
        logger.info(f"Starting batch processing: {len(pdf_paths)} documents")
        
        progress = self.storage.load_progress()
        completed = set(progress.get('completed', []))
        failed = progress.get('failed', [])
        
        if resume:
            remaining = [p for p in pdf_paths if Path(p).stem not in completed]
            logger.info(f"Resuming: {len(remaining)} remaining, {len(completed)} completed")
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
        
        logger.info(f"\nBatch processing complete:")
        logger.info(f"  Completed: {results['completed_count']}")
        logger.info(f"  Failed: {results['failed_count']}")
        
        return results