"""
Post-processor for DocuPorter format
Handles field duplication (orig_text = text) and empty field cleanup
"""
from typing import Any, Dict, List


def process_docuporter_format(data: Any) -> Any:
    """
    Process extracted data to DocuPorter format:
    1. Duplicate text/image values to orig_text/orig_image
    2. Clean empty fields (both text and image empty -> {} or [])
    
    Args:
        data: Extracted data in any format
        
    Returns:
        Processed data in DocuPorter format
    """
    # First duplicate values to orig_ fields
    data_with_orig = duplicate_to_orig_fields(data)
    
    # Then clean empty fields
    cleaned_data = clean_empty_fields(data_with_orig)
    
    return cleaned_data


def duplicate_to_orig_fields(data: Any) -> Any:
    """
    Duplicate text/image/seq values to orig_text/orig_image/orig_seq fields.
    Both fields will have the same value.
    """
    if isinstance(data, dict):
        result = {}
        
        # Check if this dict has text/image fields that need duplication
        has_text = 'text' in data
        has_image = 'image' in data
        has_seq = 'seq' in data
        
        for key, value in data.items():
            if key == 'text':
                # Add both orig_text and text with same value
                result['orig_text'] = value
                result['text'] = value
            elif key == 'image':
                # Add both orig_image and image with same value
                result['orig_image'] = value
                result['image'] = value
            elif key == 'seq':
                # Add both orig_seq and seq with same value
                result['orig_seq'] = value
                result['seq'] = value
            elif key in ['orig_text', 'orig_image', 'orig_seq']:
                # Already has orig_ field, ensure regular field exists too
                result[key] = value
                if key == 'orig_text' and 'text' not in data:
                    result['text'] = value
                elif key == 'orig_image' and 'image' not in data:
                    result['image'] = value
                elif key == 'orig_seq' and 'seq' not in data:
                    result['seq'] = value
            else:
                # Recursively process nested structures
                result[key] = duplicate_to_orig_fields(value)
        
        return result
        
    elif isinstance(data, list):
        return [duplicate_to_orig_fields(item) for item in data]
    else:
        return data


def clean_empty_fields(data: Any) -> Any:
    """
    Clean up empty fields according to DocuPorter rules:
    - [{"orig_text": "", "text": "", "orig_image": "", "image": ""}] -> []
    - {"orig_text": "", "text": "", "orig_image": "", "image": ""} -> {}
    """
    if isinstance(data, dict):
        # Check if this is a text/image dict
        is_text_image_dict = any(
            key in data for key in 
            ['text', 'image', 'orig_text', 'orig_image']
        )
        
        if is_text_image_dict:
            # Check if all text/image fields are empty
            has_content = False
            for key, value in data.items():
                if key in ['text', 'orig_text', 'image', 'orig_image', 'seq', 'orig_seq']:
                    if value and value != "":
                        has_content = True
                        break
                elif value:  # Other fields
                    has_content = True
                    break
            
            if not has_content:
                # All fields are empty, return empty dict
                return {}
            else:
                # Has content, recursively clean nested fields
                cleaned = {}
                for key, value in data.items():
                    cleaned_value = clean_empty_fields(value)
                    # Only add if not empty
                    if cleaned_value is not None and cleaned_value != {} and cleaned_value != []:
                        cleaned[key] = cleaned_value
                    elif key in ['text', 'orig_text', 'image', 'orig_image', 'seq', 'orig_seq']:
                        # Keep these fields even if empty, as long as dict has some content
                        cleaned[key] = value
                return cleaned if cleaned else {}
        else:
            # Regular dict, recursively process
            cleaned = {}
            for key, value in data.items():
                cleaned[key] = clean_empty_fields(value)
            return cleaned
            
    elif isinstance(data, list):
        cleaned = []
        for item in data:
            cleaned_item = clean_empty_fields(item)
            # Only add non-empty items
            if cleaned_item and cleaned_item != {} and cleaned_item != []:
                cleaned.append(cleaned_item)
        return cleaned
        
    else:
        return data


def fix_task_activities_sequences(task_activities: List[Dict]) -> List[Dict]:
    """
    Fix task activities sequence numbering according to rule:
    - Sub-headings are NOT sequences (empty sequence_no)
    - Items under sub-headings (a, b, c) ARE sequences (get next number)
    
    Args:
        task_activities: List of task activity items
        
    Returns:
        Fixed task activities with proper sequence numbering
    """
    if not task_activities:
        return task_activities
    
    fixed_activities = []
    current_sequence_num = 0
    in_sub_heading = False
    
    for item in task_activities:
        # Check if this is a sub-heading
        sequence_name = item.get('sequence_name', {})
        sequence_no = item.get('sequence_no', {})
        
        # Get the text value (could be in 'text' or direct value)
        if isinstance(sequence_name, dict):
            name_text = sequence_name.get('text', '')
        else:
            name_text = str(sequence_name)
            
        if isinstance(sequence_no, dict):
            no_text = sequence_no.get('text', '')
        else:
            no_text = str(sequence_no)

        required_fields = [
                'step_no', 'equipment_asset'
            ]
            
        for field in required_fields:
            if field not in item or not item[field]:
                item[field] = {"orig_text": "", "text": ""}
        
        # Check if it's a sub-heading pattern
        sub_heading_patterns = [
            'tasks to be done under isolation',
            'pre-isolation',
            'post-isolation',
            'under isolation',
            'normal conditions'
        ]
        
        is_sub_heading = any(
            pattern in name_text.lower() 
            for pattern in sub_heading_patterns
        )
        
        # Check if it's a sub-item (a., b., c., 1a, 1b, etc.)
        import re
        is_sub_item = bool(re.match(r'^[a-z]\.|^\d+[a-z]', no_text.lower()))
        
        if is_sub_heading:
            # This is a sub-heading, not a sequence
            in_sub_heading = True
            # Keep empty sequence_no for sub-headings
            if isinstance(item['sequence_no'], dict):
                item['sequence_no']['orig_text'] = ""
                item['sequence_no']['text'] = ""
            else:
                item['sequence_no'] = {"orig_text": "", "text": ""}
                
        elif is_sub_item and in_sub_heading:
            # This is a sub-item under a sub-heading, it should be a sequence
            current_sequence_num += 1
            if isinstance(item['sequence_no'], dict):
                item['sequence_no']['orig_text'] = str(current_sequence_num)
                item['sequence_no']['text'] = str(current_sequence_num)
            else:
                item['sequence_no'] = {
                    "orig_text": str(current_sequence_num),
                    "text": str(current_sequence_num)
                }
        elif no_text and no_text.isdigit():
            # Regular numbered sequence
            current_sequence_num = int(no_text)
            in_sub_heading = False
        
        fixed_activities.append(item)
    
    return fixed_activities


def format_section_for_docuporter(section_data: Any, section_type: str) -> Any:
    """
    Format a section according to DocuPorter requirements.
    
    Args:
        section_data: Raw section data
        section_type: Type of section
        
    Returns:
        Formatted section data
    """
    # Apply DocuPorter format processing
    formatted = process_docuporter_format(section_data)
    
    # Special handling for task_activities
    if section_type == 'task_activities' and isinstance(formatted, list):
        formatted = fix_task_activities_sequences(formatted)
    
    # Special handling for attached images vs attached_images
    if section_type == 'attached_images':
        section_type = 'attached images'
    
    return formatted
