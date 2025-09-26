import os
import fitz
import uuid
from PyPDF2 import PdfReader
from typing import List, Tuple, Optional
from app.utils.text_utils import normalize_text, similarity_score, find_best_position_match, generate_text_variations, estimate_tokens, split_text_by_tokens
# Note: extract_meaningful_phrases and extract_key_phrases have been disabled in text_utils.py

# Global storage for text positions
text_positions_cache = {}

class PDFService:
    @staticmethod
    def _extract_page_number_from_text(text: str) -> int:
        """Extract page number from text content"""
        import re
        # Look for [PAGE X] pattern
        page_match = re.search(r'\[PAGE (\d+)\]', text)
        if page_match:
            return int(page_match.group(1)) - 1  # Convert to 0-based
        
        # Look for other page indicators (fallback patterns)
        page_patterns = [
            r'page\s+(\d+)',
            r'p\.?\s*(\d+)',
            r'^(\d+)\s*$'  # Just a number at the start
        ]
        
        for pattern in page_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    page_num = int(match.group(1)) - 1
                    if 0 <= page_num <= 2000:  # Reasonable page number range
                        return page_num
                except ValueError:
                    continue
        
        return None  # Could not determine page number

    @staticmethod
    def extract_text_with_positions(file_path):
        """Enhanced PDF text extraction that captures positions for accurate highlighting"""
        all_text = []
        reader_info = {}
        text_positions = {}  # Store exact positions for each text chunk
        
        try:
            doc = fitz.open(file_path)
            filename = os.path.basename(file_path)
            
            for page_num, page in enumerate(doc):
                page_text_blocks = []
                page_positions = []
                
                # Get text with detailed position information
                text_dict = page.get_text("dict")
                
                # Process each block with position tracking
                for block_idx, block in enumerate(text_dict["blocks"]):
                    if "lines" in block:  # Text block
                        block_text = []
                        block_rects = []
                        
                        for line in block["lines"]:
                            line_text = []
                            line_rects = []
                            
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    line_text.append(text)
                                    line_rects.append(fitz.Rect(span["bbox"]))
                            
                            if line_text:
                                block_text.append(" ".join(line_text))
                                if line_rects:
                                    # Merge rectangles in the line
                                    merged_rect = line_rects[0]
                                    for rect in line_rects[1:]:
                                        merged_rect = merged_rect | rect
                                    block_rects.append(merged_rect)
                        
                        if block_text:
                            full_block_text = "\n".join(block_text)
                            page_text_blocks.append(full_block_text)
                            
                            # Store position information
                            if block_rects:
                                merged_block_rect = block_rects[0]
                                for rect in block_rects[1:]:
                                    merged_block_rect = merged_block_rect | rect
                                page_positions.append({
                                    'text': full_block_text,
                                    'rect': merged_block_rect,
                                    'page': page_num,
                                    'block_idx': block_idx
                                })
                
                # Extract tables with positions
                try:
                    tables = page.find_tables()
                    for table_idx, table in enumerate(tables):
                        table_data = table.extract()
                        if table_data:
                            table_text = []
                            for row in table_data:
                                if row and any(cell for cell in row if cell and cell.strip()):
                                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                                    table_text.append(" | ".join(clean_row))
                            
                            if table_text:
                                full_table_text = "\n[TABLE]\n" + "\n".join(table_text) + "\n[/TABLE]\n"
                                page_text_blocks.append(full_table_text)
                                
                                # Store table position
                                table_rect = table.bbox
                                page_positions.append({
                                    'text': full_table_text,
                                    'rect': fitz.Rect(table_rect),
                                    'page': page_num,
                                    'table_idx': table_idx,
                                    'is_table': True
                                })
                except Exception as e:
                    print(f"Table extraction warning for page {page_num + 1}: {e}")
                
                # Combine page content
                if page_text_blocks:
                    page_content = "\n\n".join(page_text_blocks)
                    all_text.append(f"[PAGE {page_num + 1}]\n{page_content}")
                    
                    # Store positions for this page
                    text_positions[page_num] = page_positions
            
            doc.close()
            
            # Store positions in global cache
            text_positions_cache[filename] = text_positions
            
            # Fallback to PyPDF2 if needed
            if not all_text:
                reader = PdfReader(file_path)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        all_text.append(f"[PAGE {page_num + 1}]\n{text}")
            
            extracted_text = "\n\n".join(all_text)
            
            reader_info = {
                'num_pages': len(all_text),
                'first_page_text': all_text[0][:500] + "..." if len(all_text) > 500 else all_text[0],
                'extraction_method': 'PyMuPDF with position tracking',
                'tables_found': extracted_text.count('[TABLE]'),
                'positions_stored': len(text_positions)
            }
            
            return extracted_text, reader_info
            
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return "", {'num_pages': 0, 'first_page_text': None, 'extraction_method': 'Failed', 'tables_found': 0}

    @staticmethod
    def chunk_text_with_positions(text, filename):
        """Enhanced text chunking that preserves position information"""
        chunks = []
        chunk_positions = {}  # Map chunk text to position info
        
        # Get stored positions for this file
        positions = text_positions_cache.get(filename, {})
        
        # Split by pages first
        pages = text.split('[PAGE ')
        
        for page_section in pages:
            if not page_section.strip():
                continue
            
            # Extract page number with improved error handling
            page_num = None
            if page_section.startswith('[PAGE'):
                try:
                    page_marker, content = page_section.split(']', 1)
                    page_num = int(page_marker.replace('[PAGE', '').strip()) - 1
                except (ValueError, IndexError):
                    content = page_section
                    # Try to extract page number from the beginning of the content
                    extracted_page = PDFService._extract_page_number_from_text(page_section)
                    if extracted_page is not None:
                        page_num = extracted_page
            else:
                content = '[PAGE ' + page_section
                # Try to extract page number from the content
                extracted_page = PDFService._extract_page_number_from_text(content)
                if extracted_page is not None:
                    page_num = extracted_page
                
            # Skip empty sections
            if not content.strip():
                continue
            
            # Get position info for this page
            page_positions = positions.get(page_num, []) if page_num is not None else []
            
            # Handle pages with tables
            if '[TABLE]' in content:
                parts = content.split('[TABLE]')
                
                # Process text before tables
                if parts[0].strip():
                    text_chunks = PDFService._chunk_regular_text_with_positions(parts[0].strip(), page_positions, page_num)
                    chunks.extend([chunk['text'] for chunk in text_chunks])
                    # Store position mappings
                    for chunk_info in text_chunks:
                        chunk_positions[chunk_info['text']] = chunk_info.get('position_info')
                
                # Process tables
                for i, part in enumerate(parts[1:], 1):
                    if '[/TABLE]' in part:
                        table_part, remaining_text = part.split('[/TABLE]', 1)
                        
                        if table_part.strip():
                            table_chunk = f"[TABLE]\n{table_part.strip()}\n[/TABLE]"
                            chunks.append(table_chunk)
                            
                            # Find matching table position or create fallback
                            table_position = None
                            for pos_info in page_positions:
                                if pos_info.get('is_table') and table_part.strip() in pos_info['text']:
                                    table_position = pos_info
                                    break
                            
                            # Ensure table always has page number
                            if not table_position:
                                if page_num is not None:
                                    table_position = {'page': page_num, 'is_table': True}
                                else:
                                    # Try to extract page from table content
                                    extracted_page = PDFService._extract_page_number_from_text(table_chunk)
                                    if extracted_page is not None:
                                        table_position = {'page': extracted_page, 'is_table': True}
                                    else:
                                        table_position = {'page': 0, 'is_table': True}
                                
                            chunk_positions[table_chunk] = table_position
                        
                        if remaining_text.strip():
                            text_chunks = PDFService._chunk_regular_text_with_positions(remaining_text.strip(), page_positions, page_num)
                            chunks.extend([chunk['text'] for chunk in text_chunks])
                            for chunk_info in text_chunks:
                                chunk_positions[chunk_info['text']] = chunk_info.get('position_info')
            else:
                # Regular text chunking
                text_chunks = PDFService._chunk_regular_text_with_positions(content, page_positions, page_num)
                chunks.extend([chunk['text'] for chunk in text_chunks])
                for chunk_info in text_chunks:
                    chunk_positions[chunk_info['text']] = chunk_info.get('position_info')
        
        # Store chunk positions in cache
        text_positions_cache[f"{filename}_chunks"] = chunk_positions
        
        # Filter out chunks and add page information to chunk text if missing
        final_chunks = []
        updated_chunk_positions = {}
        
        for chunk in chunks:
            if len(chunk.strip()) > 20:
                # If chunk doesn't contain page marker but we know the page, add it
                chunk_position = chunk_positions.get(chunk, {})
                if chunk_position and 'page' in chunk_position and '[PAGE' not in chunk:
                    page_marker = f"[PAGE {chunk_position['page'] + 1}]"
                    # Add page marker at the beginning for better context
                    updated_chunk = f"{page_marker}\n{chunk}"
                    final_chunks.append(updated_chunk)
                    # Update the position mapping with the new chunk text
                    updated_chunk_positions[updated_chunk] = chunk_position
                else:
                    final_chunks.append(chunk)
                    updated_chunk_positions[chunk] = chunk_position
        
        # Update the cache with the new mappings
        text_positions_cache[f"{filename}_chunks"] = updated_chunk_positions
        
        return final_chunks

    @staticmethod
    def _chunk_regular_text_with_positions(text, page_positions, page_num=None, chunk_size=8):
        """Helper function to chunk regular text while preserving position information"""
        import re
        from app.utils.text_utils import estimate_tokens, split_text_by_tokens
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        
        # Create larger initial chunks to reduce total number of chunks
        for i in range(0, len(sentences), chunk_size):
            chunk_text = ' '.join(sentences[i:i+chunk_size]).strip()
            if chunk_text:
                # Check if chunk is too large - use conservative but reasonable limits
                estimated_tokens = estimate_tokens(chunk_text)
                if estimated_tokens > 6000:  # More reasonable limit (about 18k characters)
                    # Split large chunks further with reasonable max tokens
                    sub_chunks = split_text_by_tokens(chunk_text, max_tokens=5000)
                    for sub_chunk in sub_chunks:
                        if len(sub_chunk.strip()) > 50:  # Only add meaningful chunks
                            # Find best matching position for this sub-chunk
                            best_position = find_best_position_match(sub_chunk, page_positions)
                            
                            # ALWAYS ensure every chunk has at least a page number
                            if not best_position:
                                if page_num is not None:
                                    best_position = {'page': page_num}
                                else:
                                    # Try to extract page from the chunk text itself
                                    extracted_page = PDFService._extract_page_number_from_text(sub_chunk)
                                    if extracted_page is not None:
                                        best_position = {'page': extracted_page}
                                    else:
                                        best_position = {'page': 0}  # Default to first page
                                
                            chunks.append({
                                'text': sub_chunk,
                                'position_info': best_position
                            })
                else:
                    # Find best matching position for this chunk
                    best_position = find_best_position_match(chunk_text, page_positions)
                    
                    # ALWAYS ensure every chunk has at least a page number
                    if not best_position:
                        if page_num is not None:
                            best_position = {'page': page_num}
                        else:
                            # Try to extract page from the chunk text itself
                            extracted_page = PDFService._extract_page_number_from_text(chunk_text)
                            if extracted_page is not None:
                                best_position = {'page': extracted_page}
                            else:
                                # Only show warning for chunks that don't contain page markers and we couldn't extract page
                                if '[PAGE' not in chunk_text:
                                    print(f"Warning: Could not determine page for chunk: {chunk_text[:50]}...")
                                best_position = {'page': 0}  # Default to first page
                    
                    chunks.append({
                        'text': chunk_text,
                        'position_info': best_position
                    })
        
        return chunks# Global instance
pdf_service = PDFService()
