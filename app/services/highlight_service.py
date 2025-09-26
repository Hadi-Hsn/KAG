import fitz
from typing import Optional, List
from app.services.pdf_service import text_positions_cache
from app.utils.text_utils import (
    normalize_text, 
    similarity_score,
    generate_text_variations,
    find_best_text_match
)

class PageNavigationService:
    """
    Service for navigating to specific pages in PDF documents.
    All highlighting functionality has been removed - this service only provides page navigation.
    Uses the old highlighting logic for robust page number detection.
    """
    
    @staticmethod
    def find_page_number_for_text(doc, text_chunk: str, filename: str) -> Optional[int]:
        """
        Find page number for text chunk using comprehensive search approach.
        This uses the old highlighting logic for better accuracy but only returns page numbers.
        """
        page_number = None
        
        # Method 1: Try to get from cached chunk positions (fastest)
        chunk_positions = text_positions_cache.get(f"{filename}_chunks", {})
        position_info = chunk_positions.get(text_chunk)
        
        if position_info:
            try:
                if isinstance(position_info, dict) and 'page' in position_info:
                    page_num = position_info['page']
                    page_number = page_num + 1  # Convert from 0-based to 1-based
                    return page_number
                elif isinstance(position_info, (int, float)):
                    page_number = int(position_info) + 1  # Convert from 0-based to 1-based
                    return page_number
            except Exception as e:
                print(f"Position info access failed: {e}")
        
        # Method 2: Multi-pass text matching (like the old highlighting system)
        page_number = PageNavigationService._find_page_with_multipass_matching(doc, text_chunk, filename)
        if page_number:
            return page_number
            
        # Method 3: Simple similarity fallback
        page_number = PageNavigationService.get_page_number_from_text(doc, text_chunk)
        if page_number is not None:
            page_number += 1  # Convert from 0-based to 1-based
            return page_number
        
        print(f"WARNING: No page found for chunk: {text_chunk[:100]}...")
        return None

    @staticmethod
    def _find_page_with_multipass_matching(doc, text_chunk: str, filename: str) -> Optional[int]:
        """
        Multi-pass page detection using the old highlighting approach.
        This mimics the old highlighting logic but only returns page numbers.
        """
        # Get cached position information
        file_positions = text_positions_cache.get(filename, {})
        
        # Generate text variations for better matching
        text_variations = generate_text_variations(text_chunk)
        all_search_texts = [text_chunk] + text_variations
        
        best_page = None
        best_score = 0.0
        
        # Search through all pages
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            page_text_normalized = normalize_text(page_text)
            
            # Try direct text matching first
            for search_text in all_search_texts:
                search_normalized = normalize_text(search_text)
                
                # Check for exact substring match
                if search_normalized in page_text_normalized:
                    # Calculate match quality based on length and position
                    match_score = len(search_normalized) / max(len(page_text_normalized), 1)
                    if match_score > best_score:
                        best_score = match_score
                        best_page = page_num + 1  # Convert to 1-based
                
                # Try phrase-based matching
                matched_phrases = find_best_text_match(search_text, page_text)
                if matched_phrases:
                    phrase_score = len(' '.join(matched_phrases)) / max(len(page_text), 1)
                    if phrase_score > best_score:
                        best_score = phrase_score
                        best_page = page_num + 1  # Convert to 1-based
            
            # Also try position-based matching if we have position data
            page_positions = file_positions.get(page_num, [])
            if page_positions:
                for pos_info in page_positions:
                    pos_text = pos_info.get('text', '')
                    if pos_text:
                        similarity = similarity_score(normalize_text(text_chunk), normalize_text(pos_text))
                        if similarity > 0.4 and similarity > best_score:
                            best_score = similarity
                            best_page = page_num + 1  # Convert to 1-based
        
        return best_page if best_score > 0.2 else None

    @staticmethod
    def get_page_number_from_text(doc, text_chunk: str) -> Optional[int]:
        """
        Find the most likely page number for a text chunk by comparing text similarity.
        Returns 0-based page number or None if no good match found.
        """
        text_normalized = normalize_text(text_chunk)
        best_page = None
        best_score = 0.0
        
        for page_num, page in enumerate(doc):
            page_text = normalize_text(page.get_text())
            score = similarity_score(text_normalized, page_text)
            
            if score > best_score:
                best_score = score
                best_page = page_num
        
        return best_page if best_score > 0.1 else None

# Global instance for backward compatibility
page_navigation_service = PageNavigationService()

# Alias for existing code compatibility - maintains the same interface
# but removes all highlighting functionality
highlight_service = page_navigation_service
