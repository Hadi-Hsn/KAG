import re
from typing import List
from difflib import SequenceMatcher

def normalize_text(text: str) -> str:
    """Normalize text for better matching by removing extra whitespace and standardizing punctuation"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Standardize quotes and dashes
    text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
    text = text.replace('–', '-').replace('—', '-')
    return text

def similarity_score(a: str, b: str) -> float:
    """Calculate similarity between two strings using SequenceMatcher"""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def find_best_position_match(chunk_text, page_positions):
    """Find the best matching position information for a text chunk"""
    if not page_positions:
        return None
    
    best_match = None
    best_score = 0
    
    # Normalize chunk text for comparison
    normalized_chunk = normalize_text(chunk_text)
    
    for pos_info in page_positions:
        normalized_pos_text = normalize_text(pos_info['text'])
        
        # Calculate overlap/similarity
        if normalized_chunk in normalized_pos_text:
            # Direct substring match - highest priority
            score = len(normalized_chunk) / len(normalized_pos_text)
            if score > best_score:
                best_score = score
                best_match = pos_info
        else:
            # Calculate similarity score
            similarity = similarity_score(normalized_chunk, normalized_pos_text)
            if similarity > 0.3 and similarity > best_score:
                best_score = similarity
                best_match = pos_info
    
    return best_match

# HIGHLIGHTING FUNCTIONALITY DISABLED - These functions are no longer used
# They are kept commented for potential future reference

# def extract_meaningful_phrases(text: str) -> List[str]:
#     """Extract meaningful phrases from text for highlighting"""
#     phrases = []
#     
#     # Split by sentences
#     sentences = re.split(r'[.!?]+', text)
#     for sentence in sentences:
#         sentence = sentence.strip()
#         if len(sentence) > 15:
#             phrases.append(sentence)
#             
#             # Split long sentences by commas
#             if len(sentence) > 50:
#                 parts = [part.strip() for part in sentence.split(',')]
#                 phrases.extend([part for part in parts if len(part) > 15])
#     
#     # Add noun phrases (simple extraction)
#     words = text.split()
#     for i in range(len(words) - 2):
#         phrase = ' '.join(words[i:i+3])
#         if len(phrase) > 15 and any(word[0].isupper() for word in words[i:i+3]):
#             phrases.append(phrase)
#     
#     return list(set(phrases))

# def extract_key_phrases(text: str) -> List[str]:
#     """Extract key phrases for fallback highlighting"""
#     phrases = []
#     
#     # Extract phrases between punctuation
#     parts = re.split(r'[,.;:()]+', text)
#     for part in parts:
#         part = part.strip()
#         if len(part) > 10:
#             phrases.append(part)
#     
#     # Extract quoted text
#     quoted = re.findall(r'"([^"]*)"', text)
#     phrases.extend([q for q in quoted if len(q) > 10])
#     
#     # Extract parenthetical text
#     parenthetical = re.findall(r'\(([^)]*)\)', text)
#     phrases.extend([p for p in parenthetical if len(p) > 10])
#     
#     return list(set(phrases))

def generate_text_variations(text: str) -> List[str]:
    """Generate variations of text for better matching"""
    variations = [text]
    
    # Add variations with different punctuation
    variations.append(text.replace(',', ''))
    variations.append(text.replace('.', ''))
    variations.append(text.replace(';', ''))
    
    # Add variations with different spacing
    variations.append(re.sub(r'\s+', ' ', text))
    variations.append(text.replace(' ', '  '))
    
    # Add variations with case differences
    variations.append(text.lower())
    variations.append(text.title())
    
    return list(set(variations))

def find_best_text_match(target_text: str, page_text: str, min_similarity: float = 0.7) -> List[str]:
    """Find the best matching phrases in page text for the target text"""
    target_normalized = normalize_text(target_text)
    page_normalized = normalize_text(page_text)
    
    # Split target into sentences and phrases
    sentences = re.split(r'[.!?]+', target_normalized)
    phrases = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Only meaningful sentences
            phrases.append(sentence)
            # Also add sub-phrases by splitting on commas
            sub_phrases = [p.strip() for p in sentence.split(',') if len(p.strip()) > 15]
            phrases.extend(sub_phrases)
    
    # Find best matches
    matches = []
    for phrase in phrases:
        if len(phrase) > 10:
            # Look for exact matches first
            if phrase in page_normalized:
                matches.append(phrase)
            else:
                # Look for similar phrases
                words = phrase.split()
                if len(words) >= 3:
                    # Try different word combinations
                    for i in range(len(words) - 2):
                        test_phrase = ' '.join(words[i:i+3])
                        if len(test_phrase) > 15 and test_phrase in page_normalized:
                            matches.append(test_phrase)
    
    return list(set(matches))  # Remove duplicates

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string using a more accurate approximation"""
    # More accurate approximation: ~3.2 characters per token for English text
    # This is more conservative to ensure we stay well under limits
    char_count = len(text)
    # Add extra margin for complex text (punctuation, formatting, etc.)
    estimated_tokens = int(char_count / 3.2)  # Conservative estimate
    return max(estimated_tokens, 1)  # At least 1 token

def split_text_by_tokens(text: str, max_tokens: int = 8000) -> List[str]:
    """Split text into chunks that don't exceed the token limit with more reasonable defaults"""
    chunks = []
    current_chunk = ""
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Estimate tokens for the current chunk plus this sentence
        potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
        estimated_tokens = estimate_tokens(potential_chunk)
        
        if estimated_tokens <= max_tokens:
            current_chunk = potential_chunk
        else:
            # If current chunk is not empty, save it and start a new one
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                # If even a single sentence is too long, split it by paragraphs/lines
                long_sentence_chunks = split_long_text(sentence, max_tokens)
                chunks.extend(long_sentence_chunks)
                current_chunk = ""
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def split_long_text(text: str, max_tokens: int) -> List[str]:
    """Split very long text (like a single long sentence) into smaller chunks"""
    chunks = []
    
    # Try splitting by paragraphs first
    paragraphs = text.split('\n')
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        potential_chunk = current_chunk + "\n" + paragraph if current_chunk else paragraph
        
        if estimate_tokens(potential_chunk) <= max_tokens:
            current_chunk = potential_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                # If even a single paragraph is too long, split by character count
                char_chunks = split_by_characters(paragraph, max_tokens * 3.2)  # ~3.2 chars per token
                chunks.extend(char_chunks)
                current_chunk = ""
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def split_by_characters(text: str, max_chars: int) -> List[str]:
    """Split text by character count as a last resort"""
    chunks = []
    for i in range(0, len(text), int(max_chars)):
        chunk = text[i:i + int(max_chars)]
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks
