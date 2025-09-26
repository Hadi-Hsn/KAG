import openai
from typing import List
import re
from app.utils.text_utils import estimate_tokens, split_text_by_tokens

class EmbeddingService:
    @staticmethod
    def embed_texts(texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using OpenAI with token limit handling"""
        print(f"Starting embedding process for {len(texts)} text chunks")
        all_embeddings = []
        
        # Process texts in batches to handle token limits
        processed_texts = []
        
        # First, split any oversized chunks with much more conservative limits
        for text in texts:
            estimated_tokens = estimate_tokens(text)
            if estimated_tokens > 8000:  # Much more conservative limit for individual chunks
                print(f"Warning: Text chunk too large ({estimated_tokens} estimated tokens), splitting...")
                sub_chunks = split_text_by_tokens(text, max_tokens=7000)  # Very conservative sub-chunk size
                processed_texts.extend(sub_chunks)
            else:
                processed_texts.append(text)
        
        print(f"After preprocessing: {len(processed_texts)} text chunks (was {len(texts)})")
        
        # Now process in smaller batches to avoid hitting API limits
        # OpenAI's maximum batch size is around 5,400 items, so we use much smaller batches
        max_batch_size = 500  # Very conservative batch size to stay well under API limits
        max_tokens_per_batch = 20000  # Very conservative token limit per batch
        
        # Process in batches
        batch_count = 0
        for i in range(0, len(processed_texts), max_batch_size):
            batch_end = min(i + max_batch_size, len(processed_texts))
            batch_texts = processed_texts[i:batch_end]
            batch_count += 1
            
            print(f"Processing batch {batch_count}: {len(batch_texts)} texts (items {i+1} to {batch_end})")
            
            # Further split batch if it has too many tokens
            current_batch = []
            current_batch_tokens = 0
            sub_batch_count = 0
            
            for text in batch_texts:
                text_tokens = estimate_tokens(text)
                
                # If adding this text would exceed token limits, process current batch first
                if (current_batch and 
                    (current_batch_tokens + text_tokens > max_tokens_per_batch)):
                    
                    sub_batch_count += 1
                    print(f"  Processing sub-batch {sub_batch_count}: {len(current_batch)} texts, ~{current_batch_tokens} tokens")
                    
                    # Process current batch
                    batch_embeddings = EmbeddingService._embed_batch(current_batch)
                    all_embeddings.extend(batch_embeddings)
                    
                    # Start new batch
                    current_batch = [text]
                    current_batch_tokens = text_tokens
                else:
                    current_batch.append(text)
                    current_batch_tokens += text_tokens
            
            # Process remaining texts in current batch
            if current_batch:
                sub_batch_count += 1
                print(f"  Processing final sub-batch {sub_batch_count}: {len(current_batch)} texts, ~{current_batch_tokens} tokens")
                batch_embeddings = EmbeddingService._embed_batch(current_batch)
                all_embeddings.extend(batch_embeddings)
        
        print(f"Embedding process completed: generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    @staticmethod
    def _embed_batch(texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with enhanced error handling"""
        try:
            # Additional safety check for batch size
            if len(texts) > 1000:  # Much more conservative - stay well under the ~5,461 limit
                print(f"Warning: Batch size {len(texts)} is too large, splitting into smaller batches")
                # Split into smaller sub-batches
                all_embeddings = []
                sub_batch_size = 500  # Even smaller sub-batches
                for i in range(0, len(texts), sub_batch_size):
                    sub_batch = texts[i:i + sub_batch_size]
                    print(f"  Processing sub-batch with {len(sub_batch)} texts")
                    sub_embeddings = EmbeddingService._embed_batch(sub_batch)
                    all_embeddings.extend(sub_embeddings)
                return all_embeddings
            
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            error_str = str(e)
            print(f"Embedding batch failed: {e}")
            
            # Handle batch size exceeded errors specifically
            if "batch size" in error_str.lower() and "exceeds maximum" in error_str.lower():
                print(f"Batch size exceeded, splitting batch of {len(texts)} texts into smaller chunks")
                # Split the batch in half and retry
                mid_point = len(texts) // 2
                if mid_point == 0:
                    # Single text failed, return zero vector
                    return [[0.0] * 1536]
                
                first_half = texts[:mid_point]
                second_half = texts[mid_point:]
                
                first_embeddings = EmbeddingService._embed_batch(first_half)
                second_embeddings = EmbeddingService._embed_batch(second_half)
                
                return first_embeddings + second_embeddings
            
            # Handle specific token limit errors
            elif "max_tokens_per_request" in error_str or "token" in error_str.lower():
                print(f"Token limit exceeded for batch of {len(texts)} texts, trying individual processing...")
                # Process texts individually when batch fails due to token limits
                embeddings = []
                for i, text in enumerate(texts):
                    try:
                        # Try to split large individual texts first
                        estimated_tokens = estimate_tokens(text)
                        if estimated_tokens > 5000:  # Much more conservative for individual texts
                            print(f"Text {i+1} too large ({estimated_tokens} tokens), splitting...")
                            from app.utils.text_utils import split_text_by_tokens
                            sub_texts = split_text_by_tokens(text, max_tokens=4000)  # Much smaller chunks
                            
                            # Generate embeddings for sub-texts and average them
                            sub_embeddings = []
                            for sub_text in sub_texts:
                                if len(sub_text.strip()) > 10:  # Only process meaningful text
                                    response = openai.embeddings.create(
                                        model="text-embedding-3-small",
                                        input=[sub_text]
                                    )
                                    sub_embeddings.append(response.data[0].embedding)
                            
                            if sub_embeddings:
                                # Average the embeddings
                                avg_embedding = [sum(values) / len(values) for values in zip(*sub_embeddings)]
                                embeddings.append(avg_embedding)
                            else:
                                # Fallback to zero vector
                                embeddings.append([0.0] * 1536)
                        else:
                            # Normal processing for reasonably sized texts
                            response = openai.embeddings.create(
                                model="text-embedding-3-small",
                                input=[text]
                            )
                            embeddings.append(response.data[0].embedding)
                    except Exception as individual_error:
                        print(f"Individual embedding failed for text {i+1}: {individual_error}")
                        # Return a zero vector as fallback
                        embeddings.append([0.0] * 1536)  # text-embedding-3-small dimension
                return embeddings
            else:
                # For other types of errors, try individual processing as fallback
                print("Trying individual processing as fallback...")
                embeddings = []
                for text in texts:
                    try:
                        response = openai.embeddings.create(
                            model="text-embedding-3-small",
                            input=[text]
                        )
                        embeddings.append(response.data[0].embedding)
                    except Exception as individual_error:
                        print(f"Individual embedding failed: {individual_error}")
                        # Return a zero vector as fallback
                        embeddings.append([0.0] * 1536)  # text-embedding-3-small dimension
                return embeddings
    
    @staticmethod
    def clean_markdown(md: str) -> str:
        # Normalize all newlines to \n
        md = md.replace('\r\n', '\n').replace('\r', '\n')
        # Remove leading/trailing whitespace
        md = md.strip()
        # Replace 3+ consecutive newlines (with or without spaces) with just 2
        md = re.sub(r'([ \t]*\n){3,}', '\n\n', md)
        # Remove any blank lines at the start or end again (in case)
        md = re.sub(r'^(\s*\n)+|(\n\s*)+$', '', md)
        return md

    @staticmethod
    def generate_answer(query: str, context_chunks: List[str]) -> str:
        """Generate an answer using OpenAI based on the query and context"""
        context = "\n\n".join(context_chunks)
        
        prompt = f"""Based on the following context from medical/educational documents, please provide a comprehensive and accurate answer to the question.

Context:
{context}

Question: {query}

Please provide a detailed answer based solely on the information provided in the context. If the context doesn't contain enough information to fully answer the question, please state that clearly."""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful medical education assistant. Provide accurate, well-structured answers based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            answer = response.choices[0].message.content
            return EmbeddingService.clean_markdown(answer)
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# Global instance
embedding_service = EmbeddingService()
