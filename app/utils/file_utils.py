import os
import uuid
import zipfile
import tempfile
from typing import List, Tuple
from pathlib import Path
from app.config.settings import BASE_URL

# HIGHLIGHTING FUNCTIONALITY DISABLED - This cleanup function is no longer needed
# def cleanup_highlighted_pdfs(data_dir: str = "data") -> int:
#     """Remove all highlighted PDF files from the data directory"""
#     try:
#         if os.path.exists(data_dir):
#             files = os.listdir(data_dir)
#             highlighted_files = [f for f in files if f.startswith("highlighted_") and f.endswith(".pdf")]
#             for file in highlighted_files:
#                 file_path = os.path.join(data_dir, file)
#                 try:
#                     os.remove(file_path)
#                     print(f"Deleted highlighted PDF: {file}")
#                 except Exception as e:
#                     print(f"Error deleting {file}: {e}")
#             return len(highlighted_files)
#     except Exception as e:
#         print(f"Error during cleanup: {e}")
#         return 0

def save_uploaded_file(file_content: bytes, filename: str, data_dir: str = "data") -> str:
    """Save uploaded file and return the permanent path"""
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Save uploaded file temporarily
    temp_path = os.path.join(data_dir, f"{uuid.uuid4()}_{filename}")
    with open(temp_path, "wb") as f:
        f.write(file_content)
    
    # Save permanent copy
    base_name = os.path.basename(filename)
    permanent_path = os.path.join(data_dir, base_name)
    if os.path.exists(permanent_path):
        name, ext = os.path.splitext(base_name)
        permanent_path = os.path.join(data_dir, f"{name}_{uuid.uuid4().hex[:8]}{ext}")
    os.rename(temp_path, permanent_path)
    
    return permanent_path

def get_pdf_files(data_dir: str = "data") -> List[dict]:
    """Get list of PDF files in the data directory"""
    if not os.path.exists(data_dir):
        return []
    
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf") and not f.startswith("highlighted_")]
    books = [
        {
            "filename": f,
            "url": f"{BASE_URL}/static/{f}"
        }
        for f in pdf_files
    ]
    return books

def find_pdf_file(filename: str, source: str, data_dir: str = "data") -> str:
    """Find PDF file path by filename or source name"""
    if not os.path.exists(data_dir):
        return None
    
    # Try direct path first
    direct_path = os.path.join(data_dir, filename)
    if os.path.exists(direct_path) and filename.lower().endswith(".pdf") and not filename.startswith("highlighted_"):
        return direct_path
    
    # Fallback search
    def normalize(s):
        return os.path.splitext(s.lower().replace(' ', '').replace('_', ''))[0]
    
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf") and not f.startswith("highlighted_")]
    norm_filename = normalize(filename) if filename else ""
    norm_source = normalize(source) if source else ""
    
    for f in pdf_files:
        norm_f = normalize(f)
        if (norm_filename and norm_f == norm_filename) or \
           (norm_source and (norm_f == norm_source or norm_source in norm_f)):
            return os.path.join(data_dir, f)
    
    return None

def delete_file_and_get_ids(filename: str, all_docs: dict, data_dir: str = "data") -> tuple:
    """Delete file and return IDs to delete from collection"""
    # Remove the PDF file
    file_path = os.path.join(data_dir, filename)
    file_exists = False
    
    if os.path.exists(file_path):
        os.remove(file_path)
        file_exists = True
    
    # Get IDs to delete from collection
    name, _ = os.path.splitext(filename)
    ids_to_delete = [
        id_ for id_, meta in zip(all_docs['ids'], all_docs['metadatas']) 
        if meta.get('source') == name or meta.get('source') == filename
    ]
    
    return file_exists, ids_to_delete

def extract_pdfs_from_zip(zip_content: bytes, data_dir: str = "data") -> List[str]:
    """Extract PDF files from a ZIP archive and return list of extracted file paths"""
    extracted_files = []
    
    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "upload.zip")
        
        # Save ZIP content to temporary file
        with open(zip_path, "wb") as f:
            f.write(zip_content)
        
        # Extract PDF files
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.lower().endswith('.pdf') and not file_info.is_dir():
                    # Extract to temporary location first
                    zip_ref.extract(file_info, temp_dir)
                    temp_file_path = os.path.join(temp_dir, file_info.filename)
                    
                    # Read content and save to data directory
                    with open(temp_file_path, "rb") as pdf_file:
                        pdf_content = pdf_file.read()
                    
                    # Generate unique filename
                    base_name = os.path.basename(file_info.filename)
                    permanent_path = save_uploaded_file(pdf_content, base_name, data_dir)
                    extracted_files.append(permanent_path)
    
    return extracted_files

def get_file_size_mb(file_content: bytes) -> float:
    """Get file size in megabytes"""
    return len(file_content) / (1024 * 1024)

def estimate_processing_time(files_count: int, total_size_mb: float) -> str:
    """Estimate processing time based on file count and size with more realistic estimates for large books"""
    # More realistic time estimates based on actual processing experience with large files
    base_time_per_file = 2.5   # Base time per file (in minutes) - increased for realistic estimates
    time_per_mb = 1.2          # Additional time per MB - significantly increased for large files
    
    # Additional time adjustments based on file size characteristics
    if files_count > 0:
        avg_size_per_file = total_size_mb / files_count
        
        if avg_size_per_file < 0.5:  # Files smaller than 0.5MB
            base_time_per_file *= 0.6  # Reduce base time for very small files
            time_per_mb *= 0.4         # Reduce per-MB time for small files
        elif avg_size_per_file < 2:  # Files smaller than 2MB
            base_time_per_file *= 0.8  # Reduce base time for small files
            time_per_mb *= 0.7         # Reduce per-MB time for small files
        elif avg_size_per_file > 100:  # Very large files (>100MB)
            base_time_per_file *= 4.0  # Significantly increase base time for very large files
            time_per_mb *= 2.5         # Significantly increase per-MB time for very large files
        elif avg_size_per_file > 50:  # Very large files (>50MB)
            base_time_per_file *= 3.0  # Triple base time for very large files
            time_per_mb *= 2.2         # More than double per-MB time for large files
        elif avg_size_per_file > 20:  # Large files (>20MB)
            base_time_per_file *= 2.2  # More than double base time for large files
            time_per_mb *= 1.8         # Significantly increase per-MB time for large files
        elif avg_size_per_file > 10:  # Medium-large files (>10MB)
            base_time_per_file *= 1.8  # Increase base time for medium-large files
            time_per_mb *= 1.5         # Increase per-MB time for medium-large files
    
    # Factor in chunking and embedding overhead for large files (more realistic)
    if total_size_mb > 200:  # Very large total size
        embedding_overhead = total_size_mb * 0.8  # Significant additional overhead for embedding processing
    elif total_size_mb > 100:  # Very large total size
        embedding_overhead = total_size_mb * 0.6  # High additional overhead for embedding processing
    elif total_size_mb > 50:  # Large total size
        embedding_overhead = total_size_mb * 0.4  # Moderate additional overhead
    elif total_size_mb > 20:  # Medium total size
        embedding_overhead = total_size_mb * 0.3  # Some additional overhead
    else:
        embedding_overhead = total_size_mb * 0.15  # Minimal additional overhead
    
    # Add API rate limiting delays for large files
    if total_size_mb > 50:
        rate_limit_overhead = total_size_mb * 0.2  # Additional time for API rate limits
    else:
        rate_limit_overhead = 0
    
    estimated_minutes = (files_count * base_time_per_file) + (total_size_mb * time_per_mb) + embedding_overhead + rate_limit_overhead
    
    # Minimum processing time
    estimated_minutes = max(estimated_minutes, 1.0)  # At least 1 minute for any file
    
    if estimated_minutes < 2:
        return "1-2 minutes"
    elif estimated_minutes < 5:
        minutes = int(estimated_minutes)
        return f"About {minutes} minutes"
    elif estimated_minutes < 15:
        minutes = int(estimated_minutes)
        return f"{minutes-2}-{minutes+3} minutes"
    elif estimated_minutes < 30:
        minutes = int(estimated_minutes)
        return f"About {minutes} minutes"
    elif estimated_minutes < 60:
        minutes = int(estimated_minutes)
        return f"{minutes-5}-{minutes+10} minutes"
    else:
        hours = estimated_minutes // 60
        minutes = int(estimated_minutes % 60)
        if hours < 1.5:
            total_minutes = int(estimated_minutes)
            return f"About {total_minutes} minutes (1-2 hours)"
        elif hours < 3:
            return f"About 2-3 hours"
        elif hours < 5:
            hours_rounded = int(hours)
            return f"About {hours_rounded}-{hours_rounded+1} hours"
        else:
            hours_rounded = int(hours)
            if minutes > 30:
                hours_rounded += 1
            return f"About {hours_rounded} hours or more"

def is_zip_file(filename: str) -> bool:
    """Check if file is a ZIP archive"""
    return filename.lower().endswith('.zip')

def get_pdf_files_with_metadata(data_dir: str = "data") -> List[dict]:
    """Get list of PDF files with additional metadata"""
    if not os.path.exists(data_dir):
        return []
    
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf") and not f.startswith("highlighted_")]
    books = []
    
    for f in pdf_files:
        file_path = os.path.join(data_dir, f)
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        books.append({
            "filename": f,
            "url": f"{BASE_URL}/static/{f}",
            "size_mb": round(file_size_mb, 2),
            "display_name": os.path.splitext(f)[0].replace('_', ' ')
        })
    
    return books
