from pydantic import BaseModel
from typing import Optional, List, Dict

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5  # Default to 5 results
    selected_files: Optional[List[str]] = None  # Filter by specific files

class FileProgress(BaseModel):
    filename: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int  # 0-100 for individual file
    size_mb: float
    error_message: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    estimated_completion_time: str
    files_count: int
    total_size_mb: float
    task_id: str
    files: List[FileProgress] = []

class ProcessingStatus(BaseModel):
    task_id: str
    status: str  # "processing", "completed", "failed"
    progress: int  # 0-100 overall progress
    message: str
    files_processed: int
    total_files: int
    files: List[FileProgress] = []
    estimated_time_remaining: Optional[str] = None
    current_file: Optional[str] = None

class WebSocketMessage(BaseModel):
    type: str  # "initial_status", "status_update", "files_extracted", "file_progress", "file_completed", "file_failed", "completed", "error"
    task_id: str
    message: str = ""
    progress: int = 0
    files_processed: int = 0
    total_files: int = 0
    file_index: Optional[int] = None
    filename: Optional[str] = None
    status: Optional[str] = None  # File status: "pending", "processing", "completed", "failed"
    files: List[FileProgress] = []
    estimated_time_remaining: Optional[str] = None
    overall_progress: Optional[int] = None
    failed_files: Optional[int] = None
    completed_files: Optional[int] = None
    error: Optional[str] = None
