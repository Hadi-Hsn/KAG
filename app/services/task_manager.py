"""
Task Manager Service for handling file processing tasks and their persistence.
"""
import json
import os
import time
import asyncio
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from app.models.schemas import FileProgress, WebSocketMessage, ProcessingStatus
from app.config.settings import DATA_DIR
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TaskState:
    """Enhanced task state with persistence support"""
    task_id: str
    status: str
    progress: int
    message: str
    files_processed: int
    total_files: int
    files: List[FileProgress]
    start_time: float
    estimated_time_remaining: Optional[str] = None
    current_file: Optional[str] = None
    completed_files: int = 0
    failed_files: int = 0
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class TaskManager:
    """Enhanced task manager with persistence and better WebSocket handling"""
    
    def __init__(self):
        self.tasks: Dict[str, TaskState] = {}
        self.websockets: Dict[str, any] = {}  # WebSocket connections
        self.task_persistence_file = os.path.join(DATA_DIR, "task_states.json")
        self.max_completed_tasks = 50  # Keep only last 50 completed tasks
        self.task_cleanup_interval = 3600  # Clean up old tasks every hour
        self._cleanup_task = None
        
        # Load persisted tasks on startup
        self.load_persisted_tasks()
    
    def load_persisted_tasks(self):
        """Load persisted task states from disk"""
        try:
            if os.path.exists(self.task_persistence_file):
                with open(self.task_persistence_file, 'r') as f:
                    data = json.load(f)
                    
                for task_id, task_data in data.items():
                    # Convert files list back to FileProgress objects
                    files = [FileProgress(**file_data) for file_data in task_data.get('files', [])]
                    task_data['files'] = files
                    
                    # Create TaskState object
                    self.tasks[task_id] = TaskState(**task_data)
                    
                logger.info(f"Loaded {len(self.tasks)} persisted tasks")
        except Exception as e:
            logger.error(f"Error loading persisted tasks: {e}")
    
    def save_tasks_to_disk(self):
        """Save current task states to disk"""
        try:
            # Convert TaskState objects to dictionaries for JSON serialization
            serializable_tasks = {}
            for task_id, task_state in self.tasks.items():
                task_dict = asdict(task_state)
                # Convert FileProgress objects to dicts
                task_dict['files'] = [file.model_dump() for file in task_state.files]
                serializable_tasks[task_id] = task_dict
            
            with open(self.task_persistence_file, 'w') as f:
                json.dump(serializable_tasks, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving tasks to disk: {e}")
    
    async def periodic_cleanup(self):
        """Periodically clean up old completed tasks"""
        while True:
            try:
                await asyncio.sleep(self.task_cleanup_interval)
                await self.cleanup_old_tasks()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def cleanup_old_tasks(self):
        """Remove old completed/failed tasks, keeping only recent ones"""
        current_time = time.time()
        old_task_cutoff = current_time - (24 * 3600)  # 24 hours
        
        # Find completed/failed tasks older than cutoff
        tasks_to_remove = []
        for task_id, task_state in self.tasks.items():
            if (task_state.status in ['completed', 'failed', 'completed_with_errors'] and 
                task_state.created_at < old_task_cutoff):
                tasks_to_remove.append(task_id)
        
        # Keep only the most recent completed tasks
        completed_tasks = [(task_id, task) for task_id, task in self.tasks.items() 
                          if task.status in ['completed', 'failed', 'completed_with_errors']]
        completed_tasks.sort(key=lambda x: x[1].created_at, reverse=True)
        
        if len(completed_tasks) > self.max_completed_tasks:
            tasks_to_remove.extend([task_id for task_id, _ in completed_tasks[self.max_completed_tasks:]])
        
        # Remove old tasks
        for task_id in tasks_to_remove:
            if task_id in self.tasks:
                del self.tasks[task_id]
            if task_id in self.websockets:
                del self.websockets[task_id]
        
        if tasks_to_remove:
            logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
            self.save_tasks_to_disk()
    
    def create_task(self, task_id: str, uploaded_files_data: List[Dict]) -> TaskState:
        """Create a new processing task"""
        file_progress_list = [
            FileProgress(
                filename=file_data["filename"],
                status="pending",
                progress=0,
                size_mb=file_data["size_mb"]
            ) for file_data in uploaded_files_data
        ]
        
        task_state = TaskState(
            task_id=task_id,
            status="processing",
            progress=0,
            message="Starting file processing...",
            files_processed=0,
            total_files=len(uploaded_files_data),
            files=file_progress_list,
            start_time=time.time()
        )
        
        self.tasks[task_id] = task_state
        self.save_tasks_to_disk()
        return task_state
    
    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Get task state by ID"""
        return self.tasks.get(task_id)
    
    def get_active_tasks(self) -> List[TaskState]:
        """Get all active (processing) tasks"""
        return [task for task in self.tasks.values() if task.status == "processing"]
    
    def get_recent_tasks(self, limit: int = 10) -> List[TaskState]:
        """Get recent tasks sorted by creation time"""
        all_tasks = sorted(self.tasks.values(), key=lambda x: x.created_at, reverse=True)
        return all_tasks[:limit]
    
    async def register_websocket(self, task_id: str, websocket):
        """Register a WebSocket connection for a task"""
        self.websockets[task_id] = websocket
        
        # Send current task state if task exists
        task_state = self.get_task(task_id)
        if task_state:
            initial_message = WebSocketMessage(
                type="initial_status",
                task_id=task_id,
                status=task_state.status,
                progress=task_state.progress,
                message=task_state.message,
                files_processed=task_state.files_processed,
                total_files=task_state.total_files,
                files=task_state.files,
                estimated_time_remaining=task_state.estimated_time_remaining,
                completed_files=task_state.completed_files,
                failed_files=task_state.failed_files
            )
            await self.send_websocket_message(task_id, initial_message)
    
    def unregister_websocket(self, task_id: str):
        """Unregister a WebSocket connection"""
        if task_id in self.websockets:
            del self.websockets[task_id]
    
    async def send_websocket_message(self, task_id: str, message: WebSocketMessage):
        """Send a message to the WebSocket client if connected"""
        if task_id in self.websockets:
            try:
                websocket = self.websockets[task_id]
                await websocket.send_text(message.model_dump_json())
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                # Remove disconnected websocket
                self.unregister_websocket(task_id)
    
    async def update_task_status(self, task_id: str, **kwargs):
        """Update task status and notify WebSocket clients"""
        if task_id in self.tasks:
            task_state = self.tasks[task_id]
            
            # Update task state
            for key, value in kwargs.items():
                if hasattr(task_state, key):
                    setattr(task_state, key, value)
            
            # Calculate estimated time remaining
            if task_state.progress > 0 and task_state.status == "processing":
                elapsed_time = time.time() - task_state.start_time
                if task_state.progress < 100:
                    estimated_total_time = (elapsed_time / task_state.progress) * 100
                    remaining_time = estimated_total_time - elapsed_time
                    if remaining_time > 0:
                        task_state.estimated_time_remaining = self.format_time(remaining_time)
            
            # Send WebSocket update with file statuses
            message = WebSocketMessage(
                type="status_update",
                task_id=task_id,
                status=task_state.status,
                progress=task_state.progress,
                message=task_state.message,
                files_processed=task_state.files_processed,
                total_files=task_state.total_files,
                files=task_state.files,  # Include current file statuses
                estimated_time_remaining=task_state.estimated_time_remaining
            )
            await self.send_websocket_message(task_id, message)
            
            # Save to disk
            self.save_tasks_to_disk()
    
    async def send_status_update(self, task_id: str, progress: int, message: str, 
                               files_processed: int, total_files: int):
        """Send a general status update via WebSocket"""
        if task_id in self.tasks:
            task_state = self.tasks[task_id]
            
            # Update task state
            task_state.progress = progress
            task_state.message = message
            task_state.files_processed = files_processed
            
            # Calculate estimated time remaining
            if progress > 0 and task_state.status == "processing":
                elapsed_time = time.time() - task_state.start_time
                if progress < 100:
                    estimated_total_time = (elapsed_time / progress) * 100
                    remaining_time = estimated_total_time - elapsed_time
                    if remaining_time > 0:
                        task_state.estimated_time_remaining = self.format_time(remaining_time)
            
            # Send WebSocket update with file statuses
            status_message = WebSocketMessage(
                type="status_update",
                task_id=task_id,
                status=task_state.status,
                progress=progress,
                message=message,
                files_processed=files_processed,
                total_files=total_files,
                files=task_state.files,  # Include current file statuses
                estimated_time_remaining=task_state.estimated_time_remaining
            )
            await self.send_websocket_message(task_id, status_message)
            
            # Save to disk
            self.save_tasks_to_disk()
    
    async def update_file_progress(self, task_id: str, file_index: int, status: str, 
                                 progress: int = 0, error: str = None, message: str = None):
        """Update individual file progress"""
        if task_id not in self.tasks or file_index >= len(self.tasks[task_id].files):
            return
        
        task_state = self.tasks[task_id]
        file_data = task_state.files[file_index]
        
        # Update file progress
        file_data.status = status
        file_data.progress = progress
        if error:
            file_data.error_message = error
        
        # Calculate overall progress
        completed_files = sum(1 for f in task_state.files if f.status in ["completed", "failed"])
        failed_files = sum(1 for f in task_state.files if f.status == "failed")
        overall_progress = int((completed_files / task_state.total_files) * 100)
        
        # Update task state
        task_state.progress = overall_progress
        task_state.files_processed = completed_files
        task_state.failed_files = failed_files
        task_state.completed_files = completed_files - failed_files
        task_state.current_file = file_data.filename if status == "processing" else None
        
        # Send WebSocket update
        default_message = f"Processing {file_data.filename}..." if status == "processing" else f"Completed {file_data.filename}"
        websocket_message = WebSocketMessage(
            type="file_progress",
            task_id=task_id,
            file_index=file_index,
            filename=file_data.filename,
            status=status,
            progress=progress,
            overall_progress=overall_progress,
            files_processed=completed_files,
            total_files=task_state.total_files,
            message=message if message else default_message,
            error=error
        )
        await self.send_websocket_message(task_id, websocket_message)
        
        # Send completion message for individual file
        if status in ["completed", "failed"]:
            completion_default_message = f"{'Completed' if status == 'completed' else 'Failed to process'} {file_data.filename}"
            completion_message = WebSocketMessage(
                type="file_completed" if status == "completed" else "file_failed",
                task_id=task_id,
                file_index=file_index,
                filename=file_data.filename,
                status=status,
                progress=progress,
                overall_progress=overall_progress,
                files_processed=completed_files,
                total_files=task_state.total_files,
                message=message if message else completion_default_message,
                error=error
            )
            await self.send_websocket_message(task_id, completion_message)
        
        # Save to disk
        self.save_tasks_to_disk()
    
    async def complete_task(self, task_id: str, completed_files: int, failed_files: int):
        """Mark task as completed"""
        if task_id in self.tasks:
            task_state = self.tasks[task_id]
            task_state.status = "completed_with_errors" if failed_files > 0 else "completed"
            task_state.progress = 100
            task_state.message = f"Processing completed! {completed_files} files processed successfully" + \
                               (f", {failed_files} files failed" if failed_files > 0 else "")
            task_state.completed_files = completed_files
            task_state.failed_files = failed_files
            task_state.current_file = None
            
            # Send completion message
            completion_message = WebSocketMessage(
                type="completed",
                task_id=task_id,
                message=task_state.message,
                progress=100,
                files_processed=task_state.files_processed,
                total_files=task_state.total_files,
                completed_files=completed_files,
                failed_files=failed_files
            )
            await self.send_websocket_message(task_id, completion_message)
            
            # Save to disk
            self.save_tasks_to_disk()
    
    async def fail_task(self, task_id: str, error_message: str):
        """Mark task as failed"""
        if task_id in self.tasks:
            task_state = self.tasks[task_id]
            task_state.status = "failed"
            task_state.message = f"Critical processing error: {error_message}"
            task_state.current_file = None
            
            error_message_ws = WebSocketMessage(
                type="error",
                task_id=task_id,
                message=task_state.message,
                error=error_message
            )
            await self.send_websocket_message(task_id, error_message_ws)
            
            # Save to disk
            self.save_tasks_to_disk()
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        task_state = self.tasks.get(task_id)
        if not task_state:
            return False
        
        if task_state.status != "processing":
            return False
        
        # Update task status to cancelled
        task_state.status = "cancelled"
        task_state.message = "Task cancelled by user"
        task_state.progress = 0
        
        # Send cancellation message via WebSocket
        await self.send_websocket_message(task_id, WebSocketMessage(
            type="error",
            task_id=task_id,
            message="Task cancelled by user",
            progress=0,
            files_processed=task_state.files_processed,
            total_files=task_state.total_files,
            completed_files=task_state.completed_files,
            failed_files=task_state.failed_files
        ))
        
        # Save to disk
        self.save_tasks_to_disk()
        
        logger.info(f"Task {task_id} cancelled by user")
        return True

    async def abort_all_tasks(self) -> Dict[str, any]:
        """Abort all ongoing tasks, remove them from storage, and collect files to delete"""
        cancelled_count = 0
        tasks_to_remove = []
        files_to_delete = []  # Collect files from aborted tasks
        
        for task_id, task_state in self.tasks.items():
            if task_state.status == "processing":
                # Collect file information for deletion
                for file_progress in task_state.files:
                    files_to_delete.append(file_progress.filename)
                
                # Cancel the task
                task_state.status = "cancelled"
                task_state.message = "Task aborted by user"
                task_state.progress = 0
                
                # Send cancellation message via WebSocket
                await self.send_websocket_message(task_id, WebSocketMessage(
                    type="error",
                    task_id=task_id,
                    message="Task aborted - all tasks cancelled",
                    progress=0,
                    files_processed=task_state.files_processed,
                    total_files=task_state.total_files,
                    completed_files=task_state.completed_files,
                    failed_files=task_state.failed_files
                ))
                
                cancelled_count += 1
                tasks_to_remove.append(task_id)
        
        # Remove cancelled tasks from memory and close WebSocket connections
        for task_id in tasks_to_remove:
            if task_id in self.tasks:
                del self.tasks[task_id]
            if task_id in self.websockets:
                try:
                    # Close WebSocket connection
                    websocket = self.websockets[task_id]
                    await websocket.close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket for task {task_id}: {e}")
                del self.websockets[task_id]
        
        # Save updated state to disk
        self.save_tasks_to_disk()
        
        logger.info(f"Aborted {cancelled_count} tasks and collected {len(files_to_delete)} files for deletion")
        return {
            "cancelled_count": cancelled_count,
            "files_to_delete": files_to_delete
        }
    
    def to_processing_status(self, task_state: TaskState) -> ProcessingStatus:
        """Convert TaskState to ProcessingStatus for API responses"""
        return ProcessingStatus(
            task_id=task_state.task_id,
            status=task_state.status,
            progress=task_state.progress,
            message=task_state.message,
            files_processed=task_state.files_processed,
            total_files=task_state.total_files,
            files=task_state.files,
            estimated_time_remaining=task_state.estimated_time_remaining,
            current_file=task_state.current_file
        )
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time duration in human-readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes}m {remaining_seconds}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def start_background_tasks(self):
        """Start background cleanup task if not already started"""
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                self._cleanup_task = asyncio.create_task(self.periodic_cleanup())
                logger.info("Started background cleanup task")
            except Exception as e:
                logger.error(f"Failed to start background cleanup task: {e}")

# Global task manager instance
task_manager = TaskManager()
