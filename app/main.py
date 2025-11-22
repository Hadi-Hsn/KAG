import os
import uuid
import fitz
import asyncio
import json
import time
import logging
from collections import defaultdict
from typing import List, Optional, Dict
from fastapi import FastAPI, UploadFile, Form, Query, File, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Setup logging
logger = logging.getLogger(__name__)

# Import our organized modules
from app.config.settings import DATA_DIR, BASE_URL
from app.models.schemas import QueryRequest, UploadResponse, ProcessingStatus, FileProgress, WebSocketMessage
from app.services.chromadb_service import chromadb_service
from app.services.pdf_service import pdf_service, text_positions_cache
from app.services.embedding_service import embedding_service
from app.services.highlight_service import highlight_service
from app.services.task_manager import task_manager
from app.services.notification_service import notification_service
from app.utils.file_utils import (
    save_uploaded_file, 
    get_pdf_files_with_metadata, 
    find_pdf_file,
    delete_file_and_get_ids,
    extract_pdfs_from_zip,
    get_file_size_mb,
    estimate_processing_time,
    is_zip_file
)

# Initialize FastAPI app
app = FastAPI(title="KAG - Knowledge Assisted Generation", version="2.0.0")

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on startup"""
    task_manager.start_background_tasks()

# Mount the 'data' directory as a static files directory
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

# Get the project root directory (parent of app directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Serve HTML files
@app.get("/")
async def read_root():
    """Serve the main index.html file"""
    index_path = os.path.join(PROJECT_ROOT, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": f"index.html not found at {index_path}"}

@app.get("/pdf-viewer.html")
async def read_pdf_viewer():
    """Serve the pdf-viewer.html file"""
    viewer_path = os.path.join(PROJECT_ROOT, "pdf-viewer.html")
    if os.path.exists(viewer_path):
        return FileResponse(viewer_path)
    return {"error": f"pdf-viewer.html not found at {viewer_path}"}

@app.get("/index.html")
async def read_index():
    """Serve the index.html file"""
    index_path = os.path.join(PROJECT_ROOT, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": f"index.html not found at {index_path}"}

# CORS (for Postman or any client)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def process_single_file(task_id: str, file_index: int, file_data: Dict, completed_files_counter: List[int], failed_files_counter: List[int], total_files: int):
    """Process a single file asynchronously"""
    try:
        file_path = file_data["file_path"]
        filename = file_data["filename"]
        
        # Check if task has been cancelled
        current_task = task_manager.get_task(task_id)
        if current_task and current_task.status == "cancelled":
            print(f"Task {task_id} was cancelled, stopping processing for {filename}")
            return
        
        # Notify starting this file
        await task_manager.update_file_progress(
            task_id, file_index, "processing", 5, None,
            f"Starting to process {filename}..."
        )
        
        # Extract text with positions (wrapped in thread to avoid blocking)
        await task_manager.update_file_progress(
            task_id, file_index, "processing", 25, None,
            f"Extracting text from {filename}..."
        )
        text, reader_info = await asyncio.to_thread(
            pdf_service.extract_text_with_positions, file_path
        )
        
        # Chunk text with position awareness (wrapped in thread)
        await task_manager.update_file_progress(
            task_id, file_index, "processing", 45, None,
            f"Processing text chunks for {filename}..."
        )
        chunks = await asyncio.to_thread(
            pdf_service.chunk_text_with_positions, text, filename
        )
        
        # Validate chunks before proceeding
        if not chunks:
            raise Exception(f"No text chunks could be created from {filename}")
        
        # Filter out very small or empty chunks
        meaningful_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        if not meaningful_chunks:
            raise Exception(f"No meaningful text chunks found in {filename}")
        
        # Generate embeddings (wrapped in thread)
        await task_manager.update_file_progress(
            task_id, file_index, "processing", 70, None,
            f"Generating embeddings for {filename} ({len(meaningful_chunks)} chunks)..."
        )
        
        print(f"Processing {len(meaningful_chunks)} meaningful chunks for {filename} (filtered from {len(chunks)} total)")
        
        try:
            embeddings = await asyncio.to_thread(
                embedding_service.embed_texts, meaningful_chunks
            )
            
            # Verify that chunks and embeddings have the same length
            if len(meaningful_chunks) != len(embeddings):
                print(f"Warning: Chunks ({len(meaningful_chunks)}) and embeddings ({len(embeddings)}) length mismatch for {filename}")
                # Adjust to the minimum length to avoid errors
                min_length = min(len(meaningful_chunks), len(embeddings))
                meaningful_chunks = meaningful_chunks[:min_length]
                embeddings = embeddings[:min_length]
                print(f"Adjusted to {min_length} items for {filename}")
                
        except Exception as embedding_error:
            error_msg = f"Embedding generation failed for {filename}: {str(embedding_error)}"
            print(error_msg)
            
            # If it's a token limit error, try with even smaller chunks
            if "max_tokens_per_request" in str(embedding_error) or "token" in str(embedding_error).lower():
                print(f"Retrying {filename} with smaller chunks due to token limit error...")
                await task_manager.update_file_progress(
                    task_id, file_index, "processing", 75, None,
                    f"Retrying {filename} with smaller chunks..."
                )
                
                # Split chunks further and retry with multiple attempts
                smaller_chunks = []
                for chunk in meaningful_chunks:
                    from app.utils.text_utils import split_text_by_tokens, estimate_tokens
                    
                    # Try progressively smaller chunk sizes
                    chunk_tokens = estimate_tokens(chunk)
                    if chunk_tokens > 5000:  # Much smaller threshold
                        sub_chunks = split_text_by_tokens(chunk, max_tokens=4000)  # Much smaller chunks
                    else:
                        sub_chunks = split_text_by_tokens(chunk, max_tokens=5000)  # Still conservative
                    
                    smaller_chunks.extend([sc for sc in sub_chunks if len(sc.strip()) > 30])
                
                if smaller_chunks:
                    try:
                        print(f"Attempting to process {len(smaller_chunks)} smaller chunks for {filename}")
                        embeddings = await asyncio.to_thread(
                            embedding_service.embed_texts, smaller_chunks
                        )
                        meaningful_chunks = smaller_chunks
                        print(f"Successfully processed {filename} with {len(smaller_chunks)} smaller chunks")
                    except Exception as retry_error:
                        # Final attempt with micro-chunks
                        print(f"Second retry failed, attempting micro-chunks for {filename}...")
                        await task_manager.update_file_progress(
                            task_id, file_index, "processing", 80, None,
                            f"Final attempt with micro-chunks for {filename}..."
                        )
                        
                        micro_chunks = []
                        for chunk in smaller_chunks:
                            chunk_tokens = estimate_tokens(chunk)
                            if chunk_tokens > 8000:  # Micro chunks
                                sub_chunks = split_text_by_tokens(chunk, max_tokens=7000)
                                micro_chunks.extend([sc for sc in sub_chunks if len(sc.strip()) > 20])
                            else:
                                micro_chunks.append(chunk)
                        
                        if micro_chunks:
                            try:
                                print(f"Processing {len(micro_chunks)} micro-chunks for {filename}")
                                embeddings = await asyncio.to_thread(
                                    embedding_service.embed_texts, micro_chunks
                                )
                                meaningful_chunks = micro_chunks
                                print(f"Successfully processed {filename} with {len(micro_chunks)} micro-chunks")
                            except Exception as final_error:
                                raise Exception(f"All retry attempts failed for {filename}: {str(final_error)}")
                        else:
                            raise Exception(f"Could not create valid micro-chunks for {filename}")
                else:
                    raise Exception(f"Could not create valid smaller chunks for {filename}")
            else:
                raise Exception(error_msg)

        # Store in ChromaDB (wrapped in thread)
        await task_manager.update_file_progress(
            task_id, file_index, "processing", 90, None,
            f"Storing {filename} in database..."
        )
        
        if len(meaningful_chunks) == 0:
            raise Exception(f"No valid chunks generated for {filename}")
            
        ids = [str(uuid.uuid4()) for _ in meaningful_chunks]
        base_name = os.path.splitext(filename)[0]
        
        # Create metadata with page numbers from position cache
        metadata = []
        chunk_positions = text_positions_cache.get(f"{filename}_chunks", {})
        
        for chunk in meaningful_chunks:
            chunk_meta = {"source": base_name, "filename": filename}
            
            # Try to get page number from position info
            position_info = chunk_positions.get(chunk)
            if position_info:
                if isinstance(position_info, dict) and 'page' in position_info:
                    chunk_meta["page_number"] = position_info['page'] + 1  # Store as 1-based
                elif isinstance(position_info, (int, float)):
                    chunk_meta["page_number"] = int(position_info) + 1  # Store as 1-based
                    
            metadata.append(chunk_meta)

        await asyncio.to_thread(
            chromadb_service.add_documents, meaningful_chunks, embeddings, ids, metadata
        )
        
        # Mark file as completed
        await task_manager.update_file_progress(
            task_id, file_index, "completed", 100, None,
            f"✅ {filename} processed successfully"
        )
        completed_files_counter[0] += 1
        
        # Send overall progress update
        total_processed = completed_files_counter[0] + failed_files_counter[0]
        overall_progress = int(total_processed / total_files * 100)
        await task_manager.send_status_update(
            task_id,
            overall_progress,
            f"Processed {total_processed}/{total_files} files",
            total_processed,
            total_files
        )
        
    except Exception as e:
        error_str = str(e)
        
        # Provide more specific error messages for common issues
        if "batch size" in error_str.lower() and "exceeds" in error_str.lower():
            error_msg = f"Failed to process {file_data['filename']}: Large PDF requires batch processing (this should be handled automatically)"
        elif "token" in error_str.lower() and ("limit" in error_str.lower() or "max" in error_str.lower()):
            error_msg = f"Failed to process {file_data['filename']}: Text content too large for embedding processing"
        else:
            error_msg = f"Failed to process {file_data['filename']}: {error_str}"
            
        print(error_msg)
        await task_manager.update_file_progress(
            task_id, file_index, "failed", 0, error_msg, f"❌ {file_data['filename']} processing failed"
        )
        failed_files_counter[0] += 1
        
        # Send overall progress update after failed file
        total_processed = completed_files_counter[0] + failed_files_counter[0]
        overall_progress = int(total_processed / total_files * 100)
        await task_manager.send_status_update(
            task_id,
            overall_progress,
            f"Processed {total_processed}/{total_files} files ({failed_files_counter[0]} failed)",
            total_processed,
            total_files
        )

async def process_files_with_websocket(task_id: str, uploaded_files_data: List[Dict]):
    """Process uploaded files asynchronously in parallel with WebSocket updates"""
    try:
        # Create task in task manager
        task_state = task_manager.create_task(task_id, uploaded_files_data)
        
        # Send initial status
        await task_manager.send_status_update(
            task_id, 
            0, 
            "Starting file processing...", 
            0, 
            len(uploaded_files_data)
        )
        
        # Use lists as counters that can be shared across concurrent tasks
        completed_files_counter = [0]
        failed_files_counter = [0]
        
        # Create tasks for parallel processing of all files
        file_tasks = []
        for i, file_data in enumerate(uploaded_files_data):
            task = asyncio.create_task(
                process_single_file(
                    task_id, i, file_data, 
                    completed_files_counter, failed_files_counter, 
                    len(uploaded_files_data)
                )
            )
            file_tasks.append(task)
        
        # Wait for all files to be processed concurrently
        await asyncio.gather(*file_tasks, return_exceptions=True)
        
        # Complete the task
        await task_manager.complete_task(task_id, completed_files_counter[0], failed_files_counter[0])
        
    except Exception as e:
        error_msg = f"Critical error in file processing: {str(e)}"
        print(error_msg)
        await task_manager.fail_task(task_id, error_msg)

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await websocket.accept()
    
    try:
        # Register WebSocket with task manager
        await task_manager.register_websocket(task_id, websocket)
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for any message from client (optional, for ping/pong)
                data = await websocket.receive_text()
                # Echo back or handle client messages if needed
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        # Clean up when client disconnects
        task_manager.unregister_websocket(task_id)

@app.get("/")
def read_index():
    return FileResponse("index.html")

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring and load balancing"""
    try:
        # Test ChromaDB connection
        chromadb_status = "ok"
        try:
            chromadb_service.get_collection().count()
        except Exception as e:
            chromadb_status = f"error: {str(e)}"
        
        # Check disk space
        import shutil
        disk_usage = shutil.disk_usage(DATA_DIR)
        free_space_gb = disk_usage.free / (1024**3)
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "chromadb": chromadb_status,
                "disk_space_gb": round(free_space_gb, 2)
            },
            "version": "2.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "version": "2.0.0"
        }

@app.post("/upload-files", response_model=UploadResponse)
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
):
    """Upload PDF files or ZIP archives for processing"""
    try:
        uploaded_files = []
        uploaded_files_data = []
        total_size_mb = 0
        
        for file in files:
            file_content = await file.read()
            file_size_mb = get_file_size_mb(file_content)
            total_size_mb += file_size_mb
            
            if is_zip_file(file.filename):
                # Extract PDFs from ZIP (wrapped in thread to avoid blocking)
                extracted_files = await asyncio.to_thread(
                    extract_pdfs_from_zip, file_content, DATA_DIR
                )
                for extracted_file in extracted_files:
                    filename = os.path.basename(extracted_file)
                    # Get actual file size for extracted files
                    extracted_size_mb = await asyncio.to_thread(
                        lambda path: os.path.getsize(path) / (1024 * 1024), extracted_file
                    )
                    
                    uploaded_files.append(extracted_file)
                    uploaded_files_data.append({
                        "file_path": extracted_file,
                        "filename": filename,
                        "size_mb": extracted_size_mb
                    })
            elif file.filename.lower().endswith('.pdf'):
                # Save single PDF (wrapped in thread)
                permanent_path = await asyncio.to_thread(
                    save_uploaded_file, file_content, file.filename, DATA_DIR
                )
                uploaded_files.append(permanent_path)
                uploaded_files_data.append({
                    "file_path": permanent_path,
                    "filename": file.filename,
                    "size_mb": file_size_mb
                })
            else:
                continue  # Skip non-PDF, non-ZIP files
        
        if not uploaded_files:
            return UploadResponse(
                message="No valid PDF files found",
                estimated_completion_time="N/A",
                files_count=0,
                total_size_mb=0,
                task_id="",
                files=[]
            )
        
        # Generate task ID and start background processing immediately
        task_id = str(uuid.uuid4())
        
        # Create FileProgress objects for response
        file_progress_list = [
            FileProgress(
                filename=data["filename"],
                status="pending",
                progress=0,
                size_mb=data["size_mb"]
            ) for data in uploaded_files_data
        ]
        
        # Start background processing immediately using asyncio.create_task
        asyncio.create_task(process_files_with_websocket(task_id, uploaded_files_data))
        
        estimated_time = estimate_processing_time(len(uploaded_files), total_size_mb)
        
        # Send notification about file upload (don't await to avoid blocking)
        client_ip = request.client.host if request.client else "Unknown"
        asyncio.create_task(notification_service.notify_file_upload(
            len(uploaded_files), total_size_mb, client_ip
        ))
        
        return UploadResponse(
            message=f"Upload successful! Processing {len(uploaded_files)} files in background.",
            estimated_completion_time=estimated_time,
            files_count=len(uploaded_files),
            total_size_mb=round(total_size_mb, 2),
            task_id=task_id,
            files=file_progress_list
        )
        
    except Exception as e:
        return UploadResponse(
            message=f"Upload failed: {str(e)}",
            estimated_completion_time="N/A",
            files_count=0,
            total_size_mb=0,
            task_id="",
            files=[]
        )

@app.get("/processing-status/{task_id}", response_model=ProcessingStatus)
async def get_processing_status(task_id: str):
    """Get the processing status of an upload task"""
    task_state = task_manager.get_task(task_id)
    
    if not task_state:
        return ProcessingStatus(
            task_id=task_id,
            status="not_found",
            progress=0,
            message="Task not found",
            files_processed=0,
            total_files=0,
            files=[]
        )
    
    return task_manager.to_processing_status(task_state)

@app.post("/add-book")
async def add_book(file: UploadFile, source_name: str = Form(...)):
    """Add a PDF book to the knowledge base (legacy endpoint)"""
    # Save uploaded file
    file_content = await file.read()
    permanent_path = save_uploaded_file(file_content, file.filename, DATA_DIR)
    
    # Extract text with positions
    text, reader_info = pdf_service.extract_text_with_positions(permanent_path)
    
    # Chunk text with position awareness
    filename = os.path.basename(permanent_path)
    chunks = pdf_service.chunk_text_with_positions(text, filename)
    embeddings = embedding_service.embed_texts(chunks)

    # Store in ChromaDB
    ids = [str(uuid.uuid4()) for _ in chunks]
    actual_filename = os.path.basename(permanent_path)
    
    # Create metadata with page numbers from position cache
    metadata = []
    chunk_positions = text_positions_cache.get(f"{actual_filename}_chunks", {})
    
    for chunk in chunks:
        chunk_meta = {"source": source_name, "filename": actual_filename}
        
        # Try to get page number from position info
        position_info = chunk_positions.get(chunk)
        if position_info:
            if isinstance(position_info, dict) and 'page' in position_info:
                chunk_meta["page_number"] = position_info['page'] + 1  # Store as 1-based
            elif isinstance(position_info, (int, float)):
                chunk_meta["page_number"] = int(position_info) + 1  # Store as 1-based
                
        metadata.append(chunk_meta)

    chromadb_service.add_documents(chunks, embeddings, ids, metadata)
    
    return {
        "message": f"Book '{source_name}' added with {len(chunks)} chunks.",
        "extraction_details": {
            "total_chunks": len(chunks),
            "tables_found": reader_info.get('tables_found', 0),
            "extraction_method": reader_info.get('extraction_method', 'Unknown'),
            "pages_processed": reader_info.get('num_pages', 0),
            "positions_stored": reader_info.get('positions_stored', 0)
        },
        "first_chunk": chunks[0] if chunks else None,
        "reader_info": reader_info,
        "sample_chunks": chunks[:5] if len(chunks) >= 5 else chunks
    }

@app.post("/query")
async def query_book(req: QueryRequest, request: Request):
    """Query the knowledge base and return results with page numbers (highlighting functionality disabled)"""    
    # Get embeddings and query ChromaDB
    query_embedding = embedding_service.embed_texts([req.query])[0]
    results = chromadb_service.query_documents([query_embedding], req.top_k)
    
    # Filter results by selected files if specified
    if req.selected_files:
        filtered_documents = []
        filtered_metadatas = []
        filtered_distances = []
        
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            filename = meta.get("filename", "")
            source = meta.get("source", "")
            
            # Check if the file is in the selected files list
            file_matches = any(
                selected_file in filename or 
                selected_file in source or
                filename.startswith(selected_file) or
                source.startswith(selected_file)
                for selected_file in req.selected_files
            )
            
            if file_matches:
                filtered_documents.append(doc)
                filtered_metadatas.append(meta)
                filtered_distances.append(dist)
        
        # Update results with filtered data
        results = {
            "documents": [filtered_documents],
            "metadatas": [filtered_metadatas], 
            "distances": [filtered_distances]
        }
    
    matches = [doc for doc in results["documents"][0]]
    metadatas = results["metadatas"][0]
    scores = results["distances"][0]
    
    # Generate answer using the top 5 matches
    answer = embedding_service.generate_answer(req.query, matches[:5])
    
    # Group matches by filename
    file_matches = defaultdict(lambda: {"matches": [], "scores": [], "metas": [], "source": None})
    for match, score, meta in zip(matches, scores, metadatas):
        filename = meta.get("filename", meta.get("source"))
        file_matches[filename]["matches"].append(match)
        file_matches[filename]["scores"].append(score)
        file_matches[filename]["metas"].append(meta)
        if file_matches[filename]["source"] is None:
            file_matches[filename]["source"] = meta.get("source")

    # highlighted_pdfs = []  # Commented out since we no longer create highlighted PDFs
    all_results = []
    
    for filename, group in file_matches.items():
        # Find PDF file
        source = group["source"]
        pdf_path = find_pdf_file(filename, source, DATA_DIR)

        if not pdf_path:
            # Add results without PDF
            for match, score, meta in zip(group["matches"], group["scores"], group["metas"]):
                all_results.append({
                    "text": match,
                    "score": score,
                    "metadata": meta,
                    "source": source,
                    "pdf_url": None,
                    "page_number": None
                })
            continue
            
        # Process PDF page number detection for navigation
        doc = fitz.open(pdf_path)
        page_numbers = []
        
        # Find page numbers for each matching text chunk
        for i, match in enumerate(group["matches"]):
            meta = group["metas"][i]
            
            # First try to get page number from metadata (faster)
            page_num = meta.get("page_number")
            
            # If not in metadata, use the highlight service to find it
            if not page_num:
                page_num = highlight_service.find_page_number_for_text(doc, match, filename)
                
            page_numbers.append(page_num)
        
        # Use original PDF URL for navigation
        original_pdf_url = f"{BASE_URL}/static/{os.path.basename(pdf_path)}"
        
        doc.close()
        
        # Add results with enhanced information
        for i, (match, score, meta) in enumerate(zip(group["matches"], group["scores"], group["metas"])):
            page_num = page_numbers[i] if i < len(page_numbers) else None
            all_results.append({
                "text": match,
                "score": score,
                "metadata": meta,
                "source": source,
                "pdf_url": original_pdf_url,  # Use original PDF instead of highlighted version
                "page_number": page_num
            })

    # Send notification about query (don't await to avoid blocking)
    client_ip = request.client.host if request.client else "Unknown"
    asyncio.create_task(notification_service.notify_query_request(
        req.query, client_ip, len(all_results[:5])
    ))

    return {
        "query": req.query,
        "answer": answer,
        "references": all_results[:5],
        # "highlighted_pdfs": highlighted_pdfs  # Commented out since we no longer create highlighted PDFs
    }

@app.get("/all-embeddings")
def get_all_embeddings():
    """Get all stored document embeddings"""
    results = chromadb_service.get_all_documents()
    return {"documents": results["documents"]}

@app.get("/all-books")
def get_all_books():
    """Get list of all uploaded books with metadata"""
    books = get_pdf_files_with_metadata(DATA_DIR)
    return {"books": books}

@app.delete("/delete-book")
def delete_book(filename: str = Query(...)):
    """Delete a book and its embeddings"""
    # Get all documents to find IDs to delete
    all_docs = chromadb_service.get_all_documents()
    
    # Delete file and get IDs
    file_exists, ids_to_delete = delete_file_and_get_ids(filename, all_docs, DATA_DIR)
    
    if not file_exists:
        return {"error": "File not found."}
    
    # Remove embeddings from collection
    if ids_to_delete:
        chromadb_service.delete_documents(ids_to_delete)
    
    return {"message": f"Deleted {filename} and its embeddings."}

@app.post("/reset-database")
def reset_database():
    """Reset the entire ChromaDB database and delete all uploaded files"""
    try:
        # Reset ChromaDB collection
        reset_success = chromadb_service.reset_collection()
        
        if not reset_success:
            return {"error": "Failed to reset ChromaDB collection"}
        
        # Delete all files in the data directory
        import shutil
        import os
        
        files_deleted = 0
        if os.path.exists(DATA_DIR):
            for filename in os.listdir(DATA_DIR):
                file_path = os.path.join(DATA_DIR, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    files_deleted += 1
        
        return {
            "message": "Database reset successfully",
            "chromadb_reset": True,
            "files_deleted": files_deleted
        }
    except Exception as e:
        return {"error": f"Failed to reset database: {str(e)}"}

@app.post("/abort-all-tasks")
async def abort_all_tasks():
    """Abort all ongoing processing tasks, remove them from storage, and delete uploaded files"""
    try:
        abort_result = await task_manager.abort_all_tasks()
        cancelled_count = abort_result["cancelled_count"]
        files_to_delete = abort_result["files_to_delete"]
        
        # Delete uploaded files from aborted tasks
        deleted_files_count = 0
        deletion_errors = []
        
        for filename in files_to_delete:
            try:
                file_path = os.path.join(DATA_DIR, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files_count += 1
                    logger.info(f"Deleted file: {filename}")
                else:
                    logger.warning(f"File not found for deletion: {filename}")
            except Exception as e:
                error_msg = f"Failed to delete {filename}: {str(e)}"
                logger.error(error_msg)
                deletion_errors.append(error_msg)
        
        # Prepare response message
        message_parts = [f"Successfully aborted {cancelled_count} tasks"]
        if deleted_files_count > 0:
            message_parts.append(f"deleted {deleted_files_count} uploaded files")
        if deletion_errors:
            message_parts.append(f"but encountered {len(deletion_errors)} file deletion errors")
        
        message = " and ".join(message_parts) + "."
        
        response = {
            "message": message,
            "cancelled_count": cancelled_count,
            "deleted_files_count": deleted_files_count,
            "status": "success"
        }
        
        # Include deletion errors in response if any
        if deletion_errors:
            response["deletion_errors"] = deletion_errors
            response["status"] = "partial_success"
        
        return response
        
    except Exception as e:
        logger.error(f"Error aborting tasks: {e}")
        return {
            "error": f"Failed to abort tasks: {str(e)}",
            "status": "error"
        }

@app.get("/database-info")
def get_database_info():
    """Get information about the current database state"""
    try:
        # Get ChromaDB info
        chroma_info = chromadb_service.get_collection_info()
        
        # Get files info
        import os
        files_count = 0
        if os.path.exists(DATA_DIR):
            files_count = len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
        
        return {
            "chromadb": chroma_info,
            "data_directory": {
                "path": DATA_DIR,
                "files_count": files_count
            }
        }
    except Exception as e:
        return {"error": f"Failed to get database info: {str(e)}"}

@app.post("/cancel-task/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task"""
    try:
        success = task_manager.cancel_task(task_id)
        if success:
            return {"message": f"Task {task_id} has been cancelled", "success": True}
        else:
            return {"message": f"Task {task_id} not found or already completed", "success": False}
    except Exception as e:
        return {"message": f"Error cancelling task: {str(e)}", "success": False}

@app.get("/active-tasks")
async def get_active_tasks():
    """Get all active (processing) tasks"""
    active_tasks = task_manager.get_active_tasks()
    return {
        "active_tasks": [task_manager.to_processing_status(task).model_dump() for task in active_tasks],
        "count": len(active_tasks)
    }

@app.get("/recent-tasks")
async def get_recent_tasks(limit: int = Query(10, description="Number of recent tasks to return")):
    """Get recent tasks (useful for reconnecting after page refresh)"""
    recent_tasks = task_manager.get_recent_tasks(limit)
    return {
        "recent_tasks": [task_manager.to_processing_status(task).model_dump() for task in recent_tasks],
        "count": len(recent_tasks)
    }

@app.get("/task-details/{task_id}")
async def get_task_details(task_id: str):
    """Get detailed information about a specific task"""
    task_state = task_manager.get_task(task_id)
    
    if not task_state:
        return {"error": "Task not found"}
    
    return {
        "task_id": task_state.task_id,
        "status": task_state.status,
        "progress": task_state.progress,
        "message": task_state.message,
        "files_processed": task_state.files_processed,
        "total_files": task_state.total_files,
        "completed_files": task_state.completed_files,
        "failed_files": task_state.failed_files,
        "start_time": task_state.start_time,
        "created_at": task_state.created_at,
        "estimated_time_remaining": task_state.estimated_time_remaining,
        "current_file": task_state.current_file,
        "files": [file.model_dump() for file in task_state.files]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
