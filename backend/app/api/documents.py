"""
Document management API endpoints
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from app.db.session import get_db
from app.models.document import Document
from app.services.document_processor import DocumentProcessor
from app.core.config import settings
import os
import uuid
from datetime import datetime

router = APIRouter()


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a PDF document for processing
    
    This endpoint:
    1. Saves the uploaded file
    2. Creates a document record
    3. Triggers background processing (Docling extraction)
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Validate file size
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File size exceeds {settings.MAX_FILE_SIZE / 1024 / 1024}MB limit")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{file_id}{file_extension}"
    file_path = os.path.join(settings.UPLOAD_DIR, "documents", unique_filename)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Create document record
    document = Document(
        filename=file.filename,
        file_path=file_path,
        processing_status="pending"
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    
    # Define background task function
    async def process_document_task(file_path: str, doc_id: int):
        """Wrapper for async document processing in background"""
        # Create a new database session for the background task
        from app.db.session import SessionLocal
        task_db = SessionLocal()
        try:
            processor = DocumentProcessor(task_db)
            await processor.process_document(file_path, doc_id)
        except Exception as e:
            print(f"Error in background document processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            task_db.close()
    
    # Use FastAPI's BackgroundTasks (runs after response is sent)
    # This ensures the upload endpoint returns immediately
    background_tasks.add_task(process_document_task, file_path, document.id)
    
    return {
        "id": document.id,
        "filename": document.filename,
        "status": document.processing_status,
        "message": "Document uploaded successfully. Processing will begin shortly."
    }


@router.get("")
async def list_documents(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get list of all documents
    
    Optimized to return documents quickly even if count query is slow.
    """
    import time
    start_time = time.time()
    
    try:
        # First, get documents quickly (this should be fast)
        print(f"list_documents: Starting query at {time.time()}")
        documents = db.query(Document).order_by(Document.upload_date.desc()).offset(skip).limit(limit).all()
        query_time = time.time() - start_time
        print(f"list_documents: Documents fetched in {query_time:.3f}s")
        
        # Build response with documents first
        result_documents = [
            {
                "id": doc.id,
                "filename": doc.filename,
                "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                "status": doc.processing_status,
                "total_pages": doc.total_pages or 0,
                "text_chunks": doc.text_chunks_count or 0,
                "images": doc.images_count or 0,
                "tables": doc.tables_count or 0
            }
            for doc in documents
        ]
        
        # Try to get count, but don't let it block if it's slow
        # Use a timeout or estimate if count is taking too long
        total = None
        try:
            count_start = time.time()
            total = db.query(Document).count()
            count_time = time.time() - count_start
            if count_time > 0.5:  # Log if count takes more than 0.5 seconds
                print(f"WARNING: Count query took {count_time:.2f} seconds")
        except Exception as count_error:
            print(f"WARNING: Count query failed: {count_error}")
            # Estimate total based on returned documents
            # If we got a full page, there might be more
            total = len(documents) if len(documents) < limit else len(documents) + 1
        
        total_time = time.time() - start_time
        if total_time > 1.0:
            print(f"WARNING: list_documents total time: {total_time:.2f} seconds")
        
        result = {
            "documents": result_documents,
            "total": total or len(result_documents)  # Fallback to len if count failed
        }
        
        print(f"list_documents: Returning {len(result_documents)} documents (total: {total}) in {total_time:.3f}s")
        return result
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"ERROR in list_documents after {error_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching documents: {str(e)}")


@router.get("/{document_id}")
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get document details including extracted images and tables
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": document.id,
        "filename": document.filename,
        "upload_date": document.upload_date,
        "status": document.processing_status,
        "error_message": document.error_message,
        "total_pages": document.total_pages,
        "text_chunks": document.text_chunks_count,
        "images": [
            {
                "id": img.id,
                "url": f"/uploads/images/{os.path.basename(img.file_path)}",
                "page": img.page_number,
                "caption": img.caption,
                "width": img.width,
                "height": img.height
            }
            for img in document.images
        ],
        "tables": [
            {
                "id": tbl.id,
                "url": f"/uploads/tables/{os.path.basename(tbl.image_path)}",
                "page": tbl.page_number,
                "caption": tbl.caption,
                "rows": tbl.rows,
                "columns": tbl.columns,
                "data": tbl.data
            }
            for tbl in document.tables
        ]
    }


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a document and all associated data
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete physical files
    if os.path.exists(document.file_path):
        os.remove(document.file_path)
    
    for img in document.images:
        if os.path.exists(img.file_path):
            os.remove(img.file_path)
    
    for tbl in document.tables:
        if os.path.exists(tbl.image_path):
            os.remove(tbl.image_path)
    
    # Delete database record (cascade will handle related records)
    db.delete(document)
    db.commit()
    
    return {"message": "Document deleted successfully"}
