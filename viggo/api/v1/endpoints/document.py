import io
import os
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from viggo.core.config import settings
from viggo.models.schemas import DocumentInfoResponse, DocumentUploadResponse
from viggo.core.services.implementations.rag_service_impl import RAGService
from viggo.core.services.implementations.graph_service_impl import GraphService
from viggo.dependencies import get_rag_service, get_graph_service
from viggo.core.utils.file_ops import clear_indexes_and_graph

router = APIRouter()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
    graph_service: GraphService = Depends(get_graph_service)
):
    """Upload a document (PDF, EPUB, etc.), process it for RAG, and load entities/relationships into the graph."""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file size (limit to 50MB)
    file_content = await file.read()
    if len(file_content) > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB.")
    
    # Check if file format is supported
    file_extension = Path(file.filename).suffix.lower()
    if not rag_service.is_format_supported(file.filename):
        supported_formats = rag_service.get_supported_formats()
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {file_extension}. Supported formats: {', '.join(supported_formats)}"
        )
    
    try:
        # Clear existing indexes and graph
        clear_indexes_and_graph(rag_service, graph_service)
        
        # Save uploaded file
        file_location = os.path.join(settings.data_dir, file.filename)
        with open(file_location, "wb") as file_object:
            file_object.write(file_content)
        
        # Process document using the new generic method
        num_chunks, _, all_chunks_with_metadata = rag_service.process_document(file_location)
        
        # Load entities and relationships into the graph
        graph_service.extract_and_load_graph(file.filename, all_chunks_with_metadata)
        
        # Get file info for response
        file_info = rag_service.document_processor_factory.get_file_info(file_location)
        
        return DocumentUploadResponse(
            filename=file.filename, 
            num_chunks_indexed=num_chunks, 
            message=f"Document ({file_extension.upper()}) processed and indexed for RAG. Graph processing acknowledged."
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get("/supported-formats")
async def get_supported_formats(
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get list of supported document formats."""
    supported_formats = rag_service.get_supported_formats()
    return {
        "supported_formats": supported_formats,
        "description": "List of supported file extensions for document upload"
    }

# @router.get("/info", response_model=DocumentInfoResponse)
# async def get_document_info():
#     """Get information about the currently loaded document (not implemented)."""
#     raise HTTPException(status_code=501, detail="Not implemented yet. Document info is not persisted across sessions.")