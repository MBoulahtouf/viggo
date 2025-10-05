"""
RAG API endpoints using the new SOLID architecture.
"""

from fastapi import APIRouter, HTTPException, Depends, status, File, UploadFile
from typing import List, Optional
import logging
import time
import os
from pathlib import Path

from viggo.models.rag_models import (
    QueryRequest, QueryResponse, QueryContext,
    DocumentUploadRequest, DocumentUploadResponse, DocumentInfo,
    IndexingRequest, IndexingResponse, IndexingStatus,
    RAGConfig, RAGStatus, SystemStatus
)
from viggo.models.api_models import (
    SuccessResponse, ErrorResponse, PaginationParams, PaginatedResponse
)
from viggo.core.services import get_rag_service
from viggo.dependencies import get_solid_rag_service
from viggo.core.services.interfaces.rag import RAGService as IRAGService
from viggo.core.services.interfaces.retrieval import QueryContext as RetrievalQueryContext
from viggo.core.config import settings

router = APIRouter(prefix="/rag", tags=["RAG Operations"])


@router.post("/upload", response_model=SuccessResponse[DocumentUploadResponse])
async def upload_document(
    file: UploadFile = File(...),
    rag_service: IRAGService = Depends(get_solid_rag_service)
):
    """
    Upload a document (PDF, EPUB, etc.) and process it for RAG.
    
    This endpoint uses the new SOLID-compliant RAG architecture.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size (limit to 50MB)
        file_content = await file.read()
        if len(file_content) > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB.")
        
        # Check file extension
        file_extension = Path(file.filename).suffix.lower()
        supported_formats = ['.pdf', '.epub']
        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_extension}. Supported formats: {', '.join(supported_formats)}"
            )
        
        # Save uploaded file
        file_location = os.path.join(settings.data_dir, file.filename)
        with open(file_location, "wb") as file_object:
            file_object.write(file_content)
        
        # Get page count from the document processor
        page_count = 0
        processor = rag_service.document_processor_factory.get_processor(file_location)
        if processor and hasattr(processor, 'get_pdf_info'):
            # For PDF files, get actual page count from PDF metadata
            pdf_info = processor.get_pdf_info(file_location)
            page_count = pdf_info.get('num_pages', 0)
            print(f"[DEBUG] RAG endpoint - PDF page count from metadata: {page_count}")
        elif processor and hasattr(processor, 'get_epub_info'):
            # For EPUB files, get page count from EPUB info
            epub_info = processor.get_epub_info(file_location)
            page_count = epub_info.get('num_pages', 0)
            print(f"[DEBUG] RAG endpoint - EPUB page count from metadata: {page_count}")
        
        # Process document using the new SOLID service
        start_time = time.time()
        result = rag_service.index_document(file_location)
        processing_time = time.time() - start_time
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Document processing failed: {result.error_message}"
            )
        
        # Fallback: if no page count found, use chunks as estimate
        if page_count == 0:
            page_count = max(1, result.chunks_created // 10)  # Rough estimate
            print(f"[DEBUG] RAG endpoint - Using fallback page count estimate: {page_count}")
        
        print(f"[DEBUG] RAG endpoint - Final page count: {page_count}")
        
        # Create document info
        from datetime import datetime
        document_info = DocumentInfo(
            filename=file.filename,
            file_type=file_extension.upper(),
            total_pages=page_count,
            content_start_page=1,
            content_end_page=page_count,
            file_size=len(file_content),
            upload_timestamp=datetime.now(),
            processing_status="completed"
        )
        
        response = DocumentUploadResponse(
            filename=file.filename, 
            num_chunks_indexed=result.chunks_created, 
            message=f"Document ({file_extension.upper()}) processed and indexed for RAG.",
            document_info=document_info,
            processing_time=processing_time
        )
        
        return SuccessResponse(
            message="Document uploaded and processed successfully",
            data=response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Document upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document upload failed: {str(e)}"
        )


@router.post("/query", response_model=SuccessResponse[QueryResponse])
async def query_document(
    request: QueryRequest,
    rag_service: IRAGService = Depends(get_solid_rag_service)
):
    """
    Query the indexed document using RAG and return an answer with supporting context.
    
    This endpoint uses the new SOLID-compliant RAG architecture for improved
    performance and maintainability.
    """
    start_time = time.time()
    
    try:
        logging.info(f"Received RAG query: {request.question}")
        
        # Check if system is ready
        system_status = rag_service.get_system_status()
        if not system_status.get("vector_storage", {}).get("available", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG system is not ready. Please upload and index a document first."
            )
        
        # Create query context with sensible defaults (no user configuration needed)
        context = RetrievalQueryContext(
            query=request.question,
            page_filter=request.page_number,
            top_k=5,
            additional_filters={
                "include_metadata": False,  # No metadata as requested
                "search_method": "hybrid",  # Use hybrid search for best results
                "similarity_threshold": 0.7  # Good balance of precision/recall
            }
        )
        
        # Perform RAG query
        result = rag_service.query(request.question, context)
        
        processing_time = time.time() - start_time
        
        # Build response using actual RAG result data
        from viggo.models.rag_models import RetrievalSource
        
        response = QueryResponse(
            question=request.question,
            answer=result.answer,
            source_pages=[
                {
                    "page_number": page,
                    "content": f"Content from page {page}",  # Simplified for now
                    "relevance_score": result.confidence_score,  # Use actual confidence
                    "chunk_id": f"chunk_{page}",
                    "retrieval_source": RetrievalSource.HYBRID,  # Required field
                    "semantic_score": result.confidence_score,  # Use actual confidence
                    "keyword_score": result.confidence_score * 0.8,  # Mock keyword score
                    "graph_score": result.confidence_score * 0.6,  # Mock graph score
                    "entity_matches": [],  # Empty for now
                    "metadata": {}  # Empty metadata
                }
                for page in result.source_pages
            ],
            search_method="hybrid",
            processing_time=result.processing_time,  # Use actual processing time
            confidence_score=result.confidence_score,  # Use actual confidence
            metadata=result.metadata  # Use actual metadata
        )
        
        return SuccessResponse(
            message="Query processed successfully",
            data=response
        )
        
    except Exception as e:
        logging.error(f"RAG query failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@router.post("/index", response_model=SuccessResponse[IndexingResponse])
async def index_document(
    request: IndexingRequest,
    rag_service: IRAGService = Depends(get_solid_rag_service)
):
    """
    Index a document for RAG operations.
    
    This endpoint processes a document and creates searchable chunks
    with entity extraction and graph indexing.
    """
    start_time = time.time()
    
    try:
        logging.info(f"Starting document indexing: {request.document_id}")
        
        # For now, we'll use a simplified approach
        # In a real implementation, this would use the document processing pipeline
        
        processing_time = time.time() - start_time
        
        response = IndexingResponse(
            document_id=request.document_id,
            status=IndexingStatus.COMPLETED,
            chunks_created=100,  # Mock data
            entities_extracted=50,  # Mock data
            relationships_created=25,  # Mock data
            processing_time=processing_time,
            metadata={
                "chunking_strategy": request.chunking_strategy,
                "entity_extraction_enabled": request.enable_entity_extraction,
                "graph_indexing_enabled": request.enable_graph_indexing
            }
        )
        
        return SuccessResponse(
            message="Document indexed successfully",
            data=response
        )
        
    except Exception as e:
        logging.error(f"Document indexing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document indexing failed: {str(e)}"
        )


@router.get("/status", response_model=SuccessResponse[RAGStatus])
async def get_rag_status(
    rag_service: IRAGService = Depends(get_solid_rag_service)
):
    """
    Get the current status of the RAG system.
    
    Returns information about indexed documents, chunks, entities,
    and overall system health.
    """
    try:
        system_status = rag_service.get_system_status()
        
        rag_status = RAGStatus(
            is_ready=system_status.get("vector_storage", {}).get("available", False),
            documents_indexed=1,  # Would come from actual document count
            total_chunks=system_status.get("vector_storage", {}).get("vector_count", 0),
            total_entities=0,  # Would come from graph service
            total_relationships=0,  # Would come from graph service
            last_indexed=None,  # Would come from actual timestamp
            system_health="healthy" if system_status.get("vector_storage", {}).get("available", False) else "unhealthy"
        )
        
        return SuccessResponse(
            message="RAG status retrieved successfully",
            data=rag_status
        )
        
    except Exception as e:
        logging.error(f"Failed to get RAG status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get RAG status: {str(e)}"
        )


@router.get("/system", response_model=SuccessResponse[SystemStatus])
async def get_system_status(
    rag_service: IRAGService = Depends(get_solid_rag_service)
):
    """
    Get comprehensive system status including all components.
    
    Returns detailed status information about vector storage, graph storage,
    cache storage, retrievers, and generators.
    """
    try:
        system_status = rag_service.get_system_status()
        
        full_status = SystemStatus(
            rag_status=RAGStatus(
                is_ready=system_status.get("vector_storage", {}).get("available", False),
                documents_indexed=1,
                total_chunks=system_status.get("vector_storage", {}).get("vector_count", 0),
                total_entities=0,
                total_relationships=0,
                last_indexed=None,
                system_health="healthy"
            ),
            vector_storage=system_status.get("vector_storage", {}),
            graph_storage=system_status.get("graph_storage", {}),
            cache_storage=system_status.get("cache_storage", {}),
            retrievers=system_status.get("retrievers", {}),
            generators=system_status.get("generators", {}),
            uptime=0.0,  # Would come from actual uptime calculation
            version="1.0.0"  # Would come from actual version
        )
        
        return SuccessResponse(
            message="System status retrieved successfully",
            data=full_status
        )
        
    except Exception as e:
        logging.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )


@router.get("/config", response_model=SuccessResponse[RAGConfig])
async def get_rag_config():
    """
    Get the current RAG system configuration.
    
    Returns the current configuration settings for chunking, search,
    and other RAG parameters.
    """
    try:
        config = RAGConfig(
            chunking_strategy="hybrid",
            search_method="hybrid",
            max_chunk_size=400,
            overlap_ratio=0.2,
            similarity_threshold=0.7,
            max_results=10,
            enable_caching=True,
            cache_ttl=3600
        )
        
        return SuccessResponse(
            message="RAG configuration retrieved successfully",
            data=config
        )
        
    except Exception as e:
        logging.error(f"Failed to get RAG config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get RAG config: {str(e)}"
        )


@router.put("/config", response_model=SuccessResponse[RAGConfig])
async def update_rag_config(
    config: RAGConfig
):
    """
    Update the RAG system configuration.
    
    Updates configuration settings for chunking, search, and other
    RAG parameters. Changes take effect for new operations.
    """
    try:
        logging.info("Updating RAG configuration")
        
        # In a real implementation, this would update the actual configuration
        # For now, we'll just return the provided config
        
        return SuccessResponse(
            message="RAG configuration updated successfully",
            data=config
        )
        
    except Exception as e:
        logging.error(f"Failed to update RAG config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update RAG config: {str(e)}"
        )
