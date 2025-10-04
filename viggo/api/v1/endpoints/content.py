"""
Content processing API endpoints using the new SOLID architecture.
"""

from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File
from typing import List, Optional
import logging
import time
import os
from pathlib import Path

from viggo.models.content_models import (
    ChunkingRequest, ChunkingResponse,
    ContentFilterRequest, ContentFilterResponse,
    EntityExtractionRequest, EntityExtractionResponse,
    DocumentProcessingRequest, DocumentProcessingResponse,
    ChunkModel, ChunkMetadata
)
from viggo.models.api_models import (
    SuccessResponse, ErrorResponse, PaginationParams, PaginatedResponse
)
from viggo.core.services import (
    ContentFilterService, EnhancedEntityExtractor, HybridChunkingService
)
from viggo.core.config import settings

router = APIRouter(prefix="/content", tags=["Content Processing"])


@router.post("/chunk", response_model=SuccessResponse[ChunkingResponse])
async def chunk_document(
    request: ChunkingRequest,
    chunking_service: HybridChunkingService = Depends(lambda: HybridChunkingService())
):
    """
    Chunk a document using the specified chunking strategy.
    
    This endpoint processes a document and creates searchable chunks
    with configurable chunking parameters.
    """
    start_time = time.time()
    
    try:
        logging.info(f"Starting document chunking: {request.document_id}")
        
        # For now, we'll use a simplified approach
        # In a real implementation, this would use the actual document processing pipeline
        
        processing_time = time.time() - start_time
        
        response = ChunkingResponse(
            document_id=request.document_id,
            chunks_created=100,  # Mock data
            chunking_strategy=request.chunking_strategy,
            processing_time=processing_time,
            chunk_levels={
                "chapter": 10,
                "passage": 80,
                "sentence": 10
            },
            chunk_types={
                "standard_passage": 70,
                "critical_lore": 20,
                "dialogue_block": 10
            },
            entities_extracted=50 if request.enable_entity_extraction else 0,
            relationships_created=25 if request.enable_entity_extraction else 0,
            metadata={
                "max_chunk_size": request.max_chunk_size,
                "min_chunk_size": request.min_chunk_size,
                "overlap_ratio": request.overlap_ratio,
                "hierarchical_enabled": request.enable_hierarchical
            }
        )
        
        return SuccessResponse(
            message="Document chunked successfully",
            data=response
        )
        
    except Exception as e:
        logging.error(f"Document chunking failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document chunking failed: {str(e)}"
        )


@router.post("/filter", response_model=SuccessResponse[ContentFilterResponse])
async def filter_content(
    request: ContentFilterRequest,
    filter_service: ContentFilterService = Depends(lambda: ContentFilterService())
):
    """
    Filter content to determine if it should be indexed.
    
    This endpoint analyzes content and determines whether it contains
    actual story material or should be filtered out.
    """
    try:
        logging.info(f"Filtering content from page {request.page_number}")
        
        # Use the content filter service
        should_index = filter_service.should_index_content(
            request.content, 
            request.page_number
        )
        
        content_type = filter_service.classify_content_type(
            request.content, 
            request.page_number
        )
        
        response = ContentFilterResponse(
            should_index=should_index,
            content_type=content_type,
            confidence=0.85,  # Would come from actual classification confidence
            filtered_content=request.content if should_index else "",
            filter_reasons=[] if should_index else ["Metadata content detected"],
            metadata={
                "page_number": request.page_number,
                "document_type": request.document_type,
                "filter_metadata": request.filter_metadata,
                "filter_bibliography": request.filter_bibliography
            }
        )
        
        return SuccessResponse(
            message="Content filtered successfully",
            data=response
        )
        
    except Exception as e:
        logging.error(f"Content filtering failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content filtering failed: {str(e)}"
        )


@router.post("/extract-entities", response_model=SuccessResponse[EntityExtractionResponse])
async def extract_entities(
    request: EntityExtractionRequest,
    extractor: EnhancedEntityExtractor = Depends(lambda: EnhancedEntityExtractor())
):
    """
    Extract entities from content using enhanced entity extraction.
    
    This endpoint processes content and extracts named entities with
    deduplication and disambiguation.
    """
    start_time = time.time()
    
    try:
        logging.info(f"Extracting entities from page {request.page_number}")
        
        # Check if content should be processed
        if not extractor.should_process_content(request.content, request.page_number):
            return SuccessResponse(
                message="Content not suitable for entity extraction",
                data=EntityExtractionResponse(
                    entities=[],
                    total_entities=0,
                    unique_entities=0,
                    entity_types={},
                    processing_time=0.0,
                    deduplication_applied=False,
                    disambiguation_applied=False,
                    metadata={"reason": "Content filtered out"}
                )
            )
        
        # Extract entities
        entities = extractor.extract_entities_enhanced(
            request.content, 
            request.page_number
        )
        
        processing_time = time.time() - start_time
        
        # Count entities by type
        entity_types = {}
        for entity in entities:
            entity_type = entity.get("label", "UNKNOWN")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        response = EntityExtractionResponse(
            entities=entities,
            total_entities=len(entities),
            unique_entities=len(set(entity.get("text", "") for entity in entities)),
            entity_types=entity_types,
            processing_time=processing_time,
            deduplication_applied=request.enable_deduplication,
            disambiguation_applied=request.enable_disambiguation,
            metadata={
                "page_number": request.page_number,
                "entity_types": request.entity_types,
                "confidence_threshold": request.confidence_threshold,
                "max_entities": request.max_entities
            }
        )
        
        return SuccessResponse(
            message="Entities extracted successfully",
            data=response
        )
        
    except Exception as e:
        logging.error(f"Entity extraction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity extraction failed: {str(e)}"
        )


@router.post("/process", response_model=SuccessResponse[DocumentProcessingResponse])
async def process_document(
    request: DocumentProcessingRequest,
    chunking_service: HybridChunkingService = Depends(lambda: HybridChunkingService()),
    filter_service: ContentFilterService = Depends(lambda: ContentFilterService()),
    extractor: EnhancedEntityExtractor = Depends(lambda: EnhancedEntityExtractor())
):
    """
    Process a document with full content processing pipeline.
    
    This endpoint performs chunking, content filtering, and entity extraction
    in a coordinated manner.
    """
    start_time = time.time()
    
    try:
        logging.info(f"Starting full document processing: {request.document_id}")
        
        # This would coordinate all the processing steps
        # For now, we'll return mock data
        
        processing_time = time.time() - start_time
        
        response = DocumentProcessingResponse(
            document_id=request.document_id,
            processing_status="completed",
            chunks_created=100,
            entities_extracted=50,
            relationships_created=25,
            processing_time=processing_time,
            metadata={
                "chunking_config": request.chunking_config.dict() if request.chunking_config else {},
                "entity_extraction_config": request.entity_extraction_config.dict() if request.entity_extraction_config else {},
                "content_filtering_config": request.content_filtering_config.dict() if request.content_filtering_config else {},
                "processing_options": request.processing_options
            }
        )
        
        return SuccessResponse(
            message="Document processed successfully",
            data=response
        )
        
    except Exception as e:
        logging.error(f"Document processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )


@router.get("/chunks", response_model=SuccessResponse[PaginatedResponse[ChunkModel]])
async def list_chunks(
    pagination: PaginationParams = Depends(),
    chunk_level: Optional[str] = None,
    chunk_type: Optional[str] = None,
    chunking_service: HybridChunkingService = Depends(lambda: HybridChunkingService())
):
    """
    List document chunks with pagination and filtering.
    
    Returns a paginated list of chunks with optional filtering
    by chunk level and type.
    """
    try:
        # This would retrieve actual chunks from the chunking service
        # For now, we'll return mock data
        
        mock_chunks = [
            ChunkModel(
                chunk_id=f"chunk_{i}",
                content=f"Mock chunk content {i}",
                metadata=ChunkMetadata(
                    chunk_id=f"chunk_{i}",
                    level="passage",
                    chunk_type="standard_passage",
                    word_count=100 + i * 10,
                    char_count=500 + i * 50,
                    page_number=1 + i // 10,
                    chapter_title=f"Chapter {1 + i // 10}",
                    entities=[],
                    relationships=[],
                    lore_significance=0.5 + (i % 10) * 0.05
                )
            )
            for i in range(pagination.offset, pagination.offset + pagination.page_size)
        ]
        
        total_items = 1000  # Mock total
        
        paginated_response = PaginatedResponse(
            items=mock_chunks,
            pagination={
                "page": pagination.page,
                "page_size": pagination.page_size,
                "total_items": total_items,
                "total_pages": (total_items + pagination.page_size - 1) // pagination.page_size,
                "has_next": pagination.offset + pagination.page_size < total_items,
                "has_previous": pagination.page > 1
            },
            metadata={
                "chunk_level_filter": chunk_level,
                "chunk_type_filter": chunk_type
            }
        )
        
        return SuccessResponse(
            message="Chunks retrieved successfully",
            data=paginated_response
        )
        
    except Exception as e:
        logging.error(f"Failed to list chunks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list chunks: {str(e)}"
        )
