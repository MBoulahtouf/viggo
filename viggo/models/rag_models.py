"""
RAG-specific models and schemas for the Viggo system.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class QueryContext(BaseModel):
    """Context for RAG queries."""
    query: str = Field(..., description="The query text")
    page_filter: Optional[int] = Field(None, description="Filter by specific page number")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    include_metadata: bool = Field(True, description="Include metadata in results")
    search_method: str = Field("hybrid", description="Search method: semantic, keyword, hybrid")


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    page_number: Optional[int] = Field(None, ge=1, description="Specific page number to focus on")
    context: Optional[QueryContext] = Field(None, description="Additional query context")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class SourcePage(BaseModel):
    """Source page information."""
    page_number: int = Field(..., description="Page number")
    content: str = Field(..., description="Page content")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="Generated answer")
    source_pages: List[SourcePage] = Field(..., description="Source pages used for the answer")
    search_method: str = Field(..., description="Search method used")
    processing_time: float = Field(..., description="Processing time in seconds")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the answer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentInfo(BaseModel):
    """Document information model."""
    filename: str = Field(..., description="Document filename")
    file_type: str = Field(..., description="File type/extension")
    total_pages: int = Field(..., ge=1, description="Total number of pages")
    content_start_page: int = Field(..., ge=1, description="First page with actual content")
    content_end_page: int = Field(..., ge=1, description="Last page with actual content")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    upload_timestamp: datetime = Field(..., description="When the document was uploaded")
    processing_status: str = Field(..., description="Current processing status")


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    filename: str = Field(..., description="Document filename")
    file_type: str = Field(..., description="File type/extension")
    file_size: int = Field(..., ge=1, description="File size in bytes")
    force_reindex: bool = Field(False, description="Force reindexing even if document exists")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    filename: str = Field(..., description="Document filename")
    num_chunks_indexed: int = Field(..., ge=0, description="Number of chunks created")
    message: str = Field(..., description="Status message")
    document_info: DocumentInfo = Field(..., description="Document information")
    processing_time: float = Field(..., description="Processing time in seconds")


class IndexingStatus(str, Enum):
    """Indexing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IndexingRequest(BaseModel):
    """Request model for document indexing."""
    document_id: str = Field(..., description="Document identifier")
    chunking_strategy: str = Field("hybrid", description="Chunking strategy to use")
    enable_entity_extraction: bool = Field(True, description="Enable entity extraction")
    enable_graph_indexing: bool = Field(True, description="Enable graph indexing")
    force_reindex: bool = Field(False, description="Force reindexing")


class IndexingResponse(BaseModel):
    """Response model for document indexing."""
    document_id: str = Field(..., description="Document identifier")
    status: IndexingStatus = Field(..., description="Current indexing status")
    chunks_created: int = Field(0, ge=0, description="Number of chunks created")
    entities_extracted: int = Field(0, ge=0, description="Number of entities extracted")
    relationships_created: int = Field(0, ge=0, description="Number of relationships created")
    processing_time: float = Field(0.0, description="Processing time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RAGConfig(BaseModel):
    """RAG system configuration."""
    chunking_strategy: str = Field("hybrid", description="Default chunking strategy")
    search_method: str = Field("hybrid", description="Default search method")
    max_chunk_size: int = Field(400, ge=50, le=1000, description="Maximum chunk size in words")
    overlap_ratio: float = Field(0.2, ge=0.0, le=0.5, description="Chunk overlap ratio")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    max_results: int = Field(10, ge=1, le=50, description="Maximum results to return")
    enable_caching: bool = Field(True, description="Enable result caching")
    cache_ttl: int = Field(3600, ge=60, description="Cache TTL in seconds")


class RAGStatus(BaseModel):
    """RAG system status."""
    is_ready: bool = Field(..., description="Whether the system is ready")
    documents_indexed: int = Field(0, ge=0, description="Number of documents indexed")
    total_chunks: int = Field(0, ge=0, description="Total number of chunks")
    total_entities: int = Field(0, ge=0, description="Total number of entities")
    total_relationships: int = Field(0, ge=0, description="Total number of relationships")
    last_indexed: Optional[datetime] = Field(None, description="Last indexing timestamp")
    system_health: str = Field(..., description="Overall system health")


class SystemStatus(BaseModel):
    """Overall system status."""
    rag_status: RAGStatus = Field(..., description="RAG system status")
    vector_storage: Dict[str, Any] = Field(..., description="Vector storage status")
    graph_storage: Dict[str, Any] = Field(..., description="Graph storage status")
    cache_storage: Dict[str, Any] = Field(..., description="Cache storage status")
    retrievers: Dict[str, Any] = Field(..., description="Retriever status")
    generators: Dict[str, Any] = Field(..., description="Generator status")
    uptime: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="System version")
