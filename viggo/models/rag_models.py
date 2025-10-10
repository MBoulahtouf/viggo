"""
RAG-specific models and schemas for the Viggo system.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class HybridSearchConfig(BaseModel):
    """Configuration for hybrid search."""
    semantic_weight: float = Field(0.4, ge=0.0, le=1.0, description="Weight for semantic search")
    keyword_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for keyword search")
    graph_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for graph search")
    enable_entity_extraction: bool = Field(True, description="Enable entity extraction from query")
    enable_relationship_traversal: bool = Field(True, description="Enable graph relationship traversal")
    max_relationship_depth: int = Field(2, ge=1, le=5, description="Maximum relationship traversal depth")


class QueryContext(BaseModel):
    """Enhanced context for hybrid RAG queries."""
    query: str = Field(..., description="The query text")
    page_filter: int | None = Field(None, description="Filter by specific page number")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    include_metadata: bool = Field(True, description="Include metadata in results")
    search_method: str = Field("hybrid", description="Search method: semantic, keyword, graph, hybrid")
    hybrid_config: HybridSearchConfig | None = Field(None, description="Hybrid search configuration")
    extracted_entities: list[str] = Field(default_factory=list, description="Entities extracted from query")
    user_context: dict[str, Any] = Field(default_factory=dict, description="User-specific context")
    spoiler_protection: bool = Field(False, description="Enable spoiler protection")
    max_page: int | None = Field(None, description="Maximum page for spoiler protection")


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    page_number: int | None = Field(None, ge=1, description="Specific page number to focus on")
    context: QueryContext | None = Field(None, description="Additional query context")

    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class RetrievalSource(str, Enum):
    """Retrieval source types for hybrid RAG."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    GRAPH = "graph"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


class SourcePage(BaseModel):
    """Source page information with hybrid retrieval tracking."""
    page_number: int = Field(..., description="Page number")
    content: str = Field(..., description="Page content")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    chunk_id: str | None = Field(None, description="Chunk identifier")
    retrieval_source: RetrievalSource = Field(..., description="Which retrieval method found this source")
    semantic_score: float | None = Field(None, ge=0.0, le=1.0, description="Semantic similarity score")
    keyword_score: float | None = Field(None, ge=0.0, le=1.0, description="Keyword matching score")
    graph_score: float | None = Field(None, ge=0.0, le=1.0, description="Graph relationship score")
    entity_matches: list[str] = Field(default_factory=list, description="Entities found in this source")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional source metadata")


class HybridSearchMetrics(BaseModel):
    """Metrics for hybrid search performance."""
    semantic_results: int = Field(0, ge=0, description="Number of semantic search results")
    keyword_results: int = Field(0, ge=0, description="Number of keyword search results")
    graph_results: int = Field(0, ge=0, description="Number of graph search results")
    total_candidates: int = Field(0, ge=0, description="Total candidate results before ranking")
    final_results: int = Field(0, ge=0, description="Final results after ranking")
    semantic_time: float = Field(0.0, description="Semantic search time in seconds")
    keyword_time: float = Field(0.0, description="Keyword search time in seconds")
    graph_time: float = Field(0.0, description="Graph search time in seconds")
    ranking_time: float = Field(0.0, description="Result ranking time in seconds")


class QueryResponse(BaseModel):
    """Enhanced response model for hybrid RAG queries."""
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="Generated answer")
    source_pages: list[SourcePage] = Field(..., description="Source pages used for the answer")
    search_method: str = Field(..., description="Search method used")
    processing_time: float = Field(..., description="Total processing time in seconds")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the answer")
    hybrid_metrics: HybridSearchMetrics | None = Field(None, description="Hybrid search performance metrics")
    extracted_entities: list[str] = Field(default_factory=list, description="Entities extracted from query")
    related_entities: list[str] = Field(default_factory=list, description="Related entities found")
    spoiler_protection_applied: bool = Field(False, description="Whether spoiler protection was applied")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


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
    error_message: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


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
    last_indexed: datetime | None = Field(None, description="Last indexing timestamp")
    system_health: str = Field(..., description="Overall system health")


class EntityInfo(BaseModel):
    """Entity information for hybrid RAG."""
    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Type of entity (person, place, concept, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Entity extraction confidence")
    aliases: list[str] = Field(default_factory=list, description="Alternative names for the entity")
    canonical_name: str | None = Field(None, description="Canonical name for the entity")
    page_references: list[int] = Field(default_factory=list, description="Pages where entity appears")
    relationships: list[str] = Field(default_factory=list, description="Related entities")


class RelationshipInfo(BaseModel):
    """Relationship information for hybrid RAG."""
    source_entity: str = Field(..., description="Source entity name")
    target_entity: str = Field(..., description="Target entity name")
    relationship_type: str = Field(..., description="Type of relationship")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Relationship confidence")
    page_references: list[int] = Field(default_factory=list, description="Pages where relationship appears")
    context: str = Field(..., description="Context of the relationship")


class HybridRAGStatus(BaseModel):
    """Enhanced status for hybrid RAG system."""
    is_ready: bool = Field(..., description="Whether the system is ready")
    documents_indexed: int = Field(0, ge=0, description="Number of documents indexed")
    total_chunks: int = Field(0, ge=0, description="Total number of chunks")
    total_entities: int = Field(0, ge=0, description="Total number of entities")
    total_relationships: int = Field(0, ge=0, description="Total number of relationships")
    last_indexed: datetime | None = Field(None, description="Last indexing timestamp")
    system_health: str = Field(..., description="Overall system health")
    hybrid_components: dict[str, dict[str, Any]] = Field(..., description="Status of hybrid components")
    retrieval_performance: dict[str, float] = Field(default_factory=dict, description="Retrieval performance metrics")


class SystemStatus(BaseModel):
    """Overall system status."""
    rag_status: HybridRAGStatus = Field(..., description="Hybrid RAG system status")
    vector_storage: dict[str, Any] = Field(..., description="Vector storage status")
    graph_storage: dict[str, Any] = Field(..., description="Graph storage status")
    cache_storage: dict[str, Any] = Field(..., description="Cache storage status")
    retrievers: dict[str, Any] = Field(..., description="Retriever status")
    generators: dict[str, Any] = Field(..., description="Generator status")
    uptime: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="System version")
