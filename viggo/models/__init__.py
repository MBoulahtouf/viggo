"""
Viggo Models Package

This package contains all Pydantic models and schemas for the Viggo RAG system.
Organized by domain for better maintainability.
"""

# Core RAG models
from .rag_models import (
    QueryRequest, QueryResponse, QueryContext,
    DocumentUploadRequest, DocumentUploadResponse, DocumentInfo,
    IndexingRequest, IndexingResponse, IndexingStatus,
    RAGConfig, RAGStatus, SystemStatus
)

# Graph and entity models
from .graph_models import (
    NodeModel, RelationshipModel, EntityModel,
    NodeListResponse, RelationshipListResponse, EntityListResponse,
    GraphQueryRequest, GraphQueryResponse,
    AliasMapping, CanonicalGroup, AliasSuggestion
)

# Content processing models
from .content_models import (
    ChunkModel, ChunkMetadata, ChunkingRequest, ChunkingResponse,
    ContentFilterRequest, ContentFilterResponse,
    EntityExtractionRequest, EntityExtractionResponse
)

# API response models
from .api_models import (
    BaseResponse, ErrorResponse, SuccessResponse,
    PaginationParams, PaginatedResponse,
    HealthCheckResponse, VersionResponse
)

# User progress models
from .user_progress import (
    ReadingStatus, UserProgress, DocumentMetadata
)

__all__ = [
    # Core RAG models
    'QueryRequest', 'QueryResponse', 'QueryContext',
    'DocumentUploadRequest', 'DocumentUploadResponse', 'DocumentInfo',
    'IndexingRequest', 'IndexingResponse', 'IndexingStatus',
    'RAGConfig', 'RAGStatus', 'SystemStatus',
    
    # Graph and entity models
    'NodeModel', 'RelationshipModel', 'EntityModel',
    'NodeListResponse', 'RelationshipListResponse', 'EntityListResponse',
    'GraphQueryRequest', 'GraphQueryResponse',
    'AliasMapping', 'CanonicalGroup', 'AliasSuggestion',
    
    # Content processing models
    'ChunkModel', 'ChunkMetadata', 'ChunkingRequest', 'ChunkingResponse',
    'ContentFilterRequest', 'ContentFilterResponse',
    'EntityExtractionRequest', 'EntityExtractionResponse',
    
    # API response models
    'BaseResponse', 'ErrorResponse', 'SuccessResponse',
    'PaginationParams', 'PaginatedResponse',
    'HealthCheckResponse', 'VersionResponse',
    
    # User progress models
    'ReadingStatus', 'UserProgress', 'DocumentMetadata'
]
