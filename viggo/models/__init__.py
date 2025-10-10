"""
Viggo Models Package

This package contains all Pydantic models and schemas for the Viggo RAG system.
Organized by domain for better maintainability.
"""

# Core RAG models
# API response models
from .api_models import (
    BaseResponse,
    ErrorResponse,
    HealthCheckResponse,
    PaginatedResponse,
    PaginationParams,
    SuccessResponse,
    VersionResponse,
)

# Content processing models
from .content_models import (
    ChunkingRequest,
    ChunkingResponse,
    ChunkMetadata,
    ChunkModel,
    ContentFilterRequest,
    ContentFilterResponse,
    EntityExtractionRequest,
    EntityExtractionResponse,
)

# Graph and entity models
from .graph_models import (
    AliasMapping,
    AliasSuggestion,
    CanonicalGroup,
    EntityListResponse,
    EntityModel,
    GraphQueryRequest,
    GraphQueryResponse,
    NodeListResponse,
    NodeModel,
    RelationshipListResponse,
    RelationshipModel,
)
from .rag_models import (
    DocumentInfo,
    DocumentUploadRequest,
    DocumentUploadResponse,
    IndexingRequest,
    IndexingResponse,
    IndexingStatus,
    QueryContext,
    QueryRequest,
    QueryResponse,
    RAGConfig,
    RAGStatus,
    SystemStatus,
)

# User progress models
from .user_progress import DocumentMetadata, ReadingStatus, UserProgress

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
