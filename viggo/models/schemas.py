# viggo/models/schemas.py
# Legacy schemas for backward compatibility
# New models should use the organized model files in this package

from typing import Any

from pydantic import BaseModel

from .graph_models import CanonicalGroup as GroupedNodeModel
from .graph_models import EntityGraphResponse as NewEntityGraphResponse
from .graph_models import GroupedNodeListResponse as NewGroupedNodeListResponse
from .graph_models import NodeListResponse as NewNodeListResponse
from .graph_models import NodeModel as NewNodeModel
from .rag_models import DocumentInfo as DocumentInfoResponse
from .rag_models import DocumentUploadResponse as NewDocumentUploadResponse
from .rag_models import (
    EntityInfo,
    HybridSearchConfig,
    HybridSearchMetrics,
    RelationshipInfo,
    RetrievalSource,
)

# Import new models for backward compatibility
from .rag_models import QueryRequest as NewQueryRequest
from .rag_models import QueryResponse as NewQueryResponse

# Legacy aliases for backward compatibility
QueryRequest = NewQueryRequest
QueryResponse = NewQueryResponse
DocumentUploadResponse = NewDocumentUploadResponse
DocumentInfoResponse = DocumentInfoResponse
NodeModel = NewNodeModel
NodeListResponse = NewNodeListResponse
GroupedNodeModel = GroupedNodeModel
GroupedNodeListResponse = NewGroupedNodeListResponse
EntityGraphResponse = NewEntityGraphResponse

# Export new hybrid RAG models
__all__ = [
    # Legacy models
    'QueryRequest', 'QueryResponse', 'DocumentUploadResponse', 'DocumentInfoResponse',
    'NodeModel', 'NodeListResponse', 'GroupedNodeModel', 'GroupedNodeListResponse', 'EntityGraphResponse',

    # New hybrid RAG models
    'RetrievalSource', 'HybridSearchConfig', 'HybridSearchMetrics', 'EntityInfo', 'RelationshipInfo',

    # Legacy compatibility models
    'LegacyQueryRequest', 'LegacyQueryResponse', 'LegacyDocumentInfoResponse',
    'LegacyDocumentUploadResponse', 'LegacyNodeModel', 'LegacyNodeListResponse',
    'LegacyGroupedNodeModel', 'LegacyGroupedNodeListResponse', 'LegacyEntityGraphResponse'
]

# Legacy models that are no longer used but kept for compatibility
class LegacyQueryRequest(BaseModel):
    """Legacy query request model."""
    question: str
    page_number: int

class LegacyQueryResponse(BaseModel):
    """Legacy query response model."""
    question: str
    answer: str
    source_pages: list[int]

class LegacyDocumentInfoResponse(BaseModel):
    """Legacy document info response model."""
    filename: str
    total_pages: int
    content_start_page: int
    content_end_page: int

class LegacyDocumentUploadResponse(BaseModel):
    """Legacy document upload response model."""
    filename: str
    num_chunks_indexed: int
    message: str

class LegacyNodeModel(BaseModel):
    """Legacy node model."""
    name: str
    labels: list[str]

class LegacyNodeListResponse(BaseModel):
    """Legacy node list response model."""
    nodes: list[LegacyNodeModel]

class LegacyGroupedNodeModel(BaseModel):
    """Legacy grouped node model."""
    canonical: str
    aliases: list[str]
    labels: list[str]

class LegacyGroupedNodeListResponse(BaseModel):
    """Legacy grouped node list response model."""
    grouped_nodes: list[LegacyGroupedNodeModel]

class LegacyEntityGraphResponse(BaseModel):
    """Legacy entity graph response model."""
    entity_name: str
    graph_data: Any
