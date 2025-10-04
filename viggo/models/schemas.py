# viggo/models/schemas.py
# Legacy schemas for backward compatibility
# New models should use the organized model files in this package

from pydantic import BaseModel
from typing import List, Any, Dict

# Import new models for backward compatibility
from .rag_models import (
    QueryRequest as NewQueryRequest,
    QueryResponse as NewQueryResponse,
    DocumentUploadResponse as NewDocumentUploadResponse,
    DocumentInfo as DocumentInfoResponse
)
from .graph_models import (
    NodeModel as NewNodeModel,
    NodeListResponse as NewNodeListResponse,
    CanonicalGroup as GroupedNodeModel,
    GroupedNodeListResponse as NewGroupedNodeListResponse,
    EntityGraphResponse as NewEntityGraphResponse
)

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

# Legacy models that are no longer used but kept for compatibility
class LegacyQueryRequest(BaseModel):
    """Legacy query request model."""
    question: str
    page_number: int

class LegacyQueryResponse(BaseModel):
    """Legacy query response model."""
    question: str
    answer: str
    source_pages: List[int]

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
    labels: List[str]

class LegacyNodeListResponse(BaseModel):
    """Legacy node list response model."""
    nodes: List[LegacyNodeModel]

class LegacyGroupedNodeModel(BaseModel):
    """Legacy grouped node model."""
    canonical: str
    aliases: List[str]
    labels: List[str]

class LegacyGroupedNodeListResponse(BaseModel):
    """Legacy grouped node list response model."""
    grouped_nodes: List[LegacyGroupedNodeModel]

class LegacyEntityGraphResponse(BaseModel):
    """Legacy entity graph response model."""
    entity_name: str
    graph_data: Any
