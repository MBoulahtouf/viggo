# viggo/models/schemas.py
from pydantic import BaseModel
from typing import List, Any, Dict

class QueryRequest(BaseModel):
    question: str
    page_number: int

class QueryResponse(BaseModel):
    question: str
    answer: str
    source_pages: List[int]

class DocumentInfoResponse(BaseModel):
    filename: str
    total_pages: int
    content_start_page: int
    content_end_page: int

class DocumentUploadResponse(BaseModel):
    filename: str
    num_chunks_indexed: int
    message: str

class NodeModel(BaseModel):
    name: str
    labels: List[str]

class NodeListResponse(BaseModel):
    nodes: List[NodeModel]

class GroupedNodeModel(BaseModel):
    canonical: str
    aliases: List[str]
    labels: List[str]

class GroupedNodeListResponse(BaseModel):
    grouped_nodes: List[GroupedNodeModel]

class EntityGraphResponse(BaseModel):
    entity_name: str
    graph_data: Any
