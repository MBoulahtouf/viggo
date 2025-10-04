"""
Graph and entity models for the Viggo system.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class NodeModel(BaseModel):
    """Graph node model."""
    id: str = Field(..., description="Node identifier")
    name: str = Field(..., description="Node name")
    labels: List[str] = Field(..., description="Node labels")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class RelationshipModel(BaseModel):
    """Graph relationship model."""
    id: str = Field(..., description="Relationship identifier")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relationship_type: str = Field(..., description="Type of relationship")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class EntityModel(BaseModel):
    """Entity model with relationships."""
    node: NodeModel = Field(..., description="Entity node")
    relationships: List[RelationshipModel] = Field(default_factory=list, description="Entity relationships")
    aliases: List[str] = Field(default_factory=list, description="Entity aliases")
    canonical_name: str = Field(..., description="Canonical entity name")


class NodeListResponse(BaseModel):
    """Response model for node listing."""
    nodes: List[NodeModel] = Field(..., description="List of nodes")
    total_count: int = Field(..., description="Total number of nodes")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")


class RelationshipListResponse(BaseModel):
    """Response model for relationship listing."""
    relationships: List[RelationshipModel] = Field(..., description="List of relationships")
    total_count: int = Field(..., description="Total number of relationships")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")


class EntityListResponse(BaseModel):
    """Response model for entity listing."""
    entities: List[EntityModel] = Field(..., description="List of entities")
    total_count: int = Field(..., description="Total number of entities")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")


class GraphQueryRequest(BaseModel):
    """Request model for graph queries."""
    query: str = Field(..., description="Cypher query or natural language query")
    query_type: str = Field("cypher", description="Query type: cypher, natural, or template")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    limit: int = Field(100, ge=1, le=1000, description="Maximum results to return")
    timeout: int = Field(30, ge=1, le=300, description="Query timeout in seconds")


class GraphQueryResponse(BaseModel):
    """Response model for graph queries."""
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    execution_time: float = Field(..., description="Query execution time in seconds")
    nodes_returned: int = Field(0, description="Number of nodes returned")
    relationships_returned: int = Field(0, description="Number of relationships returned")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Query metadata")


class AliasMapping(BaseModel):
    """Alias mapping model."""
    alias: str = Field(..., description="Alias name")
    canonical: str = Field(..., description="Canonical name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source: str = Field(..., description="Source of the mapping")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class CanonicalGroup(BaseModel):
    """Canonical group model."""
    canonical: str = Field(..., description="Canonical name")
    aliases: List[str] = Field(..., description="List of aliases")
    labels: List[str] = Field(..., description="Entity labels")
    entity_count: int = Field(..., description="Number of entities in group")
    confidence_scores: Dict[str, float] = Field(default_factory=dict, description="Confidence scores for aliases")


class GroupedNodeListResponse(BaseModel):
    """Response model for grouped nodes."""
    grouped_nodes: List[CanonicalGroup] = Field(..., description="List of grouped nodes")
    total_groups: int = Field(..., description="Total number of groups")
    total_entities: int = Field(..., description="Total number of entities")


class AliasSuggestion(BaseModel):
    """Alias suggestion model."""
    suggested_alias: str = Field(..., description="Suggested alias")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field(..., description="Reason for suggestion")
    source: str = Field(..., description="Source of suggestion")


class EntityGraphResponse(BaseModel):
    """Response model for entity graph data."""
    entity_name: str = Field(..., description="Entity name")
    graph_data: Dict[str, Any] = Field(..., description="Graph data structure")
    nodes: List[NodeModel] = Field(default_factory=list, description="Related nodes")
    relationships: List[RelationshipModel] = Field(default_factory=list, description="Related relationships")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EntitySearchRequest(BaseModel):
    """Request model for entity search."""
    query: str = Field(..., description="Search query")
    entity_types: List[str] = Field(default_factory=list, description="Filter by entity types")
    limit: int = Field(20, ge=1, le=100, description="Maximum results to return")
    include_aliases: bool = Field(True, description="Include aliases in search")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")


class EntitySearchResponse(BaseModel):
    """Response model for entity search."""
    entities: List[EntityModel] = Field(..., description="Found entities")
    total_count: int = Field(..., description="Total number of matches")
    search_time: float = Field(..., description="Search time in seconds")
    query: str = Field(..., description="Original search query")
