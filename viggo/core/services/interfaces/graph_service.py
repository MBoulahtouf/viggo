"""
Graph service interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PaginationParams:
    """Parameters for pagination."""
    limit: int = 100
    offset: int = 0
    
    def __post_init__(self):
        if self.limit <= 0:
            raise ValueError("Limit must be positive")
        if self.offset < 0:
            raise ValueError("Offset must be non-negative")


@dataclass
class NodeResult:
    """Result for a single node."""
    name: str
    labels: List[str]
    properties: Dict[str, Any]


@dataclass
class RelationshipResult:
    """Result for a relationship."""
    type: str
    properties: Dict[str, Any]
    target_node: NodeResult


@dataclass
class EntityGraphResult:
    """Complete entity graph data."""
    name: str
    labels: List[str]
    properties: Dict[str, Any]
    relationships: List[RelationshipResult]


class GraphServiceError(Exception):
    """Base exception for GraphService errors."""
    pass


class IGraphService(ABC):
    """Interface for graph database operations."""
    
    @abstractmethod
    def close(self) -> None:
        """Close the graph database connection."""
        pass
    
    @abstractmethod
    def clear_database(self) -> None:
        """Clear all nodes and relationships from the database."""
        pass
    
    @abstractmethod
    def create_document_node(self, filename: str, path: str) -> str:
        """Create a document node in the graph."""
        pass
    
    @abstractmethod
    def create_page_node(self, document_filename: str, page_number: int) -> int:
        """Create a page node in the graph."""
        pass
    
    @abstractmethod
    def create_chunk_node(self, page_number: int, chunk_id: str, content: str) -> str:
        """Create a chunk node in the graph."""
        pass
    
    @abstractmethod
    def create_entity_node(self, name: str, label: str, description: str = "") -> str:
        """Create an entity node in the graph."""
        pass
    
    @abstractmethod
    def link_chunk_to_entity(self, chunk_id: str, entity_name: str, entity_label: str) -> None:
        """Link a chunk to an entity."""
        pass
    
    @abstractmethod
    def create_relationship(self, source_entity: str, source_label: str, target_entity: str, target_label: str, relationship_type: str) -> None:
        """Create a relationship between entities."""
        pass
    
    @abstractmethod
    def extract_and_load_graph(self, filename: str, processed_chunks_with_metadata: List[Dict]) -> None:
        """Extract and load graph data from processed chunks."""
        pass
    
    @abstractmethod
    def get_related_info_for_entity(self, entity_name: str, entity_label: str = "", excluded_rel_types: List[str] = None, excluded_node_labels: List[str] = None) -> List[Dict[str, Any]]:
        """Get related information for an entity."""
        pass
    
    @abstractmethod
    def get_entity_graph_data(self, entity_name: str, entity_label: str = "") -> Dict[str, Any]:
        """Get complete graph data for an entity."""
        pass
    
    @abstractmethod
    def list_all_nodes(self, label: Optional[str] = None, pagination: Optional[PaginationParams] = None) -> List[NodeResult]:
        """List all nodes with optional filtering and pagination."""
        pass
    
    @abstractmethod
    def grouped_nodes(self, label: Optional[str] = None) -> List[Dict[str, Any]]:
        """Group nodes by canonical name showing all aliases and labels."""
        pass
    
    @abstractmethod
    def add_alias_mapping(self, alias: str, canonical: str, confidence: float = 1.0, source: str = "manual") -> None:
        """Add an alias mapping to the aliasing service."""
        pass
    
    @abstractmethod
    def resolve_entity_name(self, entity_name: str) -> str:
        """Resolve an entity name to its canonical form."""
        pass
    
    @abstractmethod
    def get_entity_with_aliases(self, entity_name: str, entity_label: Optional[str] = None) -> Dict[str, Any]:
        """Get entity data including all its aliases."""
        pass
    
    @abstractmethod
    def suggest_aliases_for_entity(self, entity_name: str) -> List[str]:
        """Suggest potential aliases for an entity."""
        pass
    
    @abstractmethod
    def get_entity_details(self, entity_name: str, entity_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific entity."""
        pass
    
    @abstractmethod
    def find_relationships(self, query: str) -> List[Dict[str, Any]]:
        """Find relationships based on a query string."""
        pass
