"""
Storage interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generic, TypeVar
from dataclasses import dataclass
from enum import Enum

T = TypeVar('T')


class StorageType(Enum):
    """Types of storage systems."""
    VECTOR = "vector"
    GRAPH = "graph"
    DOCUMENT = "document"
    CACHE = "cache"
    METADATA = "metadata"


@dataclass
class StorageMetadata:
    """Metadata for stored items."""
    id: str
    storage_type: StorageType
    created_at: float
    updated_at: float
    size_bytes: int
    additional_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metadata is None:
            self.additional_metadata = {}


class StorageBackend(ABC, Generic[T]):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def store(self, key: str, data: T, metadata: Optional[StorageMetadata] = None) -> bool:
        """Store data with the given key."""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Optional[T]:
        """Retrieve data by key."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data by key."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if data exists for the given key."""
        pass
    
    @abstractmethod
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all keys, optionally filtered by pattern."""
        pass
    
    @abstractmethod
    def get_storage_type(self) -> StorageType:
        """Get the type of storage backend."""
        pass


class VectorStorage(ABC):
    """Abstract base class for vector storage."""
    
    @abstractmethod
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to the storage."""
        pass
    
    @abstractmethod
    def search_vectors(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def get_vector_count(self) -> int:
        """Get the number of stored vectors."""
        pass
    
    @abstractmethod
    def clear_vectors(self) -> bool:
        """Clear all vectors from storage."""
        pass


class GraphStorage(ABC):
    """Abstract base class for graph storage."""
    
    @abstractmethod
    def add_node(self, node_id: str, labels: List[str], properties: Dict[str, Any]) -> bool:
        """Add a node to the graph."""
        pass
    
    @abstractmethod
    def add_relationship(self, from_node: str, to_node: str, relationship_type: str, properties: Dict[str, Any]) -> bool:
        """Add a relationship between nodes."""
        pass
    
    @abstractmethod
    def query_nodes(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query nodes in the graph."""
        pass
    
    @abstractmethod
    def query_relationships(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query relationships in the graph."""
        pass
    
    @abstractmethod
    def clear_graph(self) -> bool:
        """Clear all nodes and relationships."""
        pass


class CacheStorage(ABC):
    """Abstract base class for cache storage."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self, pattern: Optional[str] = None) -> bool:
        """Clear cache entries, optionally filtered by pattern."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass
