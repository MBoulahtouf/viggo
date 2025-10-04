"""
Concrete implementations of storage services following SOLID principles.
"""

import os
import pickle
import time
from typing import List, Dict, Any, Optional, Generic, TypeVar

from viggo.core.services.interfaces.storage import (
    StorageBackend, VectorStorage, GraphStorage, CacheStorage, 
    StorageMetadata, StorageType
)
from viggo.core.services.redis_service import RedisService
from viggo.core.services.graph_service import GraphService
from faiss import IndexFlatL2, write_index, read_index

T = TypeVar('T')


class FileStorageBackend(StorageBackend[T]):
    """Concrete implementation of file-based storage backend."""
    
    def __init__(self, base_path: str = "storage"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def store(self, key: str, data: T, metadata: Optional[StorageMetadata] = None) -> bool:
        """Store data with the given key."""
        try:
            file_path = os.path.join(self.base_path, f"{key}.pkl")
            
            # Serialize data
            serialized_data = pickle.dumps(data)
            
            # Store data
            with open(file_path, 'wb') as f:
                f.write(serialized_data)
            
            # Store metadata if provided
            if metadata:
                metadata_path = os.path.join(self.base_path, f"{key}_metadata.pkl")
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
            
            return True
            
        except Exception as e:
            print(f"Error storing data for key {key}: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[T]:
        """Retrieve data by key."""
        try:
            file_path = os.path.join(self.base_path, f"{key}.pkl")
            
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            return data
            
        except Exception as e:
            print(f"Error retrieving data for key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete data by key."""
        try:
            file_path = os.path.join(self.base_path, f"{key}.pkl")
            metadata_path = os.path.join(self.base_path, f"{key}_metadata.pkl")
            
            success = True
            
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                success = False
            
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            return success
            
        except Exception as e:
            print(f"Error deleting data for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if data exists for the given key."""
        file_path = os.path.join(self.base_path, f"{key}.pkl")
        return os.path.exists(file_path)
    
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all keys, optionally filtered by pattern."""
        try:
            keys = []
            for filename in os.listdir(self.base_path):
                if filename.endswith('.pkl') and not filename.endswith('_metadata.pkl'):
                    key = filename[:-4]  # Remove .pkl extension
                    if pattern is None or pattern in key:
                        keys.append(key)
            return keys
            
        except Exception as e:
            print(f"Error listing keys: {e}")
            return []
    
    def get_storage_type(self) -> StorageType:
        """Get the type of storage backend."""
        return StorageType.DOCUMENT


class FAISSVectorStorage(VectorStorage):
    """Concrete implementation of FAISS vector storage."""
    
    def __init__(self, index_path: str = "vector_index.bin", dimension: int = 384):
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self.metadata_store = []
        self._load_index()
    
    def _load_index(self):
        """Load existing index if available."""
        try:
            if os.path.exists(self.index_path):
                self.index = read_index(self.index_path)
                print(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                self.index = IndexFlatL2(self.dimension)
                print(f"Created new FAISS index with dimension {self.dimension}")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            self.index = IndexFlatL2(self.dimension)
    
    def _save_index(self):
        """Save index to disk."""
        try:
            write_index(self.index, self.index_path)
            print(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
    
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to the storage."""
        try:
            if not vectors:
                return True
            
            # Ensure vectors are numpy arrays
            import numpy as np
            vectors_array = np.array(vectors, dtype=np.float32)
            
            # Add to index
            self.index.add(vectors_array)
            
            # Store metadata
            self.metadata_store.extend(metadata)
            
            # Save index
            self._save_index()
            
            return True
            
        except Exception as e:
            print(f"Error adding vectors: {e}")
            return False
    
    def search_vectors(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Ensure query vector is numpy array
            import numpy as np
            query_array = np.array([query_vector], dtype=np.float32)
            
            # Search
            distances, indices = self.index.search(query_array, top_k)
            
            # Build results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata_store):
                    result = {
                        "content": self.metadata_store[idx].get("content", ""),
                        "score": 1.0 - distance,  # Convert distance to similarity
                        "metadata": self.metadata_store[idx],
                        "index": int(idx)
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []
    
    def get_vector_count(self) -> int:
        """Get the number of stored vectors."""
        return self.index.ntotal if self.index else 0
    
    def clear_vectors(self) -> bool:
        """Clear all vectors from storage."""
        try:
            self.index = IndexFlatL2(self.dimension)
            self.metadata_store.clear()
            self._save_index()
            return True
        except Exception as e:
            print(f"Error clearing vectors: {e}")
            return False


class Neo4jGraphStorage(GraphStorage):
    """Concrete implementation of Neo4j graph storage."""
    
    def __init__(self, graph_service: GraphService):
        self.graph_service = graph_service
    
    def add_node(self, node_id: str, labels: List[str], properties: Dict[str, Any]) -> bool:
        """Add a node to the graph."""
        try:
            # Use existing graph service implementation
            if "Character" in labels:
                self.graph_service.create_entity_node(node_id, "PERSON", properties.get("description", ""))
            elif "Location" in labels:
                self.graph_service.create_entity_node(node_id, "LOC", properties.get("description", ""))
            elif "Organization" in labels:
                self.graph_service.create_entity_node(node_id, "ORG", properties.get("description", ""))
            else:
                # Generic node creation
                self.graph_service.create_entity_node(node_id, labels[0] if labels else "Entity", properties.get("description", ""))
            
            return True
            
        except Exception as e:
            print(f"Error adding node {node_id}: {e}")
            return False
    
    def add_relationship(self, from_node: str, to_node: str, relationship_type: str, properties: Dict[str, Any]) -> bool:
        """Add a relationship between nodes."""
        try:
            # Use existing graph service implementation
            self.graph_service.create_relationship(
                from_node, "Entity",  # Would need to determine actual labels
                to_node, "Entity",    # Would need to determine actual labels
                relationship_type
            )
            return True
            
        except Exception as e:
            print(f"Error adding relationship {from_node} -> {to_node}: {e}")
            return False
    
    def query_nodes(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query nodes in the graph."""
        try:
            # Use existing graph service implementation
            results = self.graph_service.get_related_info_for_entity(query)
            return results
            
        except Exception as e:
            print(f"Error querying nodes: {e}")
            return []
    
    def query_relationships(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query relationships in the graph."""
        try:
            # Use existing graph service implementation
            results = self.graph_service.find_relationships(query)
            return results
            
        except Exception as e:
            print(f"Error querying relationships: {e}")
            return []
    
    def clear_graph(self) -> bool:
        """Clear all nodes and relationships."""
        try:
            self.graph_service.clear_database()
            return True
        except Exception as e:
            print(f"Error clearing graph: {e}")
            return False


class RedisCacheStorage(CacheStorage):
    """Concrete implementation of Redis cache storage."""
    
    def __init__(self, redis_service: RedisService):
        self.redis_service = redis_service
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            # Use existing Redis service implementation
            return self.redis_service.get_cached_query_result(key, 5, None)
        except Exception as e:
            print(f"Error getting from cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        try:
            # Use existing Redis service implementation
            return self.redis_service.cache_query_result(key, 5, None, value, ttl)
        except Exception as e:
            print(f"Error setting cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            # Use existing Redis service implementation
            return self.redis_service.clear_cache(f"*{key}*")
        except Exception as e:
            print(f"Error deleting from cache: {e}")
            return False
    
    def clear(self, pattern: Optional[str] = None) -> bool:
        """Clear cache entries, optionally filtered by pattern."""
        try:
            return self.redis_service.clear_cache(pattern)
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            return self.redis_service.get_cache_stats()
        except Exception as e:
            print(f"Error getting cache stats: {e}")
            return {"error": str(e)}
