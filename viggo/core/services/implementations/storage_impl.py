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
from .redis_service_impl import RedisService
from .graph_service_impl import GraphService
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from viggo.core.config import settings

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


class ElasticsearchVectorStorage(VectorStorage):
    """Concrete implementation of Elasticsearch vector storage."""
    
    def __init__(self, es_client: Elasticsearch = None, index_name: str = None, dimension: int = 384):
        self.es_client = es_client or self._create_default_client()
        self.index_name = index_name or settings.local_elasticsearch_index
        self.dimension = dimension
        self._ensure_index_exists()
    
    def _create_default_client(self) -> Elasticsearch:
        """Create default Elasticsearch client."""
        try:
            return Elasticsearch(
                hosts=[f"{settings.local_elasticsearch_host}:{settings.local_elasticsearch_port}"],
                request_timeout=30,
                retry_on_timeout=True
            )
        except Exception as e:
            print(f"Error creating Elasticsearch client: {e}")
            return None
    
    def _ensure_index_exists(self):
        """Ensure the Elasticsearch index exists with proper mapping."""
        if not self.es_client:
            return
            
        try:
            # Check if index exists
            if not self.es_client.indices.exists(index=self.index_name):
                self._create_index()
        except Exception as e:
            print(f"Error ensuring index exists: {e}")
    
    def _create_index(self):
        """Create Elasticsearch index with proper mapping."""
        mapping = {
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "content_vector": {
                        "type": "dense_vector",
                        "dims": self.dimension,
                        "index": True,
                        "similarity": "l2_norm"
                    },
                    "page": {
                        "type": "integer"
                    },
                    "chunk_id": {
                        "type": "keyword"
                    },
                    "entities": {
                        "type": "keyword"
                    },
                    "entity_labels": {
                        "type": "keyword"
                    },
                    "chapter_title": {
                        "type": "text"
                    },
                    "chunk_type": {
                        "type": "keyword"
                    },
                    "lore_significance": {
                        "type": "float"
                    },
                    "word_count": {
                        "type": "integer"
                    },
                    "char_count": {
                        "type": "integer"
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index.knn": True,
                "index.knn.algo_param.ef_search": 100
            }
        }
        
        try:
            self.es_client.indices.create(index=self.index_name, body=mapping)
            print(f"Created Elasticsearch index: {self.index_name}")
        except Exception as e:
            print(f"Error creating index: {e}")
    
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to Elasticsearch index."""
        if not self.es_client:
            print("Elasticsearch client not available")
            return False
            
        try:
            if not vectors:
                return True
            
            bulk_actions = []
            for i, (vector, meta) in enumerate(zip(vectors, metadata)):
                doc = {
                    "content_vector": vector,
                    **meta
                }
                bulk_actions.append({
                    "_index": self.index_name,
                    "_source": doc
                })
            
            # Bulk index
            response = bulk(self.es_client, bulk_actions)
            print(f"Indexed {response[0]} vectors to Elasticsearch")
            return response[0] > 0
            
        except Exception as e:
            print(f"Error adding vectors to Elasticsearch: {e}")
            return False
    
    def search_vectors(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors using Elasticsearch."""
        if not self.es_client:
            return []
            
        try:
            query = {
                "knn": {
                    "field": "content_vector",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": top_k * 10
                }
            }
            
            response = self.es_client.search(
                index=self.index_name,
                body={"query": query},
                size=top_k
            )
            
            results = []
            for hit in response['hits']['hits']:
                result = {
                    "content": hit['_source'].get('content', ''),
                    "score": hit['_score'],
                    "metadata": hit['_source'],
                    "index": hit['_id']
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error searching vectors in Elasticsearch: {e}")
            return []
    
    def get_vector_count(self) -> int:
        """Get the number of stored vectors."""
        if not self.es_client:
            return 0
            
        try:
            response = self.es_client.count(index=self.index_name)
            return response['count']
        except Exception as e:
            print(f"Error getting vector count: {e}")
            return 0
    
    def clear_vectors(self) -> bool:
        """Clear all vectors from storage."""
        if not self.es_client:
            return False
            
        try:
            self.es_client.indices.delete(index=self.index_name)
            self._create_index()
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
