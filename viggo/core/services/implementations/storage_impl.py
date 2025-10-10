"""
Concrete implementations of storage services following SOLID principles.
"""

import os
import pickle
from typing import Any, TypeVar

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery

from viggo.core.config import settings
from viggo.core.services.interfaces.storage import (
    CacheStorage,
    GraphStorage,
    StorageBackend,
    StorageMetadata,
    StorageType,
    VectorStorage,
)

from .graph_service_impl import GraphService
from .redis_service_impl import RedisService

T = TypeVar('T')


class FileStorageBackend(StorageBackend[T]):
    """Concrete implementation of file-based storage backend."""

    def __init__(self, base_path: str = "storage"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def store(self, key: str, data: T, metadata: StorageMetadata | None = None) -> bool:
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

    def retrieve(self, key: str) -> T | None:
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

    def list_keys(self, pattern: str | None = None) -> list[str]:
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


class AzureSearchVectorStorage(VectorStorage):
    """Concrete implementation of Azure Cognitive Search vector storage."""

    def __init__(self, search_client: SearchClient = None, index_client: SearchIndexClient = None,
                 index_name: str = None, dimension: int = 384):
        self.dimension = dimension
        self.index_name = index_name or f"{settings.elasticsearch_index_prefix}-vectors"

        # Initialize Azure Search clients
        if search_client and index_client:
            self.search_client = search_client
            self.index_client = index_client
        else:
            self.search_client = self._create_search_client()
            self.index_client = self._create_index_client()

        # Ensure index exists
        self._ensure_index_exists()

    def _create_search_client(self) -> SearchClient:
        """Create Azure Search client."""
        try:
            return SearchClient(
                endpoint=settings.elasticsearch_url,
                index_name=self.index_name,
                credential=AzureKeyCredential(settings.elasticsearch_api_key)
            )
        except Exception as e:
            print(f"Error creating Azure Search client: {e}")
            return None

    def _create_index_client(self) -> SearchIndexClient:
        """Create Azure Search index client."""
        try:
            return SearchIndexClient(
                endpoint=settings.elasticsearch_url,
                credential=AzureKeyCredential(settings.elasticsearch_api_key)
            )
        except Exception as e:
            print(f"Error creating Azure Search index client: {e}")
            return None

    def _ensure_index_exists(self):
        """Ensure the Azure Search index exists with proper mapping."""
        if not self.index_client:
            return

        try:
            # Check if index exists
            if not self.index_client.get_index(self.index_name):
                self._create_index()
        except Exception:
            # Index doesn't exist, create it
            self._create_index()

    def _create_index(self):
        """Create Azure Search index with proper mapping."""
        index_definition = {
            "name": self.index_name,
            "fields": [
                {
                    "name": "id",
                    "type": "Edm.String",
                    "key": True,
                    "searchable": False,
                    "filterable": True,
                    "sortable": False,
                    "facetable": False
                },
                {
                    "name": "content",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": False,
                    "sortable": False,
                    "facetable": False
                },
                {
                    "name": "content_vector",
                    "type": "Collection(Edm.Single)",
                    "searchable": True,
                    "filterable": False,
                    "sortable": False,
                    "facetable": False,
                    "dimensions": self.dimension,
                    "vectorSearchConfiguration": "default-vector-config"
                },
                {
                    "name": "page",
                    "type": "Edm.Int32",
                    "searchable": False,
                    "filterable": True,
                    "sortable": True,
                    "facetable": False
                },
                {
                    "name": "chunk_id",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "sortable": False,
                    "facetable": False
                },
                {
                    "name": "entities",
                    "type": "Collection(Edm.String)",
                    "searchable": False,
                    "filterable": True,
                    "sortable": False,
                    "facetable": True
                },
                {
                    "name": "entity_labels",
                    "type": "Collection(Edm.String)",
                    "searchable": False,
                    "filterable": True,
                    "sortable": False,
                    "facetable": True
                },
                {
                    "name": "chapter_title",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": True,
                    "sortable": False,
                    "facetable": True
                },
                {
                    "name": "chunk_type",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "sortable": False,
                    "facetable": True
                },
                {
                    "name": "lore_significance",
                    "type": "Edm.Double",
                    "searchable": False,
                    "filterable": True,
                    "sortable": True,
                    "facetable": False
                }
            ],
            "vectorSearch": {
                "algorithmConfigurations": [
                    {
                        "name": "default-vector-config",
                        "kind": "hnsw",
                        "hnswParameters": {
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    }
                ]
            }
        }

        try:
            from azure.search.documents.indexes.models import SearchIndex
            search_index = SearchIndex(**index_definition)
            self.index_client.create_index(search_index)
            print(f"Created Azure Search index: {self.index_name}")
        except Exception as e:
            print(f"Error creating Azure Search index: {e}")

    def add_vectors(self, vectors: list[list[float]], metadata: list[dict[str, Any]]) -> bool:
        """Add vectors to Azure Search index."""
        if not self.search_client:
            print("Azure Search client not available")
            return False

        try:
            if not vectors:
                return True

            # Prepare documents for indexing
            documents = []
            for i, (vector, meta) in enumerate(zip(vectors, metadata, strict=False)):
                doc = {
                    "id": meta.get("chunk_id", f"chunk_{i}"),
                    "content_vector": vector,
                    **meta
                }
                documents.append(doc)

            # Index documents
            result = self.search_client.upload_documents(documents)
            print(f"Indexed {len(documents)} vectors to Azure Search")
            return len(result) > 0

        except Exception as e:
            print(f"Error adding vectors to Azure Search: {e}")
            return False

    def search_vectors(self, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """Search for similar vectors using Azure Search."""
        if not self.search_client:
            return []

        try:
            # Create vectorized query
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )

            # Search
            results = self.search_client.search(
                search_text="",
                vector_queries=[vector_query],
                top=top_k
            )

            search_results = []
            for result in results:
                search_result = {
                    "content": result.get("content", ""),
                    "score": result.get("@search.score", 0.0),
                    "metadata": {
                        "chunk_id": result.get("chunk_id", ""),
                        "page": result.get("page", 0),
                        "entities": result.get("entities", []),
                        "entity_labels": result.get("entity_labels", []),
                        "chapter_title": result.get("chapter_title", ""),
                        "chunk_type": result.get("chunk_type", "standard"),
                        "lore_significance": result.get("lore_significance", 0.0)
                    },
                    "index": result.get("id", "")
                }
                search_results.append(search_result)

            return search_results

        except Exception as e:
            print(f"Error searching vectors in Azure Search: {e}")
            return []

    def get_vector_count(self) -> int:
        """Get the number of stored vectors."""
        if not self.search_client:
            return 0

        try:
            # Get document count
            result = self.search_client.get_document_count()
            return result
        except Exception as e:
            print(f"Error getting vector count: {e}")
            return 0

    def clear_vectors(self) -> bool:
        """Clear all vectors from storage."""
        if not self.index_client:
            return False

        try:
            # Delete and recreate index
            self.index_client.delete_index(self.index_name)
            self._create_index()
            return True
        except Exception as e:
            print(f"Error clearing vectors: {e}")
            return False


class Neo4jGraphStorage(GraphStorage):
    """Concrete implementation of Neo4j graph storage."""

    def __init__(self, graph_service: GraphService):
        self.graph_service = graph_service

    def add_node(self, node_id: str, labels: list[str], properties: dict[str, Any]) -> bool:
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

    def add_relationship(self, from_node: str, to_node: str, relationship_type: str, properties: dict[str, Any]) -> bool:
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

    def query_nodes(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Query nodes in the graph."""
        try:
            # Use existing graph service implementation
            results = self.graph_service.get_related_info_for_entity(query)
            return results

        except Exception as e:
            print(f"Error querying nodes: {e}")
            return []

    def query_relationships(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        try:
            # Use existing Redis service implementation
            return self.redis_service.get_cached_query_result(key, 5, None)
        except Exception as e:
            print(f"Error getting from cache: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
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

    def clear(self, pattern: str | None = None) -> bool:
        """Clear cache entries, optionally filtered by pattern."""
        try:
            return self.redis_service.clear_cache(pattern)
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            return self.redis_service.get_cache_stats()
        except Exception as e:
            print(f"Error getting cache stats: {e}")
            return {"error": str(e)}
