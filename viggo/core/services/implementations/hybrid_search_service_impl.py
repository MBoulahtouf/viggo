"""
Concrete implementation of hybrid search service following SOLID principles.
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from sentence_transformers import SentenceTransformer
from viggo.core.config import settings
from viggo.core.services.interfaces.hybrid_search_service import IHybridSearchService
from viggo.core.services.content_filter_service import ContentFilterService


class HybridSearchService(IHybridSearchService):
    """
    Hybrid search service that combines Azure Cognitive Search with FAISS vector search.
    Provides both keyword-based and semantic search capabilities.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.content_filter = ContentFilterService()
        
        # Initialize Azure Search clients with error handling
        try:
            self.search_client = SearchClient(
                endpoint=settings.elasticsearch_url,
                index_name=f"{settings.elasticsearch_index_prefix}-documents",
                credential=AzureKeyCredential(settings.elasticsearch_api_key)
            )
            self.index_client = SearchIndexClient(
                endpoint=settings.elasticsearch_url,
                credential=AzureKeyCredential(settings.elasticsearch_api_key)
            )
            print(f"[HybridSearchService] Successfully initialized with endpoint: {settings.elasticsearch_url}")
        except Exception as e:
            print(f"[HybridSearchService] Failed to initialize Azure Search clients: {e}")
            self.search_client = None
            self.index_client = None
    
    def create_index(self, index_name: str = None) -> bool:
        """
        Create Azure Cognitive Search index for hybrid search.
        
        Args:
            index_name: Name of the index to create
            
        Returns:
            True if successful, False otherwise
        """
        if index_name is None:
            index_name = f"{settings.elasticsearch_index_prefix}-documents"
        
        try:
            # Check if index already exists
            try:
                existing_index = self.index_client.get_index(index_name)
                print(f"Index {index_name} already exists")
                return True
            except Exception:
                # Index doesn't exist, proceed with creation
                pass
            # Define the index schema
            index_definition = {
                "name": index_name,
                "fields": [
                    {
                        "name": "id",
                        "type": "Edm.String",
                        "key": True,
                        "searchable": False,
                        "filterable": True,
                        "sortable": True,
                        "facetable": False
                    },
                    {
                        "name": "content",
                        "type": "Edm.String",
                        "searchable": True,
                        "filterable": False,
                        "sortable": False,
                        "facetable": False,
                        "analyzer": "standard"
                    },
                    {
                        "name": "page",
                        "type": "Edm.Int32",
                        "searchable": False,
                        "filterable": True,
                        "sortable": True,
                        "facetable": True
                    },
                    {
                        "name": "word_count",
                        "type": "Edm.Int32",
                        "searchable": False,
                        "filterable": True,
                        "sortable": True,
                        "facetable": True
                    },
                    {
                        "name": "char_count",
                        "type": "Edm.Int32",
                        "searchable": False,
                        "filterable": True,
                        "sortable": True,
                        "facetable": True
                    },
                    {
                        "name": "entities",
                        "type": "Collection(Edm.String)",
                        "searchable": True,
                        "filterable": True,
                        "sortable": False,
                        "facetable": True
                    },
                    {
                        "name": "entity_labels",
                        "type": "Collection(Edm.String)",
                        "searchable": True,
                        "filterable": True,
                        "sortable": False,
                        "facetable": True
                    },
                    {
                        "name": "chapter_title",
                        "type": "Edm.String",
                        "searchable": True,
                        "filterable": True,
                        "sortable": True,
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
                        "name": "content_type",
                        "type": "Edm.String",
                        "searchable": False,
                        "filterable": True,
                        "sortable": False,
                        "facetable": True
                    },
                    {
                        "name": "document_metadata",
                        "type": "Edm.String",
                        "searchable": False,
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
                        "dimensions": 384,  # all-MiniLM-L6-v2 embedding dimension
                        "vectorSearchConfiguration": "default-vector-config"
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
            
            # Create the index
            from azure.search.documents.indexes.models import SearchIndex
            search_index = SearchIndex(**index_definition)
            self.index_client.create_index(search_index)
            print(f"Successfully created index: {index_name}")
            return True
            
        except Exception as e:
            print(f"Error creating index: {e}")
            return False
    
    def index_documents(self, documents: List[Dict], index_name: str = None) -> bool:
        """
        Index documents in Azure Cognitive Search with content filtering.
        
        Args:
            documents: List of document dictionaries
            index_name: Name of the index to use
            
        Returns:
            True if successful, False otherwise
        """
        if index_name is None:
            index_name = f"{settings.elasticsearch_index_prefix}-documents"
        
        try:
            # Filter documents to only include story content
            print("🔍 Applying content filtering before indexing...")
            filtered_docs, filter_stats = self.content_filter.filter_chunks_for_indexing(documents)
            
            print(f"📊 Content filtering results:")
            print(f"   Total chunks: {filter_stats['total_chunks']}")
            print(f"   Filtered out: {filter_stats['filtered_out']}")
            print(f"   Story content: {filter_stats['story_content']}")
            print(f"   Metadata: {filter_stats['metadata']}")
            print(f"   Bibliography: {filter_stats['bibliography']}")
            print(f"   Publisher info: {filter_stats['publisher_info']}")
            print(f"   Technical: {filter_stats['technical']}")
            print(f"   Preface: {filter_stats['preface']}")
            
            # Prepare documents for indexing
            search_documents = []
            for i, doc in enumerate(filtered_docs):
                # Add content type classification
                enhanced_doc = self.content_filter.add_content_type_to_chunk(doc)
                
                search_doc = {
                    "id": str(i),
                    "content": enhanced_doc["content"],
                    "page": enhanced_doc.get("page", 0),
                    "word_count": enhanced_doc.get("word_count", len(enhanced_doc["content"].split())),
                    "char_count": enhanced_doc.get("char_count", len(enhanced_doc["content"])),
                    "entities": enhanced_doc.get("entities", []),
                    "entity_labels": enhanced_doc.get("entity_labels", []),
                    "chapter_title": enhanced_doc.get("chapter_title", ""),
                    "chunk_type": enhanced_doc.get("chunk_type", "standard"),
                    "document_metadata": json.dumps(enhanced_doc.get("document_metadata", {}))
                }
                search_documents.append(search_doc)
            
            # Upload documents to the index
            result = self.search_client.upload_documents(search_documents)
            
            # Check for any failures
            failed_docs = [doc for doc in result if not doc.succeeded]
            if failed_docs:
                print(f"Failed to index {len(failed_docs)} documents")
                for doc in failed_docs:
                    print(f"Error: {doc.error_message}")
                return False
            
            print(f"✅ Successfully indexed {len(search_documents)} story content documents")
            return True
            
        except Exception as e:
            print(f"Error indexing documents: {e}")
            return False
    
    def hybrid_search(self, query: str, k: int = 5, page_filter: Optional[int] = None) -> List[Dict]:
        """
        Perform hybrid search combining keyword and semantic search.
        
        Args:
            query: Search query
            k: Number of results to return
            page_filter: Optional page number filter
            
        Returns:
            List of search results with scores
        """
        try:
            # For now, use keyword search only since vector search isn't available in simple index
            return self.keyword_search(query, k, page_filter)
            
        except Exception as e:
            print(f"Error performing hybrid search: {e}")
            return []
    
    def keyword_search(self, query: str, k: int = 5, page_filter: Optional[int] = None) -> List[Dict]:
        """
        Perform keyword-only search.
        
        Args:
            query: Search query
            k: Number of results to return
            page_filter: Optional page number filter
            
        Returns:
            List of search results
        """
        try:
            # Check if search client is available
            if not self.search_client:
                print("[HybridSearchService] Search client not available, returning empty results")
                return []
            
            search_params = {
                "top": k,
                "include_total_count": True
            }
            
            # Build filter expression (content filtering is done at indexing time)
            if page_filter is not None:
                search_params["filter"] = f"page le {page_filter}"
            
            results = self.search_client.search(
                search_text=query,
                **search_params
            )
            
            search_results = []
            for result in results:
                search_results.append({
                    "content": result["content"],
                    "page": result.get("page", 0),
                    "word_count": result.get("word_count", 0),
                    "entities": result.get("entities", []),
                    "entity_labels": result.get("entity_labels", []),
                    "chapter_title": result.get("chapter_title", ""),
                    "chunk_type": result.get("chunk_type", "standard"),
                    "document_metadata": json.loads(result.get("document_metadata", "{}")),
                    "score": result.get("@search.score", 0.0)
                })
            
            return search_results
            
        except Exception as e:
            print(f"Error performing keyword search: {e}")
            return []
    
    def semantic_search(self, query: str, k: int = 5, page_filter: Optional[int] = None) -> List[Dict]:
        """
        Perform semantic-only search using vector similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            page_filter: Optional page number filter
            
        Returns:
            List of search results
        """
        try:
            query_embedding = self.model.encode(query).tolist()
            
            search_params = {
                "top": k,
                "include_total_count": True
            }
            
            if page_filter is not None:
                search_params["filter"] = f"page le {page_filter}"
            
            results = self.search_client.search(
                search_text="",  # Empty text for pure vector search
                vector_queries=[
                    VectorizedQuery(
                        vector=query_embedding,
                        k_nearest_neighbors=k,
                        fields="content_vector"
                    )
                ],
                **search_params
            )
            
            search_results = []
            for result in results:
                search_results.append({
                    "content": result["content"],
                    "page": result.get("page", 0),
                    "word_count": result.get("word_count", 0),
                    "entities": result.get("entities", []),
                    "entity_labels": result.get("entity_labels", []),
                    "chapter_title": result.get("chapter_title", ""),
                    "chunk_type": result.get("chunk_type", "standard"),
                    "document_metadata": json.loads(result.get("document_metadata", "{}")),
                    "score": result.get("@search.score", 0.0)
                })
            
            return search_results
            
        except Exception as e:
            print(f"Error performing semantic search: {e}")
            return []
    
    def get_index_stats(self, index_name: str = None) -> Dict:
        """
        Get statistics about the search index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Dictionary with index statistics
        """
        if index_name is None:
            index_name = f"{settings.elasticsearch_index_prefix}-documents"
        
        try:
            # Get index statistics
            stats = self.search_client.get_document_count()
            
            return {
                "index_name": index_name,
                "document_count": stats,
                "status": "active"
            }
            
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {
                "index_name": index_name,
                "document_count": 0,
                "status": "error",
                "error": str(e)
            }
