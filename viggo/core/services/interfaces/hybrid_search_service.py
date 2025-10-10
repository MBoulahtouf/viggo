"""
Hybrid search service interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod


class IHybridSearchService(ABC):
    """Interface for hybrid search operations."""

    @abstractmethod
    def create_index(self, index_name: str = None) -> bool:
        """Create a search index."""
        pass

    @abstractmethod
    def index_documents(self, documents: list[dict], index_name: str = None) -> bool:
        """Index documents in the search service."""
        pass

    @abstractmethod
    def hybrid_search(self, query: str, k: int = 5, page_filter: int | None = None) -> list[dict]:
        """Perform hybrid search combining keyword and semantic search."""
        pass

    @abstractmethod
    def keyword_search(self, query: str, k: int = 5, page_filter: int | None = None) -> list[dict]:
        """Perform keyword-only search."""
        pass

    @abstractmethod
    def semantic_search(self, query: str, k: int = 5, page_filter: int | None = None) -> list[dict]:
        """Perform semantic search using vector embeddings."""
        pass

    @abstractmethod
    def get_index_stats(self, index_name: str = None) -> dict:
        """Get index statistics."""
        pass
