"""
Hybrid retriever interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod


class IHybridRetriever(ABC):
    """Interface for hybrid retrieval operations."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5, page_filter: int | None = None) -> dict:
        """Retrieve results from multiple sources in parallel."""
        pass

    @abstractmethod
    def _semantic_search(self, query: str, top_k: int, page_filter: int | None = None) -> list[dict]:
        """Perform semantic search using FAISS."""
        pass

    @abstractmethod
    def _neo4j_lookup(self, query: str) -> list[dict]:
        """Perform Neo4j graph lookup."""
        pass

    @abstractmethod
    def _keyword_search(self, query: str, top_k: int, page_filter: int | None = None) -> list[dict]:
        """Perform keyword search using Azure Cognitive Search."""
        pass

    @abstractmethod
    def _extract_entities_for_neo4j(self, query: str) -> dict[str, list[str]]:
        """Extract entities from query for Neo4j lookup."""
        pass

    @abstractmethod
    def _combine_and_rank(self, semantic_results: list[dict], neo4j_results: list[dict], keyword_results: list[dict]) -> list[dict]:
        """Combine and rank results from all sources."""
        pass

    @abstractmethod
    def _apply_evidence_alignment(self, results: list[dict]) -> None:
        """Apply evidence alignment to results."""
        pass

    @abstractmethod
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        pass

    @abstractmethod
    def create_hybrid_prompt(self, query: str, results: list[dict]) -> str:
        """Create a hybrid prompt from results."""
        pass

    @abstractmethod
    def clear_cache(self, cache_type: str = "all") -> bool:
        """Clear cache entries."""
        pass

    @abstractmethod
    def get_cache_info(self) -> dict:
        """Get cache information."""
        pass
