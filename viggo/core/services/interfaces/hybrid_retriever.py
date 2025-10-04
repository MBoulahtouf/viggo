"""
Hybrid retriever interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class IHybridRetriever(ABC):
    """Interface for hybrid retrieval operations."""
    
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5, page_filter: Optional[int] = None) -> Dict:
        """Retrieve results from multiple sources in parallel."""
        pass
    
    @abstractmethod
    def _semantic_search(self, query: str, top_k: int, page_filter: Optional[int] = None) -> List[Dict]:
        """Perform semantic search using FAISS."""
        pass
    
    @abstractmethod
    def _neo4j_lookup(self, query: str) -> List[Dict]:
        """Perform Neo4j graph lookup."""
        pass
    
    @abstractmethod
    def _keyword_search(self, query: str, top_k: int, page_filter: Optional[int] = None) -> List[Dict]:
        """Perform keyword search using Azure Cognitive Search."""
        pass
    
    @abstractmethod
    def _extract_entities_for_neo4j(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query for Neo4j lookup."""
        pass
    
    @abstractmethod
    def _combine_and_rank(self, semantic_results: List[Dict], neo4j_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """Combine and rank results from all sources."""
        pass
    
    @abstractmethod
    def _apply_evidence_alignment(self, results: List[Dict]) -> None:
        """Apply evidence alignment to results."""
        pass
    
    @abstractmethod
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        pass
    
    @abstractmethod
    def create_hybrid_prompt(self, query: str, results: List[Dict]) -> str:
        """Create a hybrid prompt from results."""
        pass
    
    @abstractmethod
    def clear_cache(self, cache_type: str = "all") -> bool:
        """Clear cache entries."""
        pass
    
    @abstractmethod
    def get_cache_info(self) -> Dict:
        """Get cache information."""
        pass
