"""
Redis service interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod

import numpy as np


class IRedisService(ABC):
    """Interface for Redis caching operations."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if Redis cache is available."""
        pass

    @abstractmethod
    def cache_query_result(self, query: str, top_k: int, page_filter: int | None, result: dict, ttl: int | None = None) -> bool:
        """Cache query result with TTL."""
        pass

    @abstractmethod
    def get_cached_query_result(self, query: str, top_k: int, page_filter: int | None) -> dict | None:
        """Get cached query result."""
        pass

    @abstractmethod
    def cache_embedding(self, text: str, embedding: np.ndarray, ttl: int | None = None) -> bool:
        """Cache embedding with TTL."""
        pass

    @abstractmethod
    def get_cached_embedding(self, text: str) -> np.ndarray | None:
        """Get cached embedding."""
        pass

    @abstractmethod
    def find_similar_cached_embeddings(self, query_embedding: np.ndarray, similarity_threshold: float = 0.9) -> list[dict]:
        """Find similar cached embeddings."""
        pass

    @abstractmethod
    def cache_performance_metrics(self, source: str, metrics: dict, ttl: int | None = None) -> bool:
        """Cache performance metrics."""
        pass

    @abstractmethod
    def get_cached_performance_metrics(self, source: str) -> dict | None:
        """Get cached performance metrics."""
        pass

    @abstractmethod
    def cache_session_data(self, session_id: str, data: dict, ttl: int | None = None) -> bool:
        """Cache session data."""
        pass

    @abstractmethod
    def get_session_data(self, session_id: str) -> dict | None:
        """Get session data."""
        pass

    @abstractmethod
    def clear_cache(self, pattern: str = None) -> bool:
        """Clear cache entries."""
        pass

    @abstractmethod
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        pass

    @abstractmethod
    def health_check(self) -> dict:
        """Perform health check."""
        pass
