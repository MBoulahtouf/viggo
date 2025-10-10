"""
Performance optimizer interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class IPerformanceOptimizer(ABC):
    """Interface for performance optimization operations."""

    @abstractmethod
    def get_embedding(self, text: str, model) -> np.ndarray:
        """Get embedding with caching."""
        pass

    @abstractmethod
    def get_cached_query_result(self, query: str, top_k: int, page_filter: int | None = None) -> dict | None:
        """Get cached query result."""
        pass

    @abstractmethod
    def cache_query_result(self, query: str, top_k: int, page_filter: int | None, result: dict) -> None:
        """Cache query result."""
        pass

    @abstractmethod
    def get_source_timeout(self, source: str) -> float:
        """Get adaptive timeout for a source."""
        pass

    @abstractmethod
    def update_source_performance(self, source: str, response_time: float, success: bool = True) -> None:
        """Update source performance metrics."""
        pass

    @abstractmethod
    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        pass

    @abstractmethod
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        pass

    @abstractmethod
    def save_caches(self) -> None:
        """Save caches to persistent storage."""
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear cache entries."""
        pass
