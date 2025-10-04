"""
Concrete implementation of performance optimizer following SOLID principles.
"""

import time
import hashlib
import json
import pickle
import os
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from viggo.core.config import settings
from viggo.core.services.interfaces.performance_optimizer import IPerformanceOptimizer

@dataclass
class PerformanceMetrics:
    """Performance metrics for a retrieval source."""
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    total_queries: int = 0
    recent_times: deque = None
    
    def __post_init__(self):
        if self.recent_times is None:
            self.recent_times = deque(maxlen=10)  # Keep last 10 response times
    
    def update(self, response_time: float, success: bool = True):
        """Update metrics with new query result."""
        self.recent_times.append(response_time)
        self.avg_response_time = sum(self.recent_times) / len(self.recent_times)
        self.total_queries += 1
        
        if not success:
            # Simple success rate calculation
            self.success_rate = max(0.0, self.success_rate - 0.1)
        else:
            self.success_rate = min(1.0, self.success_rate + 0.01)

class EmbeddingCache:
    """Cache for embeddings using in-memory storage with optional Redis backing."""
    
    def __init__(self, cache_size: int = None, redis_cache_service=None):
        self.cache_size = cache_size or settings.cache_max_size
        self.redis_cache_service = redis_cache_service
        self.memory_cache = {}  # In-memory fallback cache
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        # Try Redis cache first if available
        if self.redis_cache_service and self.redis_cache_service.is_available():
            cached = self.redis_cache_service.get_cached_embedding(text)
            if cached is not None:
                return cached
        
        # Fallback to memory cache
        return self.memory_cache.get(text)
    
    def put(self, text: str, embedding: np.ndarray):
        """Store embedding in cache."""
        # Store in Redis if available
        if self.redis_cache_service and self.redis_cache_service.is_available():
            self.redis_cache_service.cache_embedding(text, embedding)
        
        # Also store in memory cache
        if len(self.memory_cache) < self.cache_size:
            self.memory_cache[text] = embedding
    
    def clear(self):
        """Clear the embedding cache."""
        if self.redis_cache_service and self.redis_cache_service.is_available():
            self.redis_cache_service.clear_cache("embedding")
        self.memory_cache.clear()

class QueryResultCache:
    """Cache for query results using in-memory storage with optional Redis backing."""
    
    def __init__(self, cache_size: int = None, redis_cache_service=None):
        self.cache_size = cache_size or settings.cache_max_size
        self.redis_cache_service = redis_cache_service
        self.memory_cache = {}  # In-memory fallback cache
    
    def _get_cache_key(self, query: str, top_k: int, page_filter: Optional[int] = None) -> str:
        """Generate cache key for query parameters."""
        key_data = f"{query.lower().strip()}:{top_k}:{page_filter}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, top_k: int, page_filter: Optional[int] = None) -> Optional[Dict]:
        """Get cached query result."""
        # Try Redis cache first if available
        if self.redis_cache_service and self.redis_cache_service.is_available():
            cached = self.redis_cache_service.get_cached_query_result(query, top_k, page_filter)
            if cached is not None:
                return cached
        
        # Fallback to memory cache
        cache_key = self._get_cache_key(query, top_k, page_filter)
        return self.memory_cache.get(cache_key)
    
    def put(self, query: str, top_k: int, page_filter: Optional[int], result: Dict):
        """Store query result in cache."""
        # Store in Redis if available
        if self.redis_cache_service and self.redis_cache_service.is_available():
            self.redis_cache_service.cache_query_result(query, top_k, page_filter, result)
        
        # Also store in memory cache
        cache_key = self._get_cache_key(query, top_k, page_filter)
        if len(self.memory_cache) < self.cache_size:
            self.memory_cache[cache_key] = result
    
    def clear(self):
        """Clear the query cache."""
        if self.redis_cache_service and self.redis_cache_service.is_available():
            self.redis_cache_service.clear_cache("query")
        self.memory_cache.clear()

class AdaptiveTimeoutManager:
    """Manages adaptive timeouts based on source performance."""
    
    def __init__(self, base_timeout: float = 10.0, min_timeout: float = 2.0, max_timeout: float = 30.0, redis_cache_service=None):
        self.base_timeout = base_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.source_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self.redis_cache_service = redis_cache_service
        self._load_metrics_from_redis()
    
    def _load_metrics_from_redis(self):
        """Load performance metrics from Redis cache."""
        try:
            if self.redis_cache_service and self.redis_cache_service.is_available():
                cached_metrics = self.redis_cache_service.get_cached_performance_metrics("performance_metrics")
                if cached_metrics:
                    for source, metrics_data in cached_metrics.items():
                        if source != "connected":  # Skip Redis connection status
                            metrics = PerformanceMetrics()
                            metrics.avg_response_time = metrics_data.get('avg_response_time', 0.0)
                            metrics.success_rate = metrics_data.get('success_rate', 1.0)
                            metrics.total_queries = metrics_data.get('total_queries', 0)
                            # Convert recent_times back to deque if available
                            recent_times = metrics_data.get('recent_times', [])
                            if recent_times:
                                metrics.recent_times = deque(recent_times, maxlen=10)
                            self.source_metrics[source] = metrics
        except Exception as e:
            print(f"Failed to load performance metrics from Redis: {e}")
    
    def _save_metrics_to_redis(self):
        """Save performance metrics to Redis cache."""
        try:
            if self.redis_cache_service and self.redis_cache_service.is_available():
                metrics_data = {}
                for source, metrics in self.source_metrics.items():
                    metrics_data[source] = {
                        'avg_response_time': metrics.avg_response_time,
                        'success_rate': metrics.success_rate,
                        'total_queries': metrics.total_queries,
                        'recent_times': list(metrics.recent_times)
                    }
                self.redis_cache_service.cache_performance_metrics("performance_metrics", metrics_data)
        except Exception as e:
            print(f"Failed to save performance metrics to Redis: {e}")
    
    def get_timeout(self, source: str) -> float:
        """Get adaptive timeout for a source based on its performance."""
        metrics = self.source_metrics[source]
        
        if metrics.total_queries < 3:
            return self.base_timeout  # Use base timeout for new sources
        
        # Calculate timeout based on average response time and success rate
        avg_time = metrics.avg_response_time
        success_rate = metrics.success_rate
        
        # Increase timeout if success rate is low or response time is high
        if success_rate < 0.8:
            timeout = min(self.max_timeout, avg_time * 3.0)
        elif avg_time > self.base_timeout:
            timeout = min(self.max_timeout, avg_time * 1.5)
        else:
            timeout = max(self.min_timeout, avg_time * 1.2)
        
        return timeout
    
    def update_metrics(self, source: str, response_time: float, success: bool = True):
        """Update performance metrics for a source."""
        self.source_metrics[source].update(response_time, success)
        # Save to Redis periodically (every 10 updates)
        if self.source_metrics[source].total_queries % 10 == 0:
            self._save_metrics_to_redis()
    
    def get_performance_summary(self) -> Dict[str, Dict]:
        """Get performance summary for all sources."""
        summary = {}
        for source, metrics in self.source_metrics.items():
            summary[source] = {
                'avg_response_time': metrics.avg_response_time,
                'success_rate': metrics.success_rate,
                'total_queries': metrics.total_queries,
                'current_timeout': self.get_timeout(source)
            }
        return summary

class PerformanceOptimizer(IPerformanceOptimizer):
    """Main performance optimization coordinator."""
    
    def __init__(self, 
                 embedding_cache_size: int = None,
                 query_cache_size: int = None,
                 base_timeout: float = 10.0,
                 redis_cache_service=None):
        
        self.redis_cache_service = redis_cache_service
        self.embedding_cache = EmbeddingCache(embedding_cache_size, redis_cache_service)
        self.query_cache = QueryResultCache(query_cache_size, redis_cache_service)
        self.timeout_manager = AdaptiveTimeoutManager(base_timeout, redis_cache_service=redis_cache_service)
        
        # Performance tracking
        self.total_queries = 0
        self.cache_hits = {'embedding': 0, 'query': 0}
        self.cache_misses = {'embedding': 0, 'query': 0}
    
    def get_embedding(self, text: str, model) -> np.ndarray:
        """Get embedding with caching."""
        # Try cache first
        cached_embedding = self.embedding_cache.get(text)
        if cached_embedding is not None:
            self.cache_hits['embedding'] += 1
            return cached_embedding
        
        # Generate and cache embedding
        self.cache_misses['embedding'] += 1
        embedding = model.encode([text])[0]
        self.embedding_cache.put(text, embedding)
        return embedding
    
    def get_cached_query_result(self, query: str, top_k: int, page_filter: Optional[int] = None) -> Optional[Dict]:
        """Get cached query result."""
        cached_result = self.query_cache.get(query, top_k, page_filter)
        if cached_result is not None:
            self.cache_hits['query'] += 1
            return cached_result
        
        self.cache_misses['query'] += 1
        return None
    
    def cache_query_result(self, query: str, top_k: int, page_filter: Optional[int], result: Dict):
        """Cache query result."""
        self.query_cache.put(query, top_k, page_filter, result)
    
    def get_source_timeout(self, source: str) -> float:
        """Get adaptive timeout for a source."""
        return self.timeout_manager.get_timeout(source)
    
    def update_source_performance(self, source: str, response_time: float, success: bool = True):
        """Update source performance metrics."""
        self.timeout_manager.update_metrics(source, response_time, success)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        total_embedding_requests = self.cache_hits['embedding'] + self.cache_misses['embedding']
        total_query_requests = self.cache_hits['query'] + self.cache_misses['query']
        
        embedding_hit_rate = (self.cache_hits['embedding'] / total_embedding_requests * 100) if total_embedding_requests > 0 else 0
        query_hit_rate = (self.cache_hits['query'] / total_query_requests * 100) if total_query_requests > 0 else 0
        
        # Get Redis cache statistics
        redis_stats = {}
        if self.redis_cache_service and self.redis_cache_service.is_available():
            redis_stats = self.redis_cache_service.get_cache_stats()
        
        return {
            'total_queries': self.total_queries,
            'embedding_cache': {
                'hits': self.cache_hits['embedding'],
                'misses': self.cache_misses['embedding'],
                'hit_rate': f"{embedding_hit_rate:.1f}%"
            },
            'query_cache': {
                'hits': self.cache_hits['query'],
                'misses': self.cache_misses['query'],
                'hit_rate': f"{query_hit_rate:.1f}%"
            },
            'redis_cache': redis_stats,
            'source_performance': self.timeout_manager.get_performance_summary()
        }
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.embedding_cache.clear()
        self.query_cache.clear()
        if self.redis_cache_service and self.redis_cache_service.is_available():
            self.redis_cache_service.clear_cache()
        self.cache_hits = {'embedding': 0, 'query': 0}
        self.cache_misses = {'embedding': 0, 'query': 0}
    
    def save_caches(self):
        """Save all caches to Redis."""
        # Redis automatically persists data, but we can save performance metrics
        self.timeout_manager._save_metrics_to_redis()
    
    def clear_cache(self):
        """Clear in-memory caches only."""
        self.embedding_cache.memory_cache.clear()
        self.query_cache.memory_cache.clear()
        self.cache_hits = {'embedding': 0, 'query': 0}
        self.cache_misses = {'embedding': 0, 'query': 0}
