#!/usr/bin/env python3
"""
Performance optimization module for hybrid RAG system.
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
    """Cache for embeddings to avoid recomputation."""
    
    def __init__(self, cache_size: int = 1000, cache_file: str = "embedding_cache.pkl"):
        self.cache_size = cache_size
        self.cache_file = cache_file
        self.cache: Dict[str, np.ndarray] = {}
        self.access_times: Dict[str, float] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', {})
                    self.access_times = data.get('access_times', {})
                print(f"Loaded embedding cache with {len(self.cache)} entries")
            except Exception as e:
                print(f"Failed to load embedding cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            data = {
                'cache': self.cache,
                'access_times': self.access_times
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Failed to save embedding cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        key = self._get_cache_key(text)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        """Store embedding in cache."""
        key = self._get_cache_key(text)
        
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.cache_size:
            self._evict_oldest()
        
        self.cache[key] = embedding
        self.access_times[key] = time.time()
    
    def _evict_oldest(self):
        """Evict the least recently used entry."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_times.clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

class QueryResultCache:
    """Cache for query results to avoid redundant processing."""
    
    def __init__(self, cache_size: int = 500, cache_file: str = "query_cache.pkl"):
        self.cache_size = cache_size
        self.cache_file = cache_file
        self.cache: Dict[str, Dict] = {}
        self.access_times: Dict[str, float] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', {})
                    self.access_times = data.get('access_times', {})
                print(f"Loaded query cache with {len(self.cache)} entries")
            except Exception as e:
                print(f"Failed to load query cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            data = {
                'cache': self.cache,
                'access_times': self.access_times
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Failed to save query cache: {e}")
    
    def _get_cache_key(self, query: str, top_k: int, page_filter: Optional[int] = None) -> str:
        """Generate cache key for query parameters."""
        key_data = f"{query.lower().strip()}:{top_k}:{page_filter}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, top_k: int, page_filter: Optional[int] = None) -> Optional[Dict]:
        """Get cached query result."""
        key = self._get_cache_key(query, top_k, page_filter)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, query: str, top_k: int, page_filter: Optional[int], result: Dict):
        """Store query result in cache."""
        key = self._get_cache_key(query, top_k, page_filter)
        
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.cache_size:
            self._evict_oldest()
        
        self.cache[key] = result
        self.access_times[key] = time.time()
    
    def _evict_oldest(self):
        """Evict the least recently used entry."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_times.clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

class AdaptiveTimeoutManager:
    """Manages adaptive timeouts based on source performance."""
    
    def __init__(self, base_timeout: float = 10.0, min_timeout: float = 2.0, max_timeout: float = 30.0):
        self.base_timeout = base_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.source_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
    
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

class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, 
                 embedding_cache_size: int = 1000,
                 query_cache_size: int = 500,
                 base_timeout: float = 10.0):
        
        self.embedding_cache = EmbeddingCache(embedding_cache_size)
        self.query_cache = QueryResultCache(query_cache_size)
        self.timeout_manager = AdaptiveTimeoutManager(base_timeout)
        
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
        
        return {
            'total_queries': self.total_queries,
            'embedding_cache': {
                'size': len(self.embedding_cache.cache),
                'hits': self.cache_hits['embedding'],
                'misses': self.cache_misses['embedding'],
                'hit_rate': f"{embedding_hit_rate:.1f}%"
            },
            'query_cache': {
                'size': len(self.query_cache.cache),
                'hits': self.cache_hits['query'],
                'misses': self.cache_misses['query'],
                'hit_rate': f"{query_hit_rate:.1f}%"
            },
            'source_performance': self.timeout_manager.get_performance_summary()
        }
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.embedding_cache.clear()
        self.query_cache.clear()
        self.cache_hits = {'embedding': 0, 'query': 0}
        self.cache_misses = {'embedding': 0, 'query': 0}
    
    def save_caches(self):
        """Save all caches to disk."""
        self.embedding_cache._save_cache()
        self.query_cache._save_cache()
