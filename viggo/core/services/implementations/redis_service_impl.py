"""
Concrete implementation of Redis service following SOLID principles.
"""

import hashlib
import json
import pickle
import time
from typing import Any

import numpy as np
import redis
from redis.exceptions import ConnectionError, RedisError
from sentence_transformers import SentenceTransformer

from viggo.core.config import settings
from viggo.core.services.interfaces.redis_service import IRedisService


class RedisService(IRedisService):
    """
    Redis-based caching service for hybrid RAG system.
    
    Features:
    - Query result caching with TTL
    - Embedding similarity caching
    - Performance metrics storage
    - Session-based caching
    - Intelligent cache invalidation
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.redis_client = None
        self.embedding_model = None
        self.cache_prefix = "viggo:cache"
        self.embedding_prefix = "viggo:embeddings"
        self.metrics_prefix = "viggo:metrics"
        self.session_prefix = "viggo:session"

        # Cache configuration
        self.default_ttl = settings.cache_ttl
        self.max_cache_size = settings.cache_max_size

        # Initialize Redis connection
        self._initialize_redis_connection()

        # Initialize embedding model for similarity caching
        self._initialize_embedding_model()

    def _initialize_redis_connection(self):
        """Initialize Redis connection with error handling."""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                ssl=settings.redis_ssl,
                db=settings.redis_db,
                decode_responses=False,  # Keep binary for pickle compatibility
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )

            # Test connection
            self.redis_client.ping()
            print("[Redis] Successfully connected to Redis cache")

        except (ConnectionError, RedisError) as e:
            print(f"[Redis] Failed to connect to Redis: {e}")
            print("[Redis] Cache will be disabled")
            self.redis_client = None

    def _initialize_embedding_model(self):
        """Initialize embedding model for semantic similarity caching."""
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            print(f"[Redis] Initialized embedding model: {self.model_name}")
        except Exception as e:
            print(f"[Redis] Failed to initialize embedding model: {e}")
            self.embedding_model = None

    def is_available(self) -> bool:
        """Check if Redis cache is available."""
        return self.redis_client is not None

    def _generate_cache_key(self, prefix: str, key_data: Any) -> str:
        """Generate a consistent cache key from data."""
        if isinstance(key_data, str):
            key_str = key_data
        else:
            key_str = json.dumps(key_data, sort_keys=True)

        # Create hash for consistent key length
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{self.cache_prefix}:{prefix}:{key_hash}"

    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for Redis storage."""
        try:
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[Redis] Serialization error: {e}")
            return pickle.dumps({"error": "serialization_failed"})

    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from Redis storage."""
        try:
            return pickle.loads(data)
        except Exception as e:
            print(f"[Redis] Deserialization error: {e}")
            return None

    # Query Result Caching

    def cache_query_result(self, query: str, top_k: int, page_filter: int | None,
                          result: dict, ttl: int | None = None) -> bool:
        """
        Cache query result with intelligent key generation.
        
        Args:
            query: The search query
            top_k: Number of results
            page_filter: Optional page filter
            result: The result data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.is_available():
            return False

        try:
            cache_key = self._generate_cache_key("query", {
                "query": query.lower().strip(),
                "top_k": top_k,
                "page_filter": page_filter
            })

            # Add metadata to result
            cached_result = {
                "data": result,
                "timestamp": time.time(),
                "query": query,
                "top_k": top_k,
                "page_filter": page_filter,
                "cache_version": "1.0"
            }

            serialized_data = self._serialize_data(cached_result)
            ttl = ttl or self.default_ttl

            success = self.redis_client.setex(cache_key, ttl, serialized_data)

            if success:
                print(f"[Redis] Cached query result: {query[:50]}...")
                self._update_cache_metrics("query_cached")

            return bool(success)

        except Exception as e:
            print(f"[Redis] Error caching query result: {e}")
            return False

    def get_cached_query_result(self, query: str, top_k: int,
                               page_filter: int | None) -> dict | None:
        """
        Retrieve cached query result.
        
        Args:
            query: The search query
            top_k: Number of results
            page_filter: Optional page filter
            
        Returns:
            Cached result if found, None otherwise
        """
        if not self.is_available():
            return None

        try:
            cache_key = self._generate_cache_key("query", {
                "query": query.lower().strip(),
                "top_k": top_k,
                "page_filter": page_filter
            })

            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                cached_result = self._deserialize_data(cached_data)
                if cached_result and "data" in cached_result:
                    print(f"[Redis] Cache hit for query: {query[:50]}...")
                    self._update_cache_metrics("query_hit")
                    return cached_result["data"]

            self._update_cache_metrics("query_miss")
            return None

        except Exception as e:
            print(f"[Redis] Error retrieving cached query: {e}")
            return None

    # Embedding Similarity Caching

    def cache_embedding(self, text: str, embedding: np.ndarray, ttl: int | None = None) -> bool:
        """
        Cache text embedding for similarity search.
        
        Args:
            text: The text that was embedded
            embedding: The embedding vector
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.is_available() or self.embedding_model is None:
            return False

        try:
            cache_key = self._generate_cache_key("embedding", text.lower().strip())

            # Store embedding as numpy array
            embedding_data = {
                "embedding": embedding.tolist(),
                "text": text,
                "timestamp": time.time(),
                "model": self.model_name
            }

            serialized_data = self._serialize_data(embedding_data)
            ttl = ttl or (self.default_ttl * 24)  # Longer TTL for embeddings

            success = self.redis_client.setex(cache_key, ttl, serialized_data)

            if success:
                print(f"[Redis] Cached embedding for: {text[:50]}...")

            return bool(success)

        except Exception as e:
            print(f"[Redis] Error caching embedding: {e}")
            return False

    def get_cached_embedding(self, text: str) -> np.ndarray | None:
        """
        Retrieve cached embedding.
        
        Args:
            text: The text to find embedding for
            
        Returns:
            Cached embedding if found, None otherwise
        """
        if not self.is_available():
            return None

        try:
            cache_key = self._generate_cache_key("embedding", text.lower().strip())
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                embedding_data = self._deserialize_data(cached_data)
                if embedding_data and "embedding" in embedding_data:
                    print(f"[Redis] Embedding cache hit for: {text[:50]}...")
                    return np.array(embedding_data["embedding"])

            return None

        except Exception as e:
            print(f"[Redis] Error retrieving cached embedding: {e}")
            return None

    def find_similar_cached_embeddings(self, query_embedding: np.ndarray,
                                     similarity_threshold: float = 0.9) -> list[dict]:
        """
        Find similar cached embeddings using cosine similarity.
        
        Args:
            query_embedding: The query embedding to compare
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar cached embeddings
        """
        if not self.is_available() or self.embedding_model is None:
            return []

        try:
            # Get all embedding keys
            pattern = f"{self.cache_prefix}:embedding:*"
            keys = self.redis_client.keys(pattern)

            similar_embeddings = []

            for key in keys:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    embedding_data = self._deserialize_data(cached_data)
                    if embedding_data and "embedding" in embedding_data:
                        cached_embedding = np.array(embedding_data["embedding"])

                        # Calculate cosine similarity
                        similarity = np.dot(query_embedding, cached_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
                        )

                        if similarity >= similarity_threshold:
                            similar_embeddings.append({
                                "text": embedding_data["text"],
                                "embedding": cached_embedding,
                                "similarity": float(similarity),
                                "timestamp": embedding_data.get("timestamp", 0)
                            })

            # Sort by similarity
            similar_embeddings.sort(key=lambda x: x["similarity"], reverse=True)

            if similar_embeddings:
                print(f"[Redis] Found {len(similar_embeddings)} similar cached embeddings")

            return similar_embeddings

        except Exception as e:
            print(f"[Redis] Error finding similar embeddings: {e}")
            return []

    # Performance Metrics Caching

    def cache_performance_metrics(self, source: str, metrics: dict, ttl: int | None = None) -> bool:
        """
        Cache performance metrics for a source.
        
        Args:
            source: The source name (neo4j, semantic, keyword)
            metrics: Performance metrics data
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.is_available():
            return False

        try:
            cache_key = f"{self.metrics_prefix}:{source}"

            metrics_data = {
                "metrics": metrics,
                "timestamp": time.time(),
                "source": source
            }

            serialized_data = self._serialize_data(metrics_data)
            ttl = ttl or (self.default_ttl * 6)  # Longer TTL for metrics

            success = self.redis_client.setex(cache_key, ttl, serialized_data)

            if success:
                print(f"[Redis] Cached performance metrics for {source}")

            return bool(success)

        except Exception as e:
            print(f"[Redis] Error caching performance metrics: {e}")
            return False

    def get_cached_performance_metrics(self, source: str) -> dict | None:
        """
        Retrieve cached performance metrics.
        
        Args:
            source: The source name
            
        Returns:
            Cached metrics if found, None otherwise
        """
        if not self.is_available():
            return None

        try:
            cache_key = f"{self.metrics_prefix}:{source}"
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                metrics_data = self._deserialize_data(cached_data)
                if metrics_data and "metrics" in metrics_data:
                    return metrics_data["metrics"]

            return None

        except Exception as e:
            print(f"[Redis] Error retrieving performance metrics: {e}")
            return None

    # Session-based Caching

    def cache_session_data(self, session_id: str, data: dict, ttl: int | None = None) -> bool:
        """
        Cache session-specific data.
        
        Args:
            session_id: Unique session identifier
            data: Session data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.is_available():
            return False

        try:
            cache_key = f"{self.session_prefix}:{session_id}"

            session_data = {
                "data": data,
                "timestamp": time.time(),
                "session_id": session_id
            }

            serialized_data = self._serialize_data(session_data)
            ttl = ttl or (self.default_ttl * 2)  # Moderate TTL for sessions

            success = self.redis_client.setex(cache_key, ttl, serialized_data)

            if success:
                print(f"[Redis] Cached session data for {session_id}")

            return bool(success)

        except Exception as e:
            print(f"[Redis] Error caching session data: {e}")
            return False

    def get_session_data(self, session_id: str) -> dict | None:
        """
        Retrieve session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data if found, None otherwise
        """
        if not self.is_available():
            return None

        try:
            cache_key = f"{self.session_prefix}:{session_id}"
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                session_data = self._deserialize_data(cached_data)
                if session_data and "data" in session_data:
                    return session_data["data"]

            return None

        except Exception as e:
            print(f"[Redis] Error retrieving session data: {e}")
            return None

    # Cache Management

    def clear_cache(self, pattern: str = None) -> bool:
        """
        Clear cache entries matching pattern.
        
        Args:
            pattern: Redis key pattern (default: clear all viggo cache)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False

        try:
            if pattern is None:
                pattern = f"{self.cache_prefix}:*"

            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                print(f"[Redis] Cleared {deleted} cache entries matching {pattern}")
                return True
            else:
                print(f"[Redis] No cache entries found matching {pattern}")
                return True

        except Exception as e:
            print(f"[Redis] Error clearing cache: {e}")
            return False

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics and health information.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.is_available():
            return {"status": "unavailable", "error": "Redis not connected"}

        try:
            # Get cache key counts
            query_keys = len(self.redis_client.keys(f"{self.cache_prefix}:query:*"))
            embedding_keys = len(self.redis_client.keys(f"{self.cache_prefix}:embedding:*"))
            metrics_keys = len(self.redis_client.keys(f"{self.metrics_prefix}:*"))
            session_keys = len(self.redis_client.keys(f"{self.session_prefix}:*"))

            # Get Redis info
            info = self.redis_client.info()

            return {
                "status": "available",
                "query_cache_entries": query_keys,
                "embedding_cache_entries": embedding_keys,
                "metrics_cache_entries": metrics_keys,
                "session_cache_entries": session_keys,
                "total_memory_used": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "redis_version": info.get("redis_version", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0)
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _update_cache_metrics(self, metric_type: str):
        """Update internal cache metrics."""
        try:
            if self.is_available():
                metrics_key = f"{self.metrics_prefix}:cache_stats"
                current_stats = self.get_cached_performance_metrics("cache_stats") or {}

                current_stats[metric_type] = current_stats.get(metric_type, 0) + 1
                current_stats["last_updated"] = time.time()

                self.cache_performance_metrics("cache_stats", current_stats)
        except Exception:
            pass  # Ignore metrics update errors

    def health_check(self) -> dict:
        """
        Perform Redis health check.
        
        Returns:
            Health status dictionary
        """
        if not self.is_available():
            return {
                "status": "unhealthy",
                "error": "Redis connection not available",
                "timestamp": time.time()
            }

        try:
            # Test basic operations
            start_time = time.time()
            self.redis_client.ping()
            ping_time = (time.time() - start_time) * 1000

            # Test set/get
            test_key = f"{self.cache_prefix}:health_check:{int(time.time())}"
            test_value = "health_check"
            self.redis_client.setex(test_key, 10, test_value)
            retrieved_value = self.redis_client.get(test_key)
            self.redis_client.delete(test_key)

            success = retrieved_value and retrieved_value.decode() == test_value

            return {
                "status": "healthy" if success else "unhealthy",
                "ping_time_ms": round(ping_time, 2),
                "basic_operations": "working" if success else "failed",
                "timestamp": time.time()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
