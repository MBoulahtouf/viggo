"""
Concrete implementation of hybrid retriever following SOLID principles.
"""

import asyncio
import json
import time
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from viggo.core.config import settings
from viggo.core.services.interfaces.hybrid_retriever import IHybridRetriever
from viggo.core.services.interfaces.hybrid_search_service import IHybridSearchService
from viggo.core.services.interfaces.graph_service import IGraphService
from viggo.core.services.interfaces.performance_optimizer import IPerformanceOptimizer
from viggo.core.services.interfaces.redis_service import IRedisService
from .hybrid_search_service_impl import HybridSearchService
from .graph_service_impl import GraphService
from .performance_optimizer_impl import PerformanceOptimizer
from .redis_service_impl import RedisService


class HybridRetriever(IHybridRetriever):
    """
    Implements true hybrid RAG with parallel retrieval from:
    1. Semantic Search (Azure Search) - contextual understanding
    2. Neo4j Structured Lookups - authoritative facts
    3. Azure Cognitive Search - keyword precision
    """
    
    def __init__(self, vector_storage=None, model_name="all-MiniLM-L6-v2"):
        self.vector_storage = vector_storage
        self.model_name = model_name
        
        # Initialize services
        self.hybrid_search_service = HybridSearchService(model_name)
        self.graph_service = None  # Will be set by RAG service if available
        
        # Initialize Redis cache service first
        self.redis_cache = RedisService(model_name)
        
        # Initialize performance optimizer with Redis cache service
        self.performance_optimizer = PerformanceOptimizer(redis_cache_service=self.redis_cache)
        
        # Weights for fusion ranking (Neo4j > Semantic > Keyword)
        self.weights = {
            "neo4j": 1.0,      # Highest weight - authoritative facts
            "semantic": 0.7,   # Medium weight - contextual understanding  
            "keyword": 0.4     # Lower weight - precision matches
        }
        
        # Performance tracking
        self.retrieval_times = {}
        self.source_usage_stats = {"neo4j": 0, "semantic": 0, "keyword": 0}
        
        # Cache configuration
        self.cache_enabled = self.redis_cache.is_available()
        if self.cache_enabled:
            print("[HybridRetriever] Redis cache enabled")
        else:
            print("[HybridRetriever] Redis cache disabled - using in-memory cache only")
    
    async def retrieve(self, query: str, top_k: int = 5, page_filter: Optional[int] = None) -> Dict:
        """
        Perform parallel hybrid retrieval from all sources with performance optimization.
        
        Args:
            query: User query
            top_k: Number of results to return
            page_filter: Optional page number filter
            
        Returns:
            Combined and ranked results with metadata
        """
        start_time = time.time()
        self.performance_optimizer.total_queries += 1
        
        # Check Redis cache first (primary cache)
        cached_result = None
        if self.cache_enabled:
            cached_result = self.redis_cache.get_cached_query_result(query, top_k, page_filter)
            if cached_result:
                print(f"[REDIS CACHE HIT] Query result found in Redis cache")
                return cached_result
        
        # Check in-memory cache as fallback
        cached_result = self.performance_optimizer.get_cached_query_result(query, top_k, page_filter)
        if cached_result:
            print(f"[MEMORY CACHE HIT] Query result found in memory cache")
            return cached_result
        
        # Run all retrieval methods in parallel with adaptive timeouts
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all retrieval tasks with adaptive timeouts
            futures = {}
            for source in ["semantic", "neo4j", "keyword"]:
                timeout = self.performance_optimizer.get_source_timeout(source)
                if source == "semantic":
                    futures[source] = executor.submit(self._semantic_search, query, top_k, page_filter)
                elif source == "neo4j":
                    futures[source] = executor.submit(self._neo4j_lookup, query)
                elif source == "keyword":
                    futures[source] = executor.submit(self._keyword_search, query, top_k, page_filter)
            
            # Collect results as they complete with adaptive timeouts
            results = {}
            for source, future in futures.items():
                try:
                    start_source = time.time()
                    timeout = self.performance_optimizer.get_source_timeout(source)
                    results[source] = future.result(timeout=timeout)
                    response_time = time.time() - start_source
                    self.retrieval_times[source] = response_time
                    self.source_usage_stats[source] += 1
                    
                    # Update performance metrics
                    self.performance_optimizer.update_source_performance(source, response_time, success=True)
                    
                except Exception as e:
                    response_time = time.time() - start_source
                    print(f"[WARNING] {source} retrieval failed after {response_time:.2f}s: {e}")
                    results[source] = []
                    
                    # Update performance metrics for failure
                    self.performance_optimizer.update_source_performance(source, response_time, success=False)
        
        # Combine and rank results
        combined_results = self._combine_and_rank(
            results["semantic"], 
            results["neo4j"], 
            results["keyword"]
        )
        
        total_time = time.time() - start_time
        
        result = {
            "results": combined_results[:top_k],
            "metadata": {
                "total_time": total_time,
                "retrieval_times": self.retrieval_times,
                "sources_used": [k for k, v in results.items() if v],
                "source_stats": self.source_usage_stats,
                "query": query,
                "cache_hit": False
            }
        }
        
        # Cache the result for future queries
        if self.cache_enabled:
            # Cache in Redis (primary cache)
            self.redis_cache.cache_query_result(query, top_k, page_filter, result)
        
        # Also cache in memory as backup
        self.performance_optimizer.cache_query_result(query, top_k, page_filter, result)
        
        return result
    
    def _semantic_search(self, query: str, top_k: int, page_filter: Optional[int] = None) -> List[Dict]:
        """
        Perform semantic search using Azure Search vector similarity with Redis embedding caching.
        """
        try:
            if not self.vector_storage:
                return []
            
            # Generate query embedding with Redis caching
            query_embedding = None
            if self.cache_enabled:
                query_embedding = self.redis_cache.get_cached_embedding(query)
            
            if query_embedding is None:
                # Generate new embedding
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.model_name)
                query_embedding = self.performance_optimizer.get_embedding(query, model)
                
                # Cache the embedding in Redis
                if self.cache_enabled:
                    self.redis_cache.cache_embedding(query, query_embedding)
                    print(f"[Redis] Cached embedding for query: {query[:50]}...")
            
            # Ensure query_embedding is a list for Azure Search
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            elif isinstance(query_embedding, list) and len(query_embedding) > 0 and hasattr(query_embedding[0], 'tolist'):
                query_embedding = query_embedding[0].tolist()
            
            # Search Azure Search index
            search_results = self.vector_storage.search_vectors(query_embedding, top_k)
            
            # Filter by page if specified
            if page_filter is not None:
                search_results = [r for r in search_results if r.get("metadata", {}).get("page", 0) <= page_filter]
            
            # Get relevant chunks
            results = []
            for result in search_results:
                metadata = result.get("metadata", {})
                results.append({
                    "content": result.get("content", ""),
                    "page": metadata.get("page", 0),
                    "score": result.get("score", 0.0),
                    "source": "semantic",
                    "weight": self.weights["semantic"],
                    "entities": metadata.get("entities", []),
                    "entity_labels": metadata.get("entity_labels", []),
                    "chapter_title": metadata.get("chapter_title", ""),
                    "chunk_type": metadata.get("chunk_type", "standard"),
                    "document_metadata": metadata.get("document_metadata", {})
                })
            
            return results
            
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    def _neo4j_lookup(self, query: str) -> List[Dict]:
        """
        Perform structured lookup in Neo4j knowledge graph.
        """
        if not self.graph_service:
            print("Neo4j service not available, skipping Neo4j lookup")
            return []
            
        try:
            # Extract entities from query for targeted lookups
            entities = self._extract_entities_for_neo4j(query)
            
            results = []
            
            # Query for each entity type
            for entity_type, entity_names in entities.items():
                for entity_name in entity_names:
                    # Get entity details and relationships
                    entity_data = self.graph_service.get_entity_details(entity_name, entity_type)
                    if entity_data:
                        results.append({
                            "content": f"Entity: {entity_name} ({entity_type})",
                            "page": 0,  # Neo4j data doesn't have pages
                            "score": 1.0,  # High confidence for structured data
                            "source": "neo4j",
                            "weight": self.weights["neo4j"],
                            "entities": [entity_name],
                            "entity_labels": [entity_type],
                            "neo4j_data": entity_data,
                            "chunk_type": "structured_fact"
                        })
            
            # Query for relationships
            relationships = self.graph_service.find_relationships(query)
            for rel in relationships:
                results.append({
                    "content": f"Relationship: {rel.get('description', 'Unknown relationship')}",
                    "page": 0,
                    "score": 1.0,
                    "source": "neo4j", 
                    "weight": self.weights["neo4j"],
                    "entities": [rel.get("source", ""), rel.get("target", "")],
                    "entity_labels": ["relationship"],
                    "neo4j_data": rel,
                    "chunk_type": "relationship"
                })
            
            return results
            
        except Exception as e:
            print(f"Neo4j lookup error: {e}")
            return []
    
    def _keyword_search(self, query: str, top_k: int, page_filter: Optional[int] = None) -> List[Dict]:
        """
        Perform keyword search using Azure Cognitive Search.
        """
        try:
            # Use the hybrid search service for keyword search
            results = self.hybrid_search_service.keyword_search(query, top_k, page_filter)
            
            # Convert to standard format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result["content"],
                    "page": result.get("page", 0),
                    "score": result.get("score", 0.0),
                    "source": "keyword",
                    "weight": self.weights["keyword"],
                    "entities": result.get("entities", []),
                    "entity_labels": result.get("entity_labels", []),
                    "chapter_title": result.get("chapter_title", ""),
                    "chunk_type": result.get("chunk_type", "standard"),
                    "document_metadata": result.get("document_metadata", {})
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []
    
    def _extract_entities_for_neo4j(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from query for targeted Neo4j lookups.
        """
        # Simple entity extraction - in production, use more sophisticated NLP
        entities = {
            "Character": [],
            "Location": [], 
            "Event": [],
            "Organization": []
        }
        
        # Common fantasy/Lovecraft terms to look for
        fantasy_terms = {
            "Character": ["Olney", "Lovecraft", "Kingsport", "Arkham", "Miskatonic"],
            "Location": ["Kingsport", "Arkham", "Miskatonic", "New England"],
            "Event": ["Battle", "Siege", "War", "Mist", "House"],
            "Organization": ["Elder Ones", "Congregational", "Neptune", "Nodens"]
        }
        
        query_lower = query.lower()
        for entity_type, terms in fantasy_terms.items():
            for term in terms:
                if term.lower() in query_lower:
                    entities[entity_type].append(term)
        
        return entities
    
    def _combine_and_rank(self, semantic_results: List[Dict], neo4j_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """
        Combine and rank results from all sources using weighted scoring.
        """
        all_results = semantic_results + neo4j_results + keyword_results
        
        # Calculate weighted scores
        for result in all_results:
            base_score = result.get("score", 0.0)
            weight = result.get("weight", 0.5)
            result["weighted_score"] = base_score * weight
        
        # Sort by weighted score
        ranked_results = sorted(all_results, key=lambda x: x["weighted_score"], reverse=True)
        
        # Evidence alignment - boost results that appear in multiple sources
        self._apply_evidence_alignment(ranked_results)
        
        return ranked_results
    
    def _apply_evidence_alignment(self, results: List[Dict]):
        """
        Apply evidence alignment to boost results that appear in multiple sources.
        """
        # Group results by content similarity
        content_groups = {}
        for result in results:
            content_key = result["content"][:100]  # Use first 100 chars as key
            if content_key not in content_groups:
                content_groups[content_key] = []
            content_groups[content_key].append(result)
        
        # Boost scores for results that appear in multiple sources
        for group in content_groups.values():
            if len(group) > 1:  # Multiple sources agree
                boost_factor = 1.2  # 20% boost for agreement
                for result in group:
                    result["weighted_score"] *= boost_factor
                    result["evidence_alignment"] = len(group)
    
    def get_performance_stats(self) -> Dict:
        """
        Get comprehensive performance statistics for monitoring.
        """
        base_stats = {
            "retrieval_times": self.retrieval_times,
            "source_usage": self.source_usage_stats,
            "total_queries": sum(self.source_usage_stats.values())
        }
        
        # Add performance optimizer stats
        optimizer_stats = self.performance_optimizer.get_performance_stats()
        
        # Add Redis cache stats
        cache_stats = {}
        if self.cache_enabled:
            cache_stats = {
                "redis_cache": self.redis_cache.get_cache_stats(),
                "redis_health": self.redis_cache.health_check()
            }
        else:
            cache_stats = {
                "redis_cache": {"status": "disabled"},
                "redis_health": {"status": "disabled"}
            }
        
        return {
            **base_stats,
            "optimization": optimizer_stats,
            "cache": cache_stats
        }
    
    def create_hybrid_prompt(self, query: str, results: List[Dict]) -> str:
        """
        Create hybrid prompt template with structured data, lore context, and exact matches.
        """
        # Separate results by source
        neo4j_data = [r for r in results if r["source"] == "neo4j"]
        semantic_data = [r for r in results if r["source"] == "semantic"] 
        keyword_data = [r for r in results if r["source"] == "keyword"]
        
        prompt = f"""You are Viggo, a lore expert. Answer the following question using the provided context from multiple sources:

Question: {query}

Context from different sources:

1. Structured Data (Neo4j - Authoritative Facts):
{self._format_neo4j_context(neo4j_data)}

2. Lore Context (Semantic Search - Narrative Understanding):
{self._format_semantic_context(semantic_data)}

3. Exact Matches (Keyword Search - Precision):
{self._format_keyword_context(keyword_data)}

Instructions:
- Prioritize Neo4j data for authoritative facts (dates, relationships, names)
- Use semantic search for narrative context and story understanding
- Include keyword matches for precision and exact references
- If sources conflict, prioritize Neo4j structured data
- Provide citations indicating the source of information
- Synthesize a cohesive, lore-consistent answer

Answer:"""
        
        return prompt
    
    def _format_neo4j_context(self, neo4j_results: List[Dict]) -> str:
        """Format Neo4j results for prompt."""
        if not neo4j_results:
            return "No structured data found."
        
        context = ""
        for result in neo4j_results:
            context += f"- {result['content']}\n"
            if "neo4j_data" in result:
                context += f"  Details: {json.dumps(result['neo4j_data'], indent=2)}\n"
        
        return context
    
    def _format_semantic_context(self, semantic_results: List[Dict]) -> str:
        """Format semantic search results for prompt."""
        if not semantic_results:
            return "No contextual lore found."
        
        context = ""
        for result in semantic_results:
            context += f"- Page {result.get('page', 'N/A')}: {result['content'][:200]}...\n"
        
        return context
    
    def _format_keyword_context(self, keyword_results: List[Dict]) -> str:
        """Format keyword search results for prompt."""
        if not keyword_results:
            return "No exact matches found."
        
        context = ""
        for result in keyword_results:
            context += f"- Page {result.get('page', 'N/A')}: {result['content'][:200]}...\n"
        
        return context
    
    def clear_cache(self, cache_type: str = "all") -> bool:
        """
        Clear cache entries.
        
        Args:
            cache_type: Type of cache to clear ("all", "query", "embedding", "memory")
            
        Returns:
            True if successful, False otherwise
        """
        success = True
        
        if cache_type in ["all", "query", "embedding"] and self.cache_enabled:
            if cache_type == "all":
                success = self.redis_cache.clear_cache()
            elif cache_type == "query":
                success = self.redis_cache.clear_cache(f"{self.redis_cache.cache_prefix}:query:*")
            elif cache_type == "embedding":
                success = self.redis_cache.clear_cache(f"{self.redis_cache.cache_prefix}:embedding:*")
        
        if cache_type in ["all", "memory"]:
            # Clear in-memory cache
            self.performance_optimizer.clear_cache()
        
        if success:
            print(f"[HybridRetriever] Successfully cleared {cache_type} cache")
        
        return success
    
    def get_cache_info(self) -> Dict:
        """
        Get detailed cache information.
        
        Returns:
            Dictionary with cache information
        """
        cache_info = {
            "cache_enabled": self.cache_enabled,
            "cache_type": "redis" if self.cache_enabled else "memory_only"
        }
        
        if self.cache_enabled:
            cache_info.update({
                "redis_stats": self.redis_cache.get_cache_stats(),
                "redis_health": self.redis_cache.health_check()
            })
        
        # Add memory cache info
        cache_info["memory_cache"] = {
            "performance_optimizer": "available",
            "cache_size": getattr(self.performance_optimizer, 'cache_size', 0)
        }
        
        return cache_info
