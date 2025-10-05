"""
Hybrid search utilities for Viggo RAG system.

This module provides utilities for:
- Hybrid search result combination and ranking
- Retrieval source tracking and metrics
- Query preprocessing for hybrid search
- Performance monitoring and optimization
"""

from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass
from enum import Enum

from viggo.models.rag_models import RetrievalSource, HybridSearchConfig, HybridSearchMetrics


@dataclass
class RetrievalResult:
    """Individual retrieval result with source tracking."""
    content: str
    page_number: int
    score: float
    source: RetrievalSource
    metadata: Dict[str, Any]
    retrieval_time: float


@dataclass
class HybridSearchResult:
    """Combined hybrid search result."""
    semantic_results: List[RetrievalResult]
    keyword_results: List[RetrievalResult]
    graph_results: List[RetrievalResult]
    combined_results: List[RetrievalResult]
    metrics: HybridSearchMetrics


class HybridSearchUtils:
    """Utilities for hybrid search operations."""
    
    @staticmethod
    def combine_search_results(
        semantic_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        config: HybridSearchConfig
    ) -> HybridSearchResult:
        """
        Combine results from different retrieval methods using weighted scoring.
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            graph_results: Results from graph search
            config: Hybrid search configuration
            
        Returns:
            Combined hybrid search result
        """
        start_time = time.time()
        
        # Normalize scores to 0-1 range for each method
        normalized_semantic = HybridSearchUtils._normalize_scores(semantic_results)
        normalized_keyword = HybridSearchUtils._normalize_scores(keyword_results)
        normalized_graph = HybridSearchUtils._normalize_scores(graph_results)
        
        # Create combined results with weighted scores
        combined_results = []
        seen_content = set()
        
        # Add semantic results with weight
        for result in normalized_semantic:
            if result.content not in seen_content:
                result.score *= config.semantic_weight
                combined_results.append(result)
                seen_content.add(result.content)
        
        # Add keyword results with weight
        for result in normalized_keyword:
            if result.content not in seen_content:
                result.score *= config.keyword_weight
                combined_results.append(result)
                seen_content.add(result.content)
            else:
                # Boost existing result with keyword score
                for existing in combined_results:
                    if existing.content == result.content:
                        existing.score += result.score * config.keyword_weight
                        break
        
        # Add graph results with weight
        for result in normalized_graph:
            if result.content not in seen_content:
                result.score *= config.graph_weight
                combined_results.append(result)
                seen_content.add(result.content)
            else:
                # Boost existing result with graph score
                for existing in combined_results:
                    if existing.content == result.content:
                        existing.score += result.score * config.graph_weight
                        break
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # Create metrics
        metrics = HybridSearchMetrics(
            semantic_results=len(semantic_results),
            keyword_results=len(keyword_results),
            graph_results=len(graph_results),
            total_candidates=len(semantic_results) + len(keyword_results) + len(graph_results),
            final_results=len(combined_results),
            semantic_time=sum(r.retrieval_time for r in semantic_results),
            keyword_time=sum(r.retrieval_time for r in keyword_results),
            graph_time=sum(r.retrieval_time for r in graph_results),
            ranking_time=time.time() - start_time
        )
        
        return HybridSearchResult(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            graph_results=graph_results,
            combined_results=combined_results,
            metrics=metrics
        )
    
    @staticmethod
    def _normalize_scores(results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Normalize scores to 0-1 range."""
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same, set to 1.0
            for result in results:
                result.score = 1.0
        else:
            # Normalize to 0-1 range
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)
        
        return results
    
    @staticmethod
    def extract_entities_from_query(query: str) -> List[str]:
        """
        Extract potential entities from a query for graph search.
        
        Args:
            query: User query text
            
        Returns:
            List of potential entity names
        """
        # Simple entity extraction - in production, use spaCy or similar
        import re
        
        # Look for capitalized words (potential proper nouns)
        entities = re.findall(r'\b[A-Z][a-z]+\b', query)
        
        # Remove common words
        common_words = {'The', 'This', 'That', 'What', 'Who', 'Where', 'When', 'Why', 'How'}
        entities = [e for e in entities if e not in common_words]
        
        return entities
    
    @staticmethod
    def apply_spoiler_protection(
        results: List[RetrievalResult],
        max_page: int
    ) -> List[RetrievalResult]:
        """
        Apply spoiler protection by filtering results beyond max_page.
        
        Args:
            results: List of retrieval results
            max_page: Maximum page number allowed
            
        Returns:
            Filtered results respecting spoiler protection
        """
        return [r for r in results if r.page_number <= max_page]
    
    @staticmethod
    def calculate_hybrid_confidence(
        semantic_score: float,
        keyword_score: float,
        graph_score: float,
        config: HybridSearchConfig
    ) -> float:
        """
        Calculate overall confidence score from individual method scores.
        
        Args:
            semantic_score: Semantic search confidence
            keyword_score: Keyword search confidence
            graph_score: Graph search confidence
            config: Hybrid search configuration
            
        Returns:
            Weighted confidence score
        """
        return (
            semantic_score * config.semantic_weight +
            keyword_score * config.keyword_weight +
            graph_score * config.graph_weight
        )


class QueryPreprocessor:
    """Utilities for preprocessing queries for hybrid search."""
    
    @staticmethod
    def preprocess_query(query: str) -> Dict[str, Any]:
        """
        Preprocess query for hybrid search.
        
        Args:
            query: Raw user query
            
        Returns:
            Preprocessed query information
        """
        return {
            "original_query": query,
            "normalized_query": query.lower().strip(),
            "extracted_entities": HybridSearchUtils.extract_entities_from_query(query),
            "query_type": QueryPreprocessor._classify_query_type(query),
            "keywords": QueryPreprocessor._extract_keywords(query)
        }
    
    @staticmethod
    def _classify_query_type(query: str) -> str:
        """Classify the type of query (character, plot, setting, etc.)."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['who', 'character', 'person', 'protagonist']):
            return "character"
        elif any(word in query_lower for word in ['what', 'happens', 'plot', 'story']):
            return "plot"
        elif any(word in query_lower for word in ['where', 'location', 'place', 'setting']):
            return "setting"
        elif any(word in query_lower for word in ['when', 'time', 'period', 'era']):
            return "temporal"
        else:
            return "general"
    
    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        """Extract important keywords from query."""
        import re
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords


class PerformanceMonitor:
    """Utilities for monitoring hybrid search performance."""
    
    def __init__(self):
        self.metrics_history: List[HybridSearchMetrics] = []
    
    def record_search_metrics(self, metrics: HybridSearchMetrics):
        """Record search metrics for analysis."""
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from recent metrics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 searches
        
        return {
            "avg_semantic_time": sum(m.semantic_time for m in recent_metrics) / len(recent_metrics),
            "avg_keyword_time": sum(m.keyword_time for m in recent_metrics) / len(recent_metrics),
            "avg_graph_time": sum(m.graph_time for m in recent_metrics) / len(recent_metrics),
            "avg_ranking_time": sum(m.ranking_time for m in recent_metrics) / len(recent_metrics),
            "avg_total_results": sum(m.final_results for m in recent_metrics) / len(recent_metrics),
            "semantic_success_rate": sum(1 for m in recent_metrics if m.semantic_results > 0) / len(recent_metrics),
            "keyword_success_rate": sum(1 for m in recent_metrics if m.keyword_results > 0) / len(recent_metrics),
            "graph_success_rate": sum(1 for m in recent_metrics if m.graph_results > 0) / len(recent_metrics)
        }
