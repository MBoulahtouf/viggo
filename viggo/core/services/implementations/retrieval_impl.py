"""
Concrete implementations of retrieval services following SOLID principles.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from viggo.core.services.interfaces.retrieval import (
    HybridRetriever,
    QueryContext,
    ResultRanker,
    RetrievalResult,
    RetrievalSource,
    Retriever,
)

from .graph_service_impl import GraphService
from .hybrid_retriever_impl import HybridRetriever
from .hybrid_search_service_impl import HybridSearchService


class SemanticRetriever(Retriever):
    """Concrete implementation of semantic retriever using Azure Cognitive Search."""

    def __init__(self, vector_storage, model_name: str = "all-MiniLM-L6-v2"):
        self.vector_storage = vector_storage
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def retrieve(self, context: QueryContext) -> list[RetrievalResult]:
        """Retrieve relevant content using semantic similarity."""
        if not self.is_available():
            return []

        try:
            # Generate query embedding
            model = self._get_model()
            query_embedding = model.encode([context.query])[0].tolist()

            # Search Azure Search index
            search_results = self.vector_storage.search_vectors(query_embedding, context.top_k)

            results = []
            for result in search_results:
                metadata = result.get("metadata", {})
                retrieval_result = RetrievalResult(
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    source=RetrievalSource.SEMANTIC,
                    metadata={
                        "chunk_id": metadata.get("chunk_id", ""),
                        "entities": metadata.get("entities", []),
                        "chapter_title": metadata.get("chapter_title", ""),
                        "lore_significance": metadata.get("lore_significance", 0.0)
                    },
                    page_number=metadata.get("page", 0),
                    chunk_id=metadata.get("chunk_id", "")
                )
                results.append(retrieval_result)

            return results

        except Exception as e:
            print(f"Semantic retrieval error: {e}")
            return []

    def get_source_type(self) -> RetrievalSource:
        """Get the type of retrieval source."""
        return RetrievalSource.SEMANTIC

    def is_available(self) -> bool:
        """Check if the retriever is available."""
        return self.vector_storage is not None and self.vector_storage.get_vector_count() > 0


class KeywordRetriever(Retriever):
    """Concrete implementation of keyword retriever using Azure Cognitive Search."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.hybrid_search_service = HybridSearchService(model_name)

    def retrieve(self, context: QueryContext) -> list[RetrievalResult]:
        """Retrieve relevant content using keyword search."""
        if not self.is_available():
            return []

        try:
            # Use hybrid search service for keyword search
            search_results = self.hybrid_search_service.keyword_search(
                context.query,
                context.top_k,
                context.page_filter
            )

            results = []
            for result in search_results:
                retrieval_result = RetrievalResult(
                    content=result["content"],
                    score=result.get("score", 0.0),
                    source=RetrievalSource.KEYWORD,
                    metadata={
                        "entities": result.get("entities", []),
                        "entity_labels": result.get("entity_labels", []),
                        "chapter_title": result.get("chapter_title", ""),
                        "chunk_type": result.get("chunk_type", "standard")
                    },
                    page_number=result.get("page", 0)
                )
                results.append(retrieval_result)

            return results

        except Exception as e:
            print(f"Keyword retrieval error: {e}")
            return []

    def get_source_type(self) -> RetrievalSource:
        """Get the type of retrieval source."""
        return RetrievalSource.KEYWORD

    def is_available(self) -> bool:
        """Check if the retriever is available."""
        return self.hybrid_search_service.search_client is not None


class GraphRetriever(Retriever):
    """Concrete implementation of graph retriever using Neo4j."""

    def __init__(self, graph_service: GraphService):
        self.graph_service = graph_service

    def retrieve(self, context: QueryContext) -> list[RetrievalResult]:
        """Retrieve relevant content using graph queries."""
        if not self.is_available():
            return []

        try:
            # Extract entities from query for targeted lookups
            entities = self._extract_entities_for_graph(context.query)

            results = []

            # Query for each entity type
            for entity_type, entity_names in entities.items():
                for entity_name in entity_names:
                    # Get entity details and relationships
                    entity_data = self.graph_service.get_entity_details(entity_name, entity_type)
                    if entity_data:
                        result = RetrievalResult(
                            content=f"Entity: {entity_name} ({entity_type})",
                            score=1.0,  # High confidence for structured data
                            source=RetrievalSource.GRAPH,
                            metadata={
                                "entity_data": entity_data,
                                "entity_type": entity_type,
                                "chunk_type": "structured_fact"
                            },
                            page_number=0  # Neo4j data doesn't have pages
                        )
                        results.append(result)

            # Query for relationships
            relationships = self.graph_service.find_relationships(context.query)
            for rel in relationships:
                result = RetrievalResult(
                    content=f"Relationship: {rel.get('description', 'Unknown relationship')}",
                    score=1.0,
                    source=RetrievalSource.GRAPH,
                    metadata={
                        "relationship_data": rel,
                        "chunk_type": "relationship"
                    },
                    page_number=0
                )
                results.append(result)

            return results

        except Exception as e:
            print(f"Graph retrieval error: {e}")
            return []

    def _extract_entities_for_graph(self, query: str) -> dict[str, list[str]]:
        """Extract entities from query for targeted graph lookups."""
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

    def get_source_type(self) -> RetrievalSource:
        """Get the type of retrieval source."""
        return RetrievalSource.GRAPH

    def is_available(self) -> bool:
        """Check if the retriever is available."""
        return self.graph_service is not None


class WeightedResultRanker(ResultRanker):
    """Concrete implementation of weighted result ranking."""

    def __init__(self, weights: dict[RetrievalSource, float] | None = None):
        self.weights = weights or {
            RetrievalSource.GRAPH: 1.0,      # Highest weight - authoritative facts
            RetrievalSource.SEMANTIC: 0.7,   # Medium weight - contextual understanding
            RetrievalSource.KEYWORD: 0.4     # Lower weight - precision matches
        }

    def rank_results(self, results: list[RetrievalResult], context: QueryContext) -> list[RetrievalResult]:
        """Rank and reorder retrieval results."""
        if not results:
            return results

        # Calculate weighted scores
        for result in results:
            base_score = result.score
            weight = self.weights.get(result.source, 0.5)
            result.metadata["weighted_score"] = base_score * weight

        # Sort by weighted score
        ranked_results = sorted(results, key=lambda x: x.metadata.get("weighted_score", 0.0), reverse=True)

        # Apply evidence alignment - boost results that appear in multiple sources
        self._apply_evidence_alignment(ranked_results)

        return ranked_results

    def _apply_evidence_alignment(self, results: list[RetrievalResult]):
        """Apply evidence alignment to boost results that appear in multiple sources."""
        # Group results by content similarity
        content_groups = {}
        for result in results:
            content_key = result.content[:100]  # Use first 100 chars as key
            if content_key not in content_groups:
                content_groups[content_key] = []
            content_groups[content_key].append(result)

        # Boost scores for results that appear in multiple sources
        for group in content_groups.values():
            if len(group) > 1:  # Multiple sources agree
                boost_factor = 1.2  # 20% boost for agreement
                for result in group:
                    current_score = result.metadata.get("weighted_score", 0.0)
                    result.metadata["weighted_score"] = current_score * boost_factor
                    result.metadata["evidence_alignment"] = len(group)

    def get_ranking_strategy(self) -> str:
        """Get the name of the ranking strategy."""
        return "weighted_ranking"


class ConcreteHybridRetriever(HybridRetriever):
    """Concrete implementation of hybrid retriever."""

    def __init__(self):
        self.retrievers: dict[RetrievalSource, Retriever] = {}
        self.ranker = WeightedResultRanker()
        self.retrieval_times = {}
        self.source_usage_stats = {}

    def add_retriever(self, retriever: Retriever) -> None:
        """Add a retriever to the hybrid system."""
        source_type = retriever.get_source_type()
        self.retrievers[source_type] = retriever
        self.source_usage_stats[source_type] = 0

    def remove_retriever(self, source_type: RetrievalSource) -> None:
        """Remove a retriever from the hybrid system."""
        if source_type in self.retrievers:
            del self.retrievers[source_type]
            if source_type in self.source_usage_stats:
                del self.source_usage_stats[source_type]

    def retrieve_hybrid(self, context: QueryContext) -> list[RetrievalResult]:
        """Perform hybrid retrieval across multiple sources."""
        all_results = []

        # Check if we have any retrievers
        if not self.retrievers:
            print("Warning: No retrievers available for hybrid retrieval")
            return []

        # Run all retrievers in parallel
        with ThreadPoolExecutor(max_workers=max(1, len(self.retrievers))) as executor:
            futures = {}
            for source_type, retriever in self.retrievers.items():
                if retriever.is_available():
                    futures[source_type] = executor.submit(self._retrieve_with_timing, retriever, context)

            # Collect results as they complete
            for source_type, future in futures.items():
                try:
                    start_time = time.time()
                    results = future.result(timeout=10.0)  # 10 second timeout
                    response_time = time.time() - start_time

                    self.retrieval_times[source_type] = response_time
                    self.source_usage_stats[source_type] += 1

                    all_results.extend(results)

                except Exception as e:
                    print(f"Retrieval failed for {source_type}: {e}")

        # Rank and return results
        ranked_results = self.ranker.rank_results(all_results, context)
        return ranked_results[:context.top_k]

    def _retrieve_with_timing(self, retriever: Retriever, context: QueryContext) -> list[RetrievalResult]:
        """Retrieve with timing information."""
        return retriever.retrieve(context)

    def get_available_sources(self) -> list[RetrievalSource]:
        """Get list of available retrieval sources."""
        return [source for source, retriever in self.retrievers.items() if retriever.is_available()]

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            "retrieval_times": self.retrieval_times,
            "source_usage": self.source_usage_stats,
            "total_queries": sum(self.source_usage_stats.values())
        }
