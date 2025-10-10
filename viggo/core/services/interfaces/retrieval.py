"""
Retrieval interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RetrievalSource(Enum):
    """Types of retrieval sources."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    content: str
    score: float
    source: RetrievalSource
    metadata: dict[str, Any]
    page_number: int | None = None
    chunk_id: str | None = None


@dataclass
class QueryContext:
    """Context for a retrieval query."""
    query: str
    top_k: int = 5
    page_filter: int | None = None
    user_id: str | None = None
    session_id: str | None = None
    additional_filters: dict[str, Any] = None

    def __post_init__(self):
        if self.additional_filters is None:
            self.additional_filters = {}


class Retriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def retrieve(self, context: QueryContext) -> list[RetrievalResult]:
        """Retrieve relevant content for a query."""
        pass

    @abstractmethod
    def get_source_type(self) -> RetrievalSource:
        """Get the type of retrieval source."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the retriever is available."""
        pass


class HybridRetriever(ABC):
    """Abstract base class for hybrid retrievers."""

    @abstractmethod
    def add_retriever(self, retriever: Retriever) -> None:
        """Add a retriever to the hybrid system."""
        pass

    @abstractmethod
    def remove_retriever(self, source_type: RetrievalSource) -> None:
        """Remove a retriever from the hybrid system."""
        pass

    @abstractmethod
    def retrieve_hybrid(self, context: QueryContext) -> list[RetrievalResult]:
        """Perform hybrid retrieval across multiple sources."""
        pass

    @abstractmethod
    def get_available_sources(self) -> list[RetrievalSource]:
        """Get list of available retrieval sources."""
        pass


class ResultRanker(ABC):
    """Abstract base class for result ranking."""

    @abstractmethod
    def rank_results(self, results: list[RetrievalResult], context: QueryContext) -> list[RetrievalResult]:
        """Rank and reorder retrieval results."""
        pass

    @abstractmethod
    def get_ranking_strategy(self) -> str:
        """Get the name of the ranking strategy."""
        pass
