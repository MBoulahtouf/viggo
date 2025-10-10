"""
Multi-agent framework interfaces following SOLID principles.
Simplified design focused on enhancing RAG capabilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class AgentType(Enum):
    """Types of agents in the multi-agent system."""
    QUERY_ANALYZER = "query_analyzer"
    ENTITY_EXTRACTOR = "entity_extractor"
    RELATIONSHIP_EXTRACTOR = "relationship_extractor"
    CONTEXT_AGGREGATOR = "context_aggregator"
    RESPONSE_GENERATOR = "response_generator"


@dataclass
class AgentResult:
    """Result from an agent operation."""
    agent_type: AgentType
    success: bool
    data: dict[str, Any]
    confidence: float
    processing_time: float
    error_message: str | None = None


@dataclass
class QueryAnalysis:
    """Analysis result from query analyzer."""
    intent: str  # "character", "plot", "setting", "relationship", "general"
    entities: list[str]
    complexity: float  # 0-1 scale
    requires_graph: bool
    requires_semantic: bool


@dataclass
class EntityExtraction:
    """Result from entity extraction."""
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    confidence: float


@dataclass
class ContextAggregation:
    """Result from context aggregation."""
    semantic_results: list[dict[str, Any]]
    graph_results: list[dict[str, Any]]
    hybrid_score: float
    source_attribution: list[dict[str, Any]]


class IAgent(ABC):
    """Base interface for all agents."""

    @abstractmethod
    def get_agent_type(self) -> AgentType:
        """Get agent type."""
        pass

    @abstractmethod
    def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Process input data and return result."""
        pass

    @abstractmethod
    def can_handle(self, input_data: dict[str, Any]) -> bool:
        """Check if agent can handle the input data."""
        pass


class IQueryAnalyzer(IAgent):
    """Interface for query analysis agent."""

    @abstractmethod
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine intent and requirements."""
        pass


class IEntityExtractor(IAgent):
    """Interface for entity extraction agent."""

    @abstractmethod
    def extract_entities(self, content: str, context: dict[str, Any] | None = None) -> EntityExtraction:
        """Extract entities and relationships from content."""
        pass


class IContextAggregator(IAgent):
    """Interface for context aggregation agent."""

    @abstractmethod
    def aggregate_context(self, query: str, semantic_results: list[dict[str, Any]],
                         graph_results: list[dict[str, Any]]) -> ContextAggregation:
        """Aggregate semantic and graph results into unified context."""
        pass


class IResponseGenerator(IAgent):
    """Interface for response generation agent."""

    @abstractmethod
    def generate_response(self, query: str, context: ContextAggregation,
                         analysis: QueryAnalysis) -> str:
        """Generate response based on query, context, and analysis."""
        pass


class IMultiAgentOrchestrator(ABC):
    """Interface for orchestrating multiple agents."""

    @abstractmethod
    def register_agent(self, agent: IAgent) -> bool:
        """Register an agent with the orchestrator."""
        pass

    @abstractmethod
    def process_query(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Process query using multi-agent system."""
        pass

    @abstractmethod
    def get_agent_status(self) -> dict[str, Any]:
        """Get status of all registered agents."""
        pass
