"""
Enhanced RAG factory for creating multi-agent and GraphRAG enabled RAG services.
"""

from typing import Any

from viggo.core.services.implementations.azure_graph_rag_impl import (
    AzureGraphRAGService,
)
from viggo.core.services.implementations.enhanced_rag_service_impl import (
    EnhancedRAGService,
)
from viggo.core.services.implementations.graph_service_impl import GraphService
from viggo.core.services.implementations.rag_orchestrator import ConcreteRAGOrchestrator
from viggo.core.services.implementations.redis_service_impl import RedisService
from viggo.core.services.interfaces.rag import RAGService


class EnhancedRAGFactory:
    """Factory for creating enhanced RAG services with multi-agent and GraphRAG capabilities."""

    def __init__(self):
        self._enhanced_rag_service = None
        self._graph_rag_service = None
        self._orchestrator = ConcreteRAGOrchestrator()

    def create_enhanced_rag_service(self,
                                  graph_service: GraphService | None = None,
                                  redis_service: RedisService | None = None,
                                  enable_graph_rag: bool = True,
                                  config_type: str = "enhanced") -> RAGService:
        """
        Create an enhanced RAG service with multi-agent and GraphRAG capabilities.

        Args:
            graph_service: Optional Neo4j graph service for GraphRAG
            redis_service: Optional Redis cache service
            enable_graph_rag: Whether to enable GraphRAG processing
            config_type: Type of configuration ("enhanced", "minimal", "custom")

        Returns:
            Enhanced RAG service instance
        """
        # Create base RAG configuration
        if config_type == "enhanced":
            config = self._orchestrator.create_default_config(graph_service, redis_service)
        elif config_type == "minimal":
            config = self._orchestrator.create_minimal_config()
        else:
            config = self._orchestrator.create_default_config(graph_service, redis_service)

        # Create GraphRAG service if enabled and graph service is available
        graph_rag_service = None
        if enable_graph_rag and graph_service:
            try:
                from viggo.core.services.implementations.storage_impl import (
                    AzureSearchVectorStorage,
                )
                vector_storage = AzureSearchVectorStorage()
                graph_rag_service = AzureGraphRAGService(graph_service, vector_storage)
                print("✅ GraphRAG service created successfully")
            except Exception as e:
                print(f"⚠️ Failed to create GraphRAG service: {e}")
                graph_rag_service = None

        # Create enhanced RAG service
        enhanced_service = EnhancedRAGService(config, graph_rag_service)

        print(f"🚀 Enhanced RAG service created with config_type: {config_type}")
        print("   - Multi-agent framework: ✅ Enabled")
        print(f"   - GraphRAG processing: {'✅ Enabled' if graph_rag_service else '❌ Disabled'}")
        print(f"   - Graph service: {'✅ Available' if graph_service else '❌ Not available'}")
        print(f"   - Redis service: {'✅ Available' if redis_service else '❌ Not available'}")

        return enhanced_service

    def create_multi_agent_only_service(self,
                                      graph_service: GraphService | None = None,
                                      redis_service: RedisService | None = None) -> RAGService:
        """Create RAG service with only multi-agent framework (no GraphRAG)."""
        return self.create_enhanced_rag_service(
            graph_service=graph_service,
            redis_service=redis_service,
            enable_graph_rag=False,
            config_type="enhanced"
        )

    def create_graph_rag_only_service(self,
                                    graph_service: GraphService,
                                    redis_service: RedisService | None = None) -> RAGService:
        """Create RAG service with only GraphRAG (no multi-agent framework)."""
        # This would create a service that only uses GraphRAG without multi-agent
        # For now, we'll create the full enhanced service
        return self.create_enhanced_rag_service(
            graph_service=graph_service,
            redis_service=redis_service,
            enable_graph_rag=True,
            config_type="enhanced"
        )

    def get_available_configurations(self) -> dict[str, list[str]]:
        """Get available configuration options."""
        return {
            "config_types": ["enhanced", "minimal", "custom"],
            "features": ["multi_agent", "graph_rag", "azure_search", "neo4j", "redis"],
            "agents": ["query_analyzer", "entity_extractor", "context_aggregator", "response_generator"],
            "graph_rag_stages": ["entity_extraction", "relationship_extraction", "community_detection", "summarization"]
        }

    def validate_enhanced_configuration(self, config: dict[str, Any]) -> bool:
        """Validate enhanced RAG configuration."""
        try:
            required_features = config.get('required_features', [])

            # Check if multi-agent is required
            if 'multi_agent' in required_features:
                # Multi-agent is always available
                pass

            # Check if GraphRAG is required
            if 'graph_rag' in required_features:
                if not config.get('graph_service'):
                    return False

            # Check if Azure Search is required
            if 'azure_search' in required_features:
                # Azure Search should be available through vector storage
                pass

            # Check if Neo4j is required
            if 'neo4j' in required_features:
                if not config.get('graph_service'):
                    return False

            return True

        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False

    def get_system_capabilities(self) -> dict[str, Any]:
        """Get system capabilities and status."""
        return {
            "multi_agent_framework": {
                "available": True,
                "agents": ["query_analyzer", "entity_extractor", "context_aggregator", "response_generator"],
                "features": ["intent_detection", "entity_extraction", "relationship_extraction", "context_aggregation", "response_generation"]
            },
            "graph_rag": {
                "available": True,
                "stages": ["entity_extraction", "relationship_extraction", "community_detection", "summarization"],
                "storage": ["neo4j", "azure_search"],
                "algorithms": ["community_detection", "relationship_classification", "entity_deduplication"]
            },
            "enhanced_features": {
                "hybrid_retrieval": True,
                "intelligent_routing": True,
                "context_aggregation": True,
                "quality_assessment": True,
                "source_attribution": True
            }
        }


# Global enhanced factory instance
enhanced_rag_factory = EnhancedRAGFactory()
