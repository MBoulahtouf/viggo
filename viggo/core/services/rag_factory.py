"""
RAG Factory Service - Provides easy access to the new SOLID-compliant RAG architecture.
"""

from typing import Optional, Dict, Any
from viggo.core.services.interfaces.rag import RAGService, RAGConfig
from viggo.core.services.implementations.rag_orchestrator import ConcreteRAGOrchestrator
from viggo.core.services.graph_service import GraphService
from viggo.core.services.redis_service import RedisService


class RAGFactory:
    """
    Factory service for creating RAG instances with the new SOLID-compliant architecture.
    
    This factory provides a clean interface to create RAG services while maintaining
    the separation of concerns and dependency injection principles.
    """
    
    def __init__(self):
        self.orchestrator = ConcreteRAGOrchestrator()
        self._default_rag_service: Optional[RAGService] = None
    
    def create_rag_service(self, 
                          graph_service: Optional[GraphService] = None,
                          redis_service: Optional[RedisService] = None,
                          config_type: str = "default",
                          **kwargs) -> RAGService:
        """
        Create a RAG service with the specified configuration.
        
        Args:
            graph_service: Optional Neo4j graph service
            redis_service: Optional Redis cache service
            config_type: Type of configuration ("default", "minimal", "custom")
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured RAG service instance
        """
        if config_type == "default":
            config = self.orchestrator.create_default_config(
                graph_service=graph_service,
                redis_service=redis_service,
                **kwargs
            )
        elif config_type == "minimal":
            config = self.orchestrator.create_minimal_config()
        elif config_type == "custom":
            components = kwargs.get('components', {})
            config = self.orchestrator.create_custom_config(components, **kwargs)
        else:
            raise ValueError(f"Unknown config type: {config_type}")
        
        return self.orchestrator.create_rag_service(config)
    
    def get_default_rag_service(self, 
                               graph_service: Optional[GraphService] = None,
                               redis_service: Optional[RedisService] = None) -> RAGService:
        """
        Get or create the default RAG service instance.
        
        Args:
            graph_service: Optional Neo4j graph service
            redis_service: Optional Redis cache service
            
        Returns:
            Default RAG service instance
        """
        if self._default_rag_service is None:
            self._default_rag_service = self.create_rag_service(
                graph_service=graph_service,
                redis_service=redis_service,
                config_type="default"
            )
        
        return self._default_rag_service
    
    def create_legacy_compatible_service(self, 
                                       graph_service: Optional[GraphService] = None,
                                       redis_service: Optional[RedisService] = None) -> 'LegacyCompatibleRAGService':
        """
        Create a RAG service that's compatible with the legacy interface.
        
        This provides a bridge between the old and new architectures.
        
        Args:
            graph_service: Optional Neo4j graph service
            redis_service: Optional Redis cache service
            
        Returns:
            Legacy-compatible RAG service
        """
        rag_service = self.create_rag_service(
            graph_service=graph_service,
            redis_service=redis_service,
            config_type="default"
        )
        
        return LegacyCompatibleRAGService(rag_service)
    
    def get_available_components(self) -> Dict[str, list]:
        """Get list of available components."""
        return self.orchestrator.get_available_components()
    
    def validate_configuration(self, config: RAGConfig) -> bool:
        """Validate a RAG configuration."""
        return self.orchestrator.validate_config(config)


class LegacyCompatibleRAGService:
    """
    Legacy-compatible wrapper for the new RAG service.
    
    This class provides the same interface as the old RAGService
    while using the new SOLID-compliant architecture underneath.
    """
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
    
    def process_document(self, file_path: str):
        """Legacy method for document processing."""
        result = self.rag_service.index_document(file_path)
        
        if result.success:
            return result.chunks_created, None, []  # Return format expected by legacy code
        else:
            raise Exception(f"Document processing failed: {result.error_message}")
    
    def perform_rag_query(self, question: str, page_number: int = None):
        """Legacy method for RAG queries."""
        from viggo.core.services.interfaces.retrieval import QueryContext
        
        context = QueryContext(
            query=question,
            page_filter=page_number
        )
        
        result = self.rag_service.query(question, context)
        
        return {
            "question": result.query,
            "answer": result.answer,
            "source_pages": result.source_pages,
            "search_method": "solid_architecture"
        }
    
    def query(self, query_text: str, k: int = 5):
        """Legacy method for simple queries."""
        from viggo.core.services.interfaces.retrieval import QueryContext
        
        context = QueryContext(query=query_text, top_k=k)
        result = self.rag_service.query(query_text, context)
        
        # Convert to legacy format
        legacy_results = []
        for i, page in enumerate(result.source_pages[:k]):
            legacy_results.append({
                "content": result.answer,  # Simplified for legacy compatibility
                "distance": 1.0 - (i * 0.1),  # Mock distance
                "metadata": {"page": page}
            })
        
        return legacy_results
    
    def get_system_status(self):
        """Get system status in legacy format."""
        status = self.rag_service.get_system_status()
        
        return {
            "vector_storage_available": status.get("vector_storage", {}).get("available", False),
            "graph_storage_available": status.get("graph_storage", {}).get("available", False),
            "cache_storage_available": status.get("cache_storage", {}).get("available", False),
            "total_vectors": status.get("vector_storage", {}).get("vector_count", 0),
            "available_retrievers": len(status.get("retrievers", {}).get("available_sources", [])),
            "available_generators": len(status.get("generators", {}).get("available_models", []))
        }


# Global factory instance
rag_factory = RAGFactory()


def get_rag_service(graph_service: Optional[GraphService] = None,
                   redis_service: Optional[RedisService] = None,
                   config_type: str = "default") -> RAGService:
    """
    Convenience function to get a RAG service instance.
    
    Args:
        graph_service: Optional Neo4j graph service
        redis_service: Optional Redis cache service
        config_type: Type of configuration ("default", "minimal", "custom")
        
    Returns:
        RAG service instance
    """
    return rag_factory.create_rag_service(
        graph_service=graph_service,
        redis_service=redis_service,
        config_type=config_type
    )


def get_legacy_compatible_service(graph_service: Optional[GraphService] = None,
                                redis_service: Optional[RedisService] = None) -> LegacyCompatibleRAGService:
    """
    Convenience function to get a legacy-compatible RAG service.
    
    Args:
        graph_service: Optional Neo4j graph service
        redis_service: Optional Redis cache service
        
    Returns:
        Legacy-compatible RAG service
    """
    return rag_factory.create_legacy_compatible_service(
        graph_service=graph_service,
        redis_service=redis_service
    )
