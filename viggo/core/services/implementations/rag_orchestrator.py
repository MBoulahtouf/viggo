"""
RAG Orchestrator following SOLID principles.
"""

from typing import Dict, List, Any, Optional

from viggo.core.services.interfaces.rag import RAGOrchestrator, RAGConfig, RAGService
from viggo.core.services.interfaces.document_processor import DocumentProcessorFactory
from viggo.core.services.interfaces.chunking import ChunkingService
from viggo.core.services.interfaces.retrieval import HybridRetriever
from viggo.core.services.interfaces.generation import GenerationService
from viggo.core.services.interfaces.storage import VectorStorage, GraphStorage, CacheStorage

from .document_processor_impl import ConcreteDocumentProcessorFactory
from .chunking_impl import ConcreteChunkingService, HybridChunkingStrategy
from .retrieval_impl import ConcreteHybridRetriever, SemanticRetriever, KeywordRetriever, GraphRetriever
from .generation_impl import ConcreteGenerationService, LLMTextGenerator, TemplateTextGenerator
from .storage_impl import AzureSearchVectorStorage, Neo4jGraphStorage, RedisCacheStorage
from .rag_service_impl import ConcreteRAGService

from .graph_service_impl import GraphService
from .redis_service_impl import RedisService


class ConcreteRAGOrchestrator(RAGOrchestrator):
    """Concrete implementation of RAG orchestrator following SOLID principles."""
    
    def __init__(self):
        self.available_components = {
            "document_processors": ["pdf", "epub"],
            "chunking_strategies": ["hybrid", "standard"],
            "retrievers": ["semantic", "keyword", "graph"],
            "generators": ["llm", "template"],
            "storage_backends": ["faiss", "neo4j", "redis", "file"]
        }
    
    def create_rag_service(self, config: RAGConfig) -> RAGService:
        """Create a RAG service with the given configuration."""
        if not self.validate_config(config):
            raise ValueError("Invalid RAG configuration")
        
        return ConcreteRAGService(config)
    
    def get_available_components(self) -> Dict[str, List[str]]:
        """Get available components for RAG configuration."""
        return self.available_components.copy()
    
    def validate_config(self, config: RAGConfig) -> bool:
        """Validate a RAG configuration."""
        try:
            # Check required components (core components that must be present)
            if not config.document_processor_factory:
                return False
            if not config.chunking_service:
                return False
            if not config.hybrid_retriever:
                return False
            if not config.generation_service:
                return False
            if not config.vector_storage:
                return False
            
            # Optional components (can be None for minimal configurations)
            # graph_storage and cache_storage are optional
            
            return True
            
        except Exception as e:
            print(f"Config validation error: {e}")
            return False
    
    def create_default_config(self, 
                            graph_service: Optional[GraphService] = None,
                            redis_service: Optional[RedisService] = None,
                            vector_index_path: str = "vector_index.bin") -> RAGConfig:
        """Create a default RAG configuration with all components."""
        
        # Document processor factory
        document_processor_factory = ConcreteDocumentProcessorFactory()
        
        # Chunking service with hybrid strategy
        chunking_service = ConcreteChunkingService()
        chunking_service.set_strategy(HybridChunkingStrategy())
        
        # Vector storage
        vector_storage = AzureSearchVectorStorage()
        
        # Graph storage
        if graph_service:
            graph_storage = Neo4jGraphStorage(graph_service)
        else:
            # Create a mock graph service for testing
            graph_storage = None  # Would need to handle this case
        
        # Cache storage
        if redis_service:
            cache_storage = RedisCacheStorage(redis_service)
        else:
            # Create a mock cache service for testing
            cache_storage = None  # Would need to handle this case
        
        # Hybrid retriever
        hybrid_retriever = ConcreteHybridRetriever()
        
        # Add retrievers to hybrid retriever
        # Note: These would need actual data to work properly
        # semantic_retriever = SemanticRetriever(None, [])  # Would need vector index and chunks
        # keyword_retriever = KeywordRetriever()
        # graph_retriever = GraphRetriever(graph_service) if graph_service else None
        
        # hybrid_retriever.add_retriever(semantic_retriever)
        # hybrid_retriever.add_retriever(keyword_retriever)
        # if graph_retriever:
        #     hybrid_retriever.add_retriever(graph_retriever)
        
        # Generation service
        generation_service = ConcreteGenerationService()
        generation_service.add_generator(LLMTextGenerator())
        generation_service.add_generator(TemplateTextGenerator())
        
        return RAGConfig(
            mode="querying",  # Default mode
            document_processor_factory=document_processor_factory,
            chunking_service=chunking_service,
            hybrid_retriever=hybrid_retriever,
            generation_service=generation_service,
            vector_storage=vector_storage,
            graph_storage=graph_storage,
            cache_storage=cache_storage
        )
    
    def create_minimal_config(self) -> RAGConfig:
        """Create a minimal RAG configuration for testing."""
        
        # Document processor factory
        document_processor_factory = ConcreteDocumentProcessorFactory()
        
        # Chunking service with standard strategy
        chunking_service = ConcreteChunkingService()
        chunking_service.set_strategy(chunking_service.create_strategy("standard"))
        
        # Vector storage
        vector_storage = AzureSearchVectorStorage()
        
        # Hybrid retriever (empty for now)
        hybrid_retriever = ConcreteHybridRetriever()
        
        # Generation service with template generator only
        generation_service = ConcreteGenerationService()
        generation_service.add_generator(TemplateTextGenerator())
        
        return RAGConfig(
            mode="querying",
            document_processor_factory=document_processor_factory,
            chunking_service=chunking_service,
            hybrid_retriever=hybrid_retriever,
            generation_service=generation_service,
            vector_storage=vector_storage,
            graph_storage=None,  # Minimal config
            cache_storage=None   # Minimal config
        )
    
    def create_custom_config(self, 
                           components: Dict[str, Any],
                           **kwargs) -> RAGConfig:
        """Create a custom RAG configuration with specified components."""
        
        # Document processor factory
        document_processor_factory = components.get('document_processor_factory', 
                                                   ConcreteDocumentProcessorFactory())
        
        # Chunking service
        chunking_service = components.get('chunking_service', ConcreteChunkingService())
        chunking_strategy = components.get('chunking_strategy', 'hybrid')
        if hasattr(chunking_service, 'create_strategy'):
            strategy = chunking_service.create_strategy(chunking_strategy)
            chunking_service.set_strategy(strategy)
        
        # Storage backends
        vector_storage = components.get('vector_storage', AzureSearchVectorStorage())
        graph_storage = components.get('graph_storage')
        cache_storage = components.get('cache_storage')
        
        # Hybrid retriever
        hybrid_retriever = components.get('hybrid_retriever', ConcreteHybridRetriever())
        
        # Add retrievers if specified
        retrievers = components.get('retrievers', [])
        for retriever in retrievers:
            hybrid_retriever.add_retriever(retriever)
        
        # Generation service
        generation_service = components.get('generation_service', ConcreteGenerationService())
        
        # Add generators if specified
        generators = components.get('generators', [])
        for generator in generators:
            generation_service.add_generator(generator)
        
        return RAGConfig(
            mode=kwargs.get('mode', 'querying'),
            document_processor_factory=document_processor_factory,
            chunking_service=chunking_service,
            hybrid_retriever=hybrid_retriever,
            generation_service=generation_service,
            vector_storage=vector_storage,
            graph_storage=graph_storage,
            cache_storage=cache_storage,
            additional_config=kwargs
        )
