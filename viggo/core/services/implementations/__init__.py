# Concrete implementations package for Viggo services

# Document processor implementations
from .aliasing_service_impl import AliasingService
from .azure_graph_rag_impl import AzureGraphRAGService

# Chunking implementations
from .chunking_impl import (
    ConcreteChunkingService,
    HybridChunkingStrategy,
    StandardChunkingStrategy,
)

# Content processing implementations
from .content_filter_service_impl import ContentFilterService
from .document_processor_impl import (
    ConcreteDocumentProcessorFactory,
    EPUBDocumentProcessor,
    PDFDocumentProcessor,
)
from .enhanced_entity_extractor_impl import (
    ContentFilter,
    EnhancedEntityExtractor,
    EntityDeduplicator,
    EntityDisambiguator,
)
from .enhanced_rag_factory import EnhancedRAGFactory
from .enhanced_rag_service_impl import EnhancedRAGService

# Generation implementations
from .generation_impl import (
    ConcreteGenerationService,
    LLMTextGenerator,
    RAGPromptTemplate,
    TemplateTextGenerator,
)

# Core service implementations
from .graph_service_impl import GraphService
from .hybrid_chunking_service_impl import HybridChunkingService
from .hybrid_retriever_impl import HybridRetriever
from .hybrid_search_service_impl import HybridSearchService

# Multi-agent implementations
from .multi_agent_impl import (
    ContextAggregatorAgent,
    EntityExtractorAgent,
    MultiAgentOrchestrator,
    QueryAnalyzerAgent,
    ResponseGeneratorAgent,
)
from .performance_optimizer_impl import PerformanceOptimizer
from .rag_orchestrator import ConcreteRAGOrchestrator

# RAG implementations
from .rag_service_impl import ConcreteRAGService
from .redis_service_impl import RedisService

# Retrieval implementations
from .retrieval_impl import (
    ConcreteHybridRetriever,
    GraphRetriever,
    KeywordRetriever,
    SemanticRetriever,
    WeightedResultRanker,
)

# Storage implementations
from .storage_impl import (
    AzureSearchVectorStorage,
    FileStorageBackend,
    Neo4jGraphStorage,
    RedisCacheStorage,
)

__all__ = [
    # Document processors
    'PDFDocumentProcessor', 'EPUBDocumentProcessor', 'ConcreteDocumentProcessorFactory',

    # Chunking
    'HybridChunkingStrategy', 'StandardChunkingStrategy', 'ConcreteChunkingService',

    # Retrieval
    'SemanticRetriever', 'KeywordRetriever', 'GraphRetriever', 'ConcreteHybridRetriever', 'WeightedResultRanker',

    # Generation
    'LLMTextGenerator', 'TemplateTextGenerator', 'RAGPromptTemplate', 'ConcreteGenerationService',

    # Storage
    'FileStorageBackend', 'AzureSearchVectorStorage', 'Neo4jGraphStorage', 'RedisCacheStorage',

    # RAG
    'ConcreteRAGService', 'ConcreteRAGOrchestrator',

    # Content processing implementations
    'ContentFilterService', 'ContentFilter', 'EntityDeduplicator', 'EntityDisambiguator', 'EnhancedEntityExtractor', 'HybridChunkingService',

    # Core service implementations
    'GraphService', 'HybridRetriever', 'HybridSearchService', 'PerformanceOptimizer', 'RedisService', 'AliasingService',

    # Multi-agent implementations
    'QueryAnalyzerAgent', 'EntityExtractorAgent', 'ContextAggregatorAgent', 'ResponseGeneratorAgent', 'MultiAgentOrchestrator',
    'EnhancedRAGService', 'AzureGraphRAGService', 'EnhancedRAGFactory'
]
