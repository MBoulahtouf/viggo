# Services package for Viggo

# Import the SOLID-compliant architecture
# Import concrete implementations
from .implementations import (
    AliasingService,
    AzureGraphRAGService,
    AzureSearchVectorStorage,
    ConcreteChunkingService,
    ConcreteDocumentProcessorFactory,
    ConcreteGenerationService,
    ConcreteHybridRetriever,
    ConcreteRAGOrchestrator,
    # RAG
    ConcreteRAGService,
    ContentFilter,
    ContentFilterService,
    ContextAggregatorAgent,
    EnhancedEntityExtractor,
    # Enhanced RAG implementations
    EnhancedRAGService,
    EntityDeduplicator,
    EntityDisambiguator,
    EntityExtractorAgent,
    EPUBDocumentProcessor,
    # Storage
    FileStorageBackend,
    GraphRetriever,
    # Core service implementations
    GraphService,
    HybridChunkingService,
    # Chunking
    HybridChunkingStrategy,
    HybridRetriever,
    HybridSearchService,
    KeywordRetriever,
    # Generation
    LLMTextGenerator,
    MultiAgentOrchestrator,
    Neo4jGraphStorage,
    # Document processors
    PDFDocumentProcessor,
    PerformanceOptimizer,
    # Multi-agent framework implementations
    QueryAnalyzerAgent,
    RAGPromptTemplate,
    RedisCacheStorage,
    RedisService,
    ResponseGeneratorAgent,
    # Retrieval
    SemanticRetriever,
    StandardChunkingStrategy,
    TemplateTextGenerator,
    WeightedResultRanker,
)

# Import enhanced RAG factory with multi-agent and GraphRAG capabilities
from .implementations.enhanced_rag_factory import (
    EnhancedRAGFactory,
    enhanced_rag_factory,
)

# Import interfaces for type hints and extension
# Content processing services (now with interfaces)
from .interfaces import (
    AgentResult,
    AgentType,
    AliasMapping,
    CacheStorage,
    CanonicalGroup,
    Chunk,
    ChunkingConfig,
    ChunkingResult,
    ChunkingService,
    # Chunking
    ChunkingStrategy,
    ChunkLevel,
    ChunkMetadata,
    ChunkType,
    ContentType,
    ContextAggregation,
    DocumentMetadata,
    DocumentPage,
    # Document processing
    DocumentProcessor,
    DocumentProcessorFactory,
    EntityExtraction,
    EntityGraphResult,
    GenerationContext,
    GenerationModel,
    GenerationResult,
    GenerationService,
    GraphServiceError,
    GraphStorage,
    HybridRetriever,
    # Multi-agent framework interfaces
    IAgent,
    IAliasingService,
    IContentFilter,
    IContentFilterService,
    IContextAggregator,
    IEnhancedEntityExtractor,
    IEntityDeduplicator,
    IEntityDisambiguator,
    IEntityExtractor,
    # Core service interfaces
    IGraphService,
    IHybridChunkingService,
    IHybridRetriever,
    IHybridSearchService,
    IMultiAgentOrchestrator,
    IndexingResult,
    IPerformanceOptimizer,
    IQueryAnalyzer,
    IRedisService,
    IResponseGenerator,
    NodeResult,
    PaginationParams,
    PromptTemplate,
    QueryAnalysis,
    QueryContext,
    RAGConfig,
    RAGMode,
    RAGOrchestrator,
    RAGResult,
    # RAG
    RAGService,
    RelationshipResult,
    ResultRanker,
    RetrievalResult,
    RetrievalSource,
    # Retrieval
    Retriever,
    # Storage
    StorageBackend,
    StorageMetadata,
    StorageType,
    # Generation
    TextGenerator,
    VectorStorage,
)
from .rag_factory import RAGFactory, get_rag_service

__all__ = [
    # SOLID architecture
    'RAGFactory',
    'get_rag_service',

    # Enhanced RAG factory
    'EnhancedRAGFactory',
    'enhanced_rag_factory',

    # Interfaces
    'DocumentProcessor', 'DocumentProcessorFactory', 'DocumentMetadata', 'DocumentPage',
    'ChunkingStrategy', 'ChunkingService', 'ChunkingResult', 'Chunk', 'ChunkMetadata', 'ChunkLevel',
    'Retriever', 'HybridRetriever', 'ResultRanker', 'RetrievalResult', 'RetrievalSource', 'QueryContext',
    'TextGenerator', 'PromptTemplate', 'GenerationService', 'GenerationResult', 'GenerationContext', 'GenerationModel',
    'StorageBackend', 'VectorStorage', 'GraphStorage', 'CacheStorage', 'StorageMetadata', 'StorageType',
    'RAGService', 'RAGConfig', 'RAGResult', 'IndexingResult', 'RAGMode', 'RAGOrchestrator',
    'IGraphService', 'PaginationParams', 'NodeResult', 'RelationshipResult', 'EntityGraphResult', 'GraphServiceError',
    'IHybridRetriever', 'IHybridSearchService', 'IPerformanceOptimizer', 'IRedisService', 'IAliasingService', 'AliasMapping', 'CanonicalGroup',

    # Multi-agent framework interfaces
    'IAgent', 'IQueryAnalyzer', 'IEntityExtractor', 'IContextAggregator', 'IResponseGenerator', 'IMultiAgentOrchestrator',
    'AgentType', 'AgentResult', 'QueryAnalysis', 'EntityExtraction', 'ContextAggregation',

    # Concrete implementations
    'PDFDocumentProcessor', 'EPUBDocumentProcessor', 'ConcreteDocumentProcessorFactory',
    'HybridChunkingStrategy', 'StandardChunkingStrategy', 'ConcreteChunkingService',
    'SemanticRetriever', 'KeywordRetriever', 'GraphRetriever', 'ConcreteHybridRetriever', 'WeightedResultRanker',
    'LLMTextGenerator', 'TemplateTextGenerator', 'RAGPromptTemplate', 'ConcreteGenerationService',
    'FileStorageBackend', 'AzureSearchVectorStorage', 'Neo4jGraphStorage', 'RedisCacheStorage',
    'ConcreteRAGService', 'ConcreteRAGOrchestrator',
    'GraphService', 'HybridRetriever', 'HybridSearchService', 'PerformanceOptimizer', 'RedisService', 'AliasingService',

    # Multi-agent framework implementations
    'QueryAnalyzerAgent', 'EntityExtractorAgent', 'ContextAggregatorAgent', 'ResponseGeneratorAgent', 'MultiAgentOrchestrator',

    # Enhanced RAG implementations
    'EnhancedRAGService', 'AzureGraphRAGService',

    # Content processing interfaces
    'IContentFilterService', 'ContentType',
    'IContentFilter', 'IEntityDeduplicator', 'IEntityDisambiguator', 'IEnhancedEntityExtractor',
    'IHybridChunkingService', 'ChunkLevel', 'ChunkType', 'ChunkMetadata', 'ChunkingConfig',

    # Content processing implementations
    'ContentFilterService', 'ContentFilter', 'EntityDeduplicator', 'EntityDisambiguator',
    'EnhancedEntityExtractor', 'HybridChunkingService'
]
