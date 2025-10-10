# Interfaces package for Viggo services

# Document processing interfaces
from .aliasing_service import AliasMapping, CanonicalGroup, IAliasingService

# Chunking interfaces
from .chunking import (
    Chunk,
    ChunkingResult,
    ChunkingService,
    ChunkingStrategy,
    ChunkLevel,
    ChunkMetadata,
)
from .chunking_service import (
    ChunkingConfig,
    ChunkLevel,
    ChunkMetadata,
    ChunkType,
    IHybridChunkingService,
)

# Content processing interfaces
from .content_filter import ContentType, IContentFilterService
from .document_processor import (
    DocumentMetadata,
    DocumentPage,
    DocumentProcessor,
    DocumentProcessorFactory,
)
from .entity_extractor import (
    IContentFilter,
    IEnhancedEntityExtractor,
    IEntityDeduplicator,
    IEntityDisambiguator,
)

# Generation interfaces
from .generation import (
    GenerationContext,
    GenerationModel,
    GenerationResult,
    GenerationService,
    PromptTemplate,
    TextGenerator,
)

# Core service interfaces
from .graph_service import (
    EntityGraphResult,
    GraphServiceError,
    IGraphService,
    NodeResult,
    PaginationParams,
    RelationshipResult,
)
from .hybrid_retriever import IHybridRetriever
from .hybrid_search_service import IHybridSearchService

# Multi-agent interfaces
from .multi_agent import (
    AgentResult,
    AgentType,
    ContextAggregation,
    EntityExtraction,
    IAgent,
    IContextAggregator,
    IEntityExtractor,
    IMultiAgentOrchestrator,
    IQueryAnalyzer,
    IResponseGenerator,
    QueryAnalysis,
)
from .performance_optimizer import IPerformanceOptimizer

# RAG interfaces
from .rag import (
    IndexingResult,
    RAGConfig,
    RAGMode,
    RAGOrchestrator,
    RAGResult,
    RAGService,
)
from .redis_service import IRedisService

# Retrieval interfaces
from .retrieval import (
    HybridRetriever,
    QueryContext,
    ResultRanker,
    RetrievalResult,
    RetrievalSource,
    Retriever,
)

# Storage interfaces
from .storage import (
    CacheStorage,
    GraphStorage,
    StorageBackend,
    StorageMetadata,
    StorageType,
    VectorStorage,
)

__all__ = [
    # Document processing
    'DocumentProcessor', 'DocumentProcessorFactory', 'DocumentMetadata', 'DocumentPage',

    # Chunking
    'ChunkingStrategy', 'ChunkingService', 'ChunkingResult', 'Chunk', 'ChunkMetadata', 'ChunkLevel',

    # Retrieval
    'Retriever', 'HybridRetriever', 'ResultRanker', 'RetrievalResult', 'RetrievalSource', 'QueryContext',

    # Generation
    'TextGenerator', 'PromptTemplate', 'GenerationService', 'GenerationResult', 'GenerationContext', 'GenerationModel',

    # Storage
    'StorageBackend', 'VectorStorage', 'GraphStorage', 'CacheStorage', 'StorageMetadata', 'StorageType',

    # RAG
    'RAGService', 'RAGConfig', 'RAGResult', 'IndexingResult', 'RAGMode', 'RAGOrchestrator',

    # Core service interfaces
    'IGraphService', 'PaginationParams', 'NodeResult', 'RelationshipResult', 'EntityGraphResult', 'GraphServiceError',
    'IHybridRetriever',
    'IHybridSearchService',
    'IPerformanceOptimizer',
    'IRedisService',
    'IAliasingService', 'AliasMapping', 'CanonicalGroup',

    # Content processing interfaces
    'IContentFilterService', 'ContentType',
    'IContentFilter', 'IEntityDeduplicator', 'IEntityDisambiguator', 'IEnhancedEntityExtractor',
    'IHybridChunkingService', 'ChunkLevel', 'ChunkType', 'ChunkMetadata', 'ChunkingConfig',

    # Multi-agent interfaces
    'IAgent', 'IQueryAnalyzer', 'IEntityExtractor', 'IContextAggregator', 'IResponseGenerator', 'IMultiAgentOrchestrator',
    'AgentResult', 'QueryAnalysis', 'EntityExtraction', 'ContextAggregation', 'AgentType'
]
