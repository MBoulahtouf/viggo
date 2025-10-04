# Interfaces package for Viggo services

# Document processing interfaces
from .document_processor import (
    DocumentProcessor, DocumentProcessorFactory, DocumentMetadata, DocumentPage
)

# Chunking interfaces
from .chunking import (
    ChunkingStrategy, ChunkingService, ChunkingResult, Chunk, ChunkMetadata, ChunkLevel
)

# Retrieval interfaces
from .retrieval import (
    Retriever, HybridRetriever, ResultRanker, RetrievalResult, RetrievalSource, QueryContext
)

# Generation interfaces
from .generation import (
    TextGenerator, PromptTemplate, GenerationService, GenerationResult, GenerationContext, GenerationModel
)

# Storage interfaces
from .storage import (
    StorageBackend, VectorStorage, GraphStorage, CacheStorage, StorageMetadata, StorageType
)

# RAG interfaces
from .rag import (
    RAGService, RAGConfig, RAGResult, IndexingResult, RAGMode, RAGOrchestrator
)

# Core service interfaces
from .graph_service import (
    IGraphService, PaginationParams, NodeResult, RelationshipResult, EntityGraphResult, GraphServiceError
)

from .hybrid_retriever import (
    IHybridRetriever
)

from .hybrid_search_service import (
    IHybridSearchService
)

from .performance_optimizer import (
    IPerformanceOptimizer
)

from .redis_service import (
    IRedisService
)

from .aliasing_service import (
    IAliasingService, AliasMapping, CanonicalGroup
)

# Content processing interfaces
from .content_filter import (
    IContentFilterService, ContentType
)

from .entity_extractor import (
    IContentFilter, IEntityDeduplicator, IEntityDisambiguator, IEnhancedEntityExtractor
)

from .chunking_service import (
    IHybridChunkingService, ChunkLevel, ChunkType, ChunkMetadata, ChunkingConfig
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
    'IHybridChunkingService', 'ChunkLevel', 'ChunkType', 'ChunkMetadata', 'ChunkingConfig'
]
