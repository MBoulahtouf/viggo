# Services package for Viggo

# Import the new SOLID-compliant architecture
from .rag_factory import (
    RAGFactory, 
    get_rag_service, 
    get_legacy_compatible_service,
    LegacyCompatibleRAGService
)

# Import interfaces for type hints and extension
from .interfaces import (
    # Document processing
    DocumentProcessor, DocumentProcessorFactory, DocumentMetadata, DocumentPage,
    
    # Chunking
    ChunkingStrategy, ChunkingService, ChunkingResult, Chunk, ChunkMetadata, ChunkLevel,
    
    # Retrieval
    Retriever, HybridRetriever, ResultRanker, RetrievalResult, RetrievalSource, QueryContext,
    
    # Generation
    TextGenerator, PromptTemplate, GenerationService, GenerationResult, GenerationContext, GenerationModel,
    
    # Storage
    StorageBackend, VectorStorage, GraphStorage, CacheStorage, StorageMetadata, StorageType,
    
    # RAG
    RAGService, RAGConfig, RAGResult, IndexingResult, RAGMode, RAGOrchestrator
)

# Import concrete implementations
from .implementations import (
    # Document processors
    PDFDocumentProcessor, EPUBDocumentProcessor, ConcreteDocumentProcessorFactory,
    
    # Chunking
    HybridChunkingStrategy, StandardChunkingStrategy, ConcreteChunkingService,
    
    # Retrieval
    SemanticRetriever, KeywordRetriever, GraphRetriever, ConcreteHybridRetriever, WeightedResultRanker,
    
    # Generation
    LLMTextGenerator, TemplateTextGenerator, RAGPromptTemplate, ConcreteGenerationService,
    
    # Storage
    FileStorageBackend, FAISSVectorStorage, Neo4jGraphStorage, RedisCacheStorage,
    
    # RAG
    ConcreteRAGService, ConcreteRAGOrchestrator
)

# Legacy imports for backward compatibility
from .rag_service import RAGService as LegacyRAGService
from .enhanced_rag_service import EnhancedRAGService
from .hybrid_retriever import HybridRetriever as LegacyHybridRetriever
from .graph_service import GraphService
from .redis_service import RedisService
from .hybrid_search_service import HybridSearchService
from .performance_optimizer import PerformanceOptimizer
from .aliasing_service import AliasingService
from .content_filter_service import ContentFilterService
from .context_aware_retriever import ContextAwareRetriever
from .entity_chunk_linker import EntityChunkLinker
from .enhanced_entity_extractor import EnhancedEntityExtractor
from .hybrid_chunking_service import HybridChunkingService
from .user_aware_rag_service import UserAwareRAGService
from .user_progress_service import UserProgressService
from .enhanced_rag_integration import EnhancedRAGIntegration

__all__ = [
    # New SOLID architecture
    'RAGFactory',
    'get_rag_service',
    'get_legacy_compatible_service',
    'LegacyCompatibleRAGService',
    
    # Interfaces
    'DocumentProcessor', 'DocumentProcessorFactory', 'DocumentMetadata', 'DocumentPage',
    'ChunkingStrategy', 'ChunkingService', 'ChunkingResult', 'Chunk', 'ChunkMetadata', 'ChunkLevel',
    'Retriever', 'HybridRetriever', 'ResultRanker', 'RetrievalResult', 'RetrievalSource', 'QueryContext',
    'TextGenerator', 'PromptTemplate', 'GenerationService', 'GenerationResult', 'GenerationContext', 'GenerationModel',
    'StorageBackend', 'VectorStorage', 'GraphStorage', 'CacheStorage', 'StorageMetadata', 'StorageType',
    'RAGService', 'RAGConfig', 'RAGResult', 'IndexingResult', 'RAGMode', 'RAGOrchestrator',
    
    # Concrete implementations
    'PDFDocumentProcessor', 'EPUBDocumentProcessor', 'ConcreteDocumentProcessorFactory',
    'HybridChunkingStrategy', 'StandardChunkingStrategy', 'ConcreteChunkingService',
    'SemanticRetriever', 'KeywordRetriever', 'GraphRetriever', 'ConcreteHybridRetriever', 'WeightedResultRanker',
    'LLMTextGenerator', 'TemplateTextGenerator', 'RAGPromptTemplate', 'ConcreteGenerationService',
    'FileStorageBackend', 'FAISSVectorStorage', 'Neo4jGraphStorage', 'RedisCacheStorage',
    'ConcreteRAGService', 'ConcreteRAGOrchestrator',
    
    # Legacy services (for backward compatibility)
    'LegacyRAGService',
    'EnhancedRAGService',
    'LegacyHybridRetriever',
    'GraphService',
    'RedisService',
    'HybridSearchService',
    'PerformanceOptimizer',
    'AliasingService',
    'ContentFilterService',
    'ContextAwareRetriever',
    'EntityChunkLinker',
    'EnhancedEntityExtractor',
    'HybridChunkingService',
    'UserAwareRAGService',
    'UserProgressService',
    'EnhancedRAGIntegration'
]
