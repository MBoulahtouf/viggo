# Concrete implementations package for Viggo services

# Document processor implementations
from .document_processor_impl import (
    PDFDocumentProcessor, EPUBDocumentProcessor, ConcreteDocumentProcessorFactory
)

# Chunking implementations
from .chunking_impl import (
    HybridChunkingStrategy, StandardChunkingStrategy, ConcreteChunkingService
)

# Retrieval implementations
from .retrieval_impl import (
    SemanticRetriever, KeywordRetriever, GraphRetriever, ConcreteHybridRetriever, WeightedResultRanker
)

# Generation implementations
from .generation_impl import (
    LLMTextGenerator, TemplateTextGenerator, RAGPromptTemplate, ConcreteGenerationService
)

# Storage implementations
from .storage_impl import (
    FileStorageBackend, FAISSVectorStorage, Neo4jGraphStorage, RedisCacheStorage
)

# RAG implementations
from .rag_service_impl import ConcreteRAGService
from .rag_orchestrator import ConcreteRAGOrchestrator

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
    'FileStorageBackend', 'FAISSVectorStorage', 'Neo4jGraphStorage', 'RedisCacheStorage',
    
    # RAG
    'ConcreteRAGService', 'ConcreteRAGOrchestrator'
]
