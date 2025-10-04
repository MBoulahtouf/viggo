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
    'RAGService', 'RAGConfig', 'RAGResult', 'IndexingResult', 'RAGMode', 'RAGOrchestrator'
]
