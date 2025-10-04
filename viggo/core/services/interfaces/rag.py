"""
RAG system interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .document_processor import DocumentProcessorFactory
from .chunking import ChunkingService
from .retrieval import HybridRetriever, QueryContext
from .generation import GenerationService, GenerationContext
from .storage import VectorStorage, GraphStorage, CacheStorage


class RAGMode(Enum):
    """Modes of RAG operation."""
    INDEXING = "indexing"
    QUERYING = "querying"
    UPDATING = "updating"


@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    mode: RAGMode
    document_processor_factory: DocumentProcessorFactory
    chunking_service: ChunkingService
    hybrid_retriever: HybridRetriever
    generation_service: GenerationService
    vector_storage: VectorStorage
    graph_storage: GraphStorage
    cache_storage: CacheStorage
    additional_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_config is None:
            self.additional_config = {}


@dataclass
class RAGResult:
    """Result of a RAG operation."""
    query: str
    answer: str
    source_pages: List[int]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]
    citations: List[str] = None
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = []


@dataclass
class IndexingResult:
    """Result of document indexing."""
    document_path: str
    chunks_created: int
    entities_extracted: int
    relationships_found: int
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class RAGService(ABC):
    """Abstract base class for RAG services."""
    
    @abstractmethod
    def index_document(self, document_path: str) -> IndexingResult:
        """Index a document for retrieval."""
        pass
    
    @abstractmethod
    def query(self, query: str, context: Optional[QueryContext] = None) -> RAGResult:
        """Query the RAG system."""
        pass
    
    @abstractmethod
    def update_document(self, document_path: str) -> IndexingResult:
        """Update an existing document index."""
        pass
    
    @abstractmethod
    def delete_document(self, document_path: str) -> bool:
        """Delete a document from the index."""
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get the status of the RAG system."""
        pass
    
    @abstractmethod
    def clear_index(self) -> bool:
        """Clear all indexed data."""
        pass


class RAGOrchestrator(ABC):
    """Abstract base class for RAG orchestration."""
    
    @abstractmethod
    def create_rag_service(self, config: RAGConfig) -> RAGService:
        """Create a RAG service with the given configuration."""
        pass
    
    @abstractmethod
    def get_available_components(self) -> Dict[str, List[str]]:
        """Get available components for RAG configuration."""
        pass
    
    @abstractmethod
    def validate_config(self, config: RAGConfig) -> bool:
        """Validate a RAG configuration."""
        pass
