"""
Document processing interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DocumentMetadata:
    """Metadata for a processed document."""
    filename: str
    file_path: str
    file_type: str
    page_count: int
    word_count: int
    processing_timestamp: float
    additional_metadata: Dict[str, Any]


@dataclass
class DocumentPage:
    """Represents a single page from a document."""
    page_number: int
    content: str
    metadata: Dict[str, Any]


class DocumentProcessor(ABC):
    """Abstract base class for document processors."""
    
    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        """Check if this processor can handle the given file."""
        pass
    
    @abstractmethod
    def process_document(self, file_path: str) -> List[DocumentPage]:
        """Process a document and return its pages."""
        pass
    
    @abstractmethod
    def get_document_metadata(self, file_path: str) -> DocumentMetadata:
        """Get metadata for a document."""
        pass


class DocumentProcessorFactory(ABC):
    """Abstract factory for document processors."""
    
    @abstractmethod
    def get_processor(self, file_path: str) -> Optional[DocumentProcessor]:
        """Get appropriate processor for file."""
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        pass
    
    @abstractmethod
    def is_supported(self, file_path: str) -> bool:
        """Check if file format is supported."""
        pass
