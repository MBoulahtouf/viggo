"""
Chunking interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ChunkLevel(Enum):
    """Hierarchical levels for document chunks."""
    CHAPTER = "chapter"
    PASSAGE = "passage"
    SENTENCE = "sentence"
    SECTION = "section"


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_id: str
    level: ChunkLevel
    page_number: int
    word_count: int
    char_count: int
    chapter_title: Optional[str] = None
    content_type: str = "story_content"
    lore_significance: float = 0.0
    entities: List[Dict[str, Any]] = None
    relationships: List[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.relationships is None:
            self.relationships = []


@dataclass
class Chunk:
    """Represents a chunk of document content."""
    id: str
    content: str
    level: ChunkLevel
    metadata: ChunkMetadata


@dataclass
class ChunkingResult:
    """Result of document chunking process."""
    chunks: Dict[ChunkLevel, List[Chunk]]
    metadata: Dict[str, ChunkMetadata]
    hierarchy: Dict[str, List[str]]
    statistics: Dict[str, Any]


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk_document(self, pages: List[Dict[str, Any]]) -> ChunkingResult:
        """Chunk a document into hierarchical pieces."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this chunking strategy."""
        pass


class ChunkingService(ABC):
    """Abstract base class for chunking services."""
    
    @abstractmethod
    def chunk_document(self, pages: List[Dict[str, Any]]) -> ChunkingResult:
        """Chunk a document using the configured strategy."""
        pass
    
    @abstractmethod
    def set_strategy(self, strategy: ChunkingStrategy) -> None:
        """Set the chunking strategy to use."""
        pass
    
    @abstractmethod
    def get_available_strategies(self) -> List[str]:
        """Get list of available chunking strategies."""
        pass
