"""
Chunking service interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class ChunkLevel(Enum):
    """Hierarchical chunk levels for multi-granularity retrieval."""
    BOOK = "book"
    CHAPTER = "chapter"
    SECTION = "section"
    PASSAGE = "passage"
    SENTENCE = "sentence"


class ChunkType(Enum):
    """Types of chunks for different retrieval strategies."""
    FULL_CHAPTER = "full_chapter"
    PARAGRAPH_GROUP = "paragraph_group"
    OVERLAPPING_PASSAGE = "overlapping_passage"
    STANDARD_PASSAGE = "standard_passage"
    CRITICAL_LORE = "critical_lore"
    DIALOGUE_BLOCK = "dialogue_block"
    NARRATIVE_BLOCK = "narrative_block"


@dataclass
class ChunkMetadata:
    """Metadata for a chunk with hierarchical information."""
    level: ChunkLevel
    chunk_type: ChunkType
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    word_count: int = 0
    char_count: int = 0
    page_number: int = 0
    chapter_title: str = ""
    section_title: str = ""
    entities: List[Dict] = None
    relationships: List[Dict] = None
    content_type: str = "story_content"
    lore_significance: float = 0.0
    overlap_ratio: float = 0.0
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.entities is None:
            self.entities = []
        if self.relationships is None:
            self.relationships = []


@dataclass
class ChunkingConfig:
    """Configuration for hybrid chunking strategy."""
    max_chapter_words: int = 2000
    min_chapter_words: int = 100
    max_passage_words: int = 400
    min_passage_words: int = 50
    passage_overlap_ratio: float = 0.2
    critical_lore_threshold: float = 0.7
    max_overlap_chunks: int = 3
    enable_hierarchical: bool = True
    max_children_per_parent: int = 10
    enable_content_filtering: bool = True
    skip_metadata_pages: int = 2


class IHybridChunkingService(ABC):
    """Interface for hybrid chunking operations."""
    
    @abstractmethod
    def chunk_document_hierarchical(self, document_store: List[Dict]) -> Dict[str, List[Dict]]:
        """Main entry point for hierarchical document chunking."""
        pass
    
    @abstractmethod
    def get_chunks_by_level(self, level: ChunkLevel) -> List[Dict]:
        """Get chunks at a specific hierarchical level."""
        pass
    
    @abstractmethod
    def get_chunk_children(self, chunk_id: str) -> List[Dict]:
        """Get child chunks of a specific chunk."""
        pass
    
    @abstractmethod
    def get_chunk_parent(self, chunk_id: str) -> Optional[ChunkMetadata]:
        """Get parent chunk of a specific chunk."""
        pass
    
    @abstractmethod
    def get_critical_lore_chunks(self, threshold: float = 0.7) -> List[Dict]:
        """Get chunks with high lore significance."""
        pass
    
    @abstractmethod
    def get_chunking_summary(self) -> Dict:
        """Get a summary of the chunking process."""
        pass
