"""
Entity extraction interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod


class IContentFilter(ABC):
    """Interface for content filtering operations."""

    @abstractmethod
    def is_story_content(self, content: str) -> bool:
        """Determine if content contains actual story material."""
        pass

    @abstractmethod
    def should_process_chunk(self, chunk_content: str, page_number: int) -> bool:
        """Determine if a chunk should be processed for entity extraction."""
        pass


class IEntityDeduplicator(ABC):
    """Interface for entity deduplication operations."""

    @abstractmethod
    def normalize_entity_name(self, entity_text: str) -> str:
        """Normalize entity name to a canonical form."""
        pass

    @abstractmethod
    def find_similar_entities(self, entity_name: str, existing_entities: list[dict]) -> list[dict]:
        """Find entities similar to the given entity name."""
        pass

    @abstractmethod
    def merge_entities(self, entities: list[dict]) -> list[dict]:
        """Merge similar entities into canonical forms."""
        pass


class IEntityDisambiguator(ABC):
    """Interface for entity disambiguation operations."""

    @abstractmethod
    def disambiguate_entity_type(self, entity_name: str, entity_label: str, context: str) -> str:
        """Disambiguate entity type based on context and known mappings."""
        pass


class IEnhancedEntityExtractor(ABC):
    """Interface for enhanced entity extraction operations."""

    @abstractmethod
    def should_process_content(self, content: str, page_number: int) -> bool:
        """Determine if content should be processed for entity extraction."""
        pass

    @abstractmethod
    def extract_entities_enhanced(self, content: str, page_number: int) -> list[dict]:
        """Extract entities with enhanced filtering and processing."""
        pass

    @abstractmethod
    def process_chunks_enhanced(self, chunks: list[dict]) -> list[dict]:
        """Process multiple chunks with enhanced entity extraction and global deduplication."""
        pass
