"""
Content filter interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod
from enum import Enum


class ContentType(Enum):
    """Types of content that can be filtered."""
    STORY_CONTENT = "story_content"
    METADATA = "metadata"
    BIBLIOGRAPHY = "bibliography"
    PREFACE = "preface"
    PUBLISHER_INFO = "publisher_info"
    TECHNICAL = "technical"


class IContentFilterService(ABC):
    """Interface for content filtering operations."""

    @abstractmethod
    def classify_content_type(self, content: str, page_number: int = 0) -> ContentType:
        """Classify content type based on patterns and context."""
        pass

    @abstractmethod
    def should_index_content(self, content: str, page_number: int = 0) -> bool:
        """Determine if content should be indexed."""
        pass

    @abstractmethod
    def filter_chunks_for_indexing(self, chunks: list[dict]) -> tuple[list[dict], dict[str, int]]:
        """Filter chunks to only include those that should be indexed."""
        pass

    @abstractmethod
    def get_indexing_filter_expression(self) -> str:
        """Get Azure Cognitive Search filter expression."""
        pass

    @abstractmethod
    def add_content_type_to_chunk(self, chunk: dict) -> dict:
        """Add content type classification to a chunk."""
        pass

    @abstractmethod
    def get_filtering_stats(self, chunks: list[dict]) -> dict[str, any]:
        """Get detailed statistics about content filtering."""
        pass
