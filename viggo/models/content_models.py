"""
Content processing models for the Viggo system.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChunkLevel(str, Enum):
    """Chunk level enumeration."""
    BOOK = "book"
    CHAPTER = "chapter"
    SECTION = "section"
    PASSAGE = "passage"
    SENTENCE = "sentence"


class ChunkType(str, Enum):
    """Chunk type enumeration."""
    FULL_CHAPTER = "full_chapter"
    PARAGRAPH_GROUP = "paragraph_group"
    OVERLAPPING_PASSAGE = "overlapping_passage"
    STANDARD_PASSAGE = "standard_passage"
    CRITICAL_LORE = "critical_lore"
    DIALOGUE_BLOCK = "dialogue_block"
    NARRATIVE_BLOCK = "narrative_block"


class ChunkMetadata(BaseModel):
    """Chunk metadata model."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    level: ChunkLevel = Field(..., description="Chunk level")
    chunk_type: ChunkType = Field(..., description="Chunk type")
    parent_id: str | None = Field(None, description="Parent chunk ID")
    children_ids: list[str] = Field(default_factory=list, description="Child chunk IDs")
    word_count: int = Field(..., ge=0, description="Word count")
    char_count: int = Field(..., ge=0, description="Character count")
    page_number: int = Field(..., ge=1, description="Page number")
    chapter_title: str | None = Field(None, description="Chapter title")
    section_title: str | None = Field(None, description="Section title")
    entities: list[dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    relationships: list[dict[str, Any]] = Field(default_factory=list, description="Extracted relationships")
    content_type: str = Field("story_content", description="Content type")
    lore_significance: float = Field(0.0, ge=0.0, le=1.0, description="Lore significance score")
    overlap_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Overlap ratio with other chunks")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class ChunkModel(BaseModel):
    """Chunk model."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk content")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    embedding: list[float] | None = Field(None, description="Chunk embedding vector")
    similarity_scores: dict[str, float] = Field(default_factory=dict, description="Similarity scores")


class ChunkingRequest(BaseModel):
    """Request model for document chunking."""
    document_id: str = Field(..., description="Document identifier")
    chunking_strategy: str = Field("hybrid", description="Chunking strategy")
    max_chunk_size: int = Field(400, ge=50, le=1000, description="Maximum chunk size in words")
    min_chunk_size: int = Field(50, ge=10, le=200, description="Minimum chunk size in words")
    overlap_ratio: float = Field(0.2, ge=0.0, le=0.5, description="Chunk overlap ratio")
    enable_hierarchical: bool = Field(True, description="Enable hierarchical chunking")
    enable_entity_extraction: bool = Field(True, description="Enable entity extraction")
    enable_content_filtering: bool = Field(True, description="Enable content filtering")
    skip_metadata_pages: int = Field(2, ge=0, le=10, description="Number of metadata pages to skip")


class ChunkingResponse(BaseModel):
    """Response model for document chunking."""
    document_id: str = Field(..., description="Document identifier")
    chunks_created: int = Field(..., ge=0, description="Number of chunks created")
    chunking_strategy: str = Field(..., description="Chunking strategy used")
    processing_time: float = Field(..., description="Processing time in seconds")
    chunk_levels: dict[str, int] = Field(..., description="Chunks by level")
    chunk_types: dict[str, int] = Field(..., description="Chunks by type")
    entities_extracted: int = Field(0, ge=0, description="Number of entities extracted")
    relationships_created: int = Field(0, ge=0, description="Number of relationships created")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ContentType(str, Enum):
    """Content type enumeration."""
    STORY_CONTENT = "story_content"
    METADATA = "metadata"
    BIBLIOGRAPHY = "bibliography"
    PREFACE = "preface"
    PUBLISHER_INFO = "publisher_info"
    TECHNICAL = "technical"


class ContentFilterRequest(BaseModel):
    """Request model for content filtering."""
    content: str = Field(..., description="Content to filter")
    page_number: int = Field(0, ge=0, description="Page number")
    document_type: str = Field("book", description="Document type")
    filter_metadata: bool = Field(True, description="Filter metadata content")
    filter_bibliography: bool = Field(True, description="Filter bibliography content")
    filter_preface: bool = Field(False, description="Filter preface content")
    custom_patterns: list[str] = Field(default_factory=list, description="Custom filter patterns")


class ContentFilterResponse(BaseModel):
    """Response model for content filtering."""
    should_index: bool = Field(..., description="Whether content should be indexed")
    content_type: ContentType = Field(..., description="Detected content type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    filtered_content: str = Field(..., description="Filtered content")
    filter_reasons: list[str] = Field(default_factory=list, description="Reasons for filtering")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EntityExtractionRequest(BaseModel):
    """Request model for entity extraction."""
    content: str = Field(..., description="Content to extract entities from")
    page_number: int = Field(0, ge=0, description="Page number")
    entity_types: list[str] = Field(default_factory=list, description="Entity types to extract")
    enable_deduplication: bool = Field(True, description="Enable entity deduplication")
    enable_disambiguation: bool = Field(True, description="Enable entity disambiguation")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_entities: int = Field(100, ge=1, le=1000, description="Maximum entities to extract")


class EntityExtractionResponse(BaseModel):
    """Response model for entity extraction."""
    entities: list[dict[str, Any]] = Field(..., description="Extracted entities")
    total_entities: int = Field(..., ge=0, description="Total number of entities")
    unique_entities: int = Field(..., ge=0, description="Number of unique entities")
    entity_types: dict[str, int] = Field(..., description="Entities by type")
    processing_time: float = Field(..., description="Processing time in seconds")
    deduplication_applied: bool = Field(..., description="Whether deduplication was applied")
    disambiguation_applied: bool = Field(..., description="Whether disambiguation was applied")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentProcessingRequest(BaseModel):
    """Request model for document processing."""
    document_id: str = Field(..., description="Document identifier")
    processing_options: dict[str, Any] = Field(default_factory=dict, description="Processing options")
    chunking_config: ChunkingRequest | None = Field(None, description="Chunking configuration")
    entity_extraction_config: EntityExtractionRequest | None = Field(None, description="Entity extraction configuration")
    content_filtering_config: ContentFilterRequest | None = Field(None, description="Content filtering configuration")


class DocumentProcessingResponse(BaseModel):
    """Response model for document processing."""
    document_id: str = Field(..., description="Document identifier")
    processing_status: str = Field(..., description="Processing status")
    chunks_created: int = Field(0, ge=0, description="Number of chunks created")
    entities_extracted: int = Field(0, ge=0, description="Number of entities extracted")
    relationships_created: int = Field(0, ge=0, description="Number of relationships created")
    processing_time: float = Field(..., description="Total processing time in seconds")
    error_message: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
