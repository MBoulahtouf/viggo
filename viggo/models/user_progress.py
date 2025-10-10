"""
User Progress Models for Viggo

This module defines models for tracking user reading progress and spoiler protection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ReadingStatus(Enum):
    """User's reading status for a document."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"
    ABANDONED = "abandoned"


@dataclass
class UserProgress:
    """
    Tracks user's reading progress for a specific document.
    """
    user_id: str
    document_id: str
    document_name: str
    current_page: int = 0
    total_pages: int = 0
    reading_status: ReadingStatus = ReadingStatus.NOT_STARTED
    finished_book: bool = False
    last_updated: datetime = field(default_factory=datetime.now)

    # Additional metadata
    reading_speed: float | None = None  # pages per minute
    time_spent: float | None = None  # total minutes spent reading
    bookmarks: dict[int, str] = field(default_factory=dict)  # page -> note
    notes: dict[int, str] = field(default_factory=dict)  # page -> note

    def __post_init__(self):
        """Update status based on current page and finished_book flag."""
        if self.finished_book:
            self.reading_status = ReadingStatus.FINISHED
        elif self.current_page > 0:
            self.reading_status = ReadingStatus.IN_PROGRESS
        else:
            self.reading_status = ReadingStatus.NOT_STARTED

    def update_progress(self, page: int, finished: bool = False) -> None:
        """
        Update user's reading progress.
        
        Args:
            page: Current page number
            finished: Whether the user has finished the book
        """
        self.current_page = page
        self.finished_book = finished
        self.last_updated = datetime.now()
        self.__post_init__()

    def get_spoiler_limit(self) -> int | None:
        """
        Get the page limit for spoiler protection.
        
        Returns:
            Page number limit, or None if no spoiler protection needed
        """
        if self.finished_book:
            return None  # No spoiler protection if finished
        return self.current_page

    def is_spoiler_protected(self) -> bool:
        """
        Check if spoiler protection is active.
        
        Returns:
            True if spoiler protection is active
        """
        return not self.finished_book and self.current_page > 0

    def get_progress_percentage(self) -> float:
        """
        Get reading progress as a percentage.
        
        Returns:
            Progress percentage (0.0 to 100.0)
        """
        if self.total_pages <= 0:
            return 0.0

        if self.finished_book:
            return 100.0

        return min(100.0, (self.current_page / self.total_pages) * 100.0)

    def get_reading_status_text(self) -> str:
        """
        Get human-readable reading status.
        
        Returns:
            Status description
        """
        if self.finished_book:
            return "Finished"
        elif self.current_page > 0:
            return f"Reading (page {self.current_page}/{self.total_pages})"
        else:
            return "Not started"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "user_id": self.user_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "current_page": self.current_page,
            "total_pages": self.total_pages,
            "reading_status": self.reading_status.value,
            "finished_book": self.finished_book,
            "last_updated": self.last_updated.isoformat(),
            "reading_speed": self.reading_speed,
            "time_spent": self.time_spent,
            "bookmarks": self.bookmarks,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'UserProgress':
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            document_id=data["document_id"],
            document_name=data["document_name"],
            current_page=data.get("current_page", 0),
            total_pages=data.get("total_pages", 0),
            reading_status=ReadingStatus(data.get("reading_status", "not_started")),
            finished_book=data.get("finished_book", False),
            last_updated=datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat())),
            reading_speed=data.get("reading_speed"),
            time_spent=data.get("time_spent"),
            bookmarks=data.get("bookmarks", {}),
            notes=data.get("notes", {})
        )


@dataclass
class DocumentMetadata:
    """
    Metadata for a document being processed.
    """
    document_id: str
    document_name: str
    file_path: str
    file_type: str  # 'pdf', 'epub', etc.
    total_pages: int
    total_chunks: int = 0
    processing_time: float | None = None
    created_at: datetime = field(default_factory=datetime.now)

    # Document-specific metadata
    author: str | None = None
    genre: str | None = None
    language: str = "en"
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat(),
            "author": self.author,
            "genre": self.genre,
            "language": self.language,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'DocumentMetadata':
        """Create from dictionary."""
        return cls(
            document_id=data["document_id"],
            document_name=data["document_name"],
            file_path=data["file_path"],
            file_type=data["file_type"],
            total_pages=data["total_pages"],
            total_chunks=data.get("total_chunks", 0),
            processing_time=data.get("processing_time"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            author=data.get("author"),
            genre=data.get("genre"),
            language=data.get("language", "en"),
            description=data.get("description")
        )
