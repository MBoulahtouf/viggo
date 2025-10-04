"""
User Progress Service for Viggo

This service manages user reading progress and integrates with the spoiler protection system.
"""

from typing import Optional, Dict, List, Any
import json
import os
from pathlib import Path

from viggo.core.models.user_progress import UserProgress, DocumentMetadata, ReadingStatus


class UserProgressService:
    """
    Service for managing user reading progress and spoiler protection.
    """
    
    def __init__(self, storage_path: str = "data/user_progress"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for active sessions
        self.active_progress: Dict[str, UserProgress] = {}
        self.document_metadata: Dict[str, DocumentMetadata] = {}
    
    def create_user_progress(self, user_id: str, document_id: str, document_name: str, 
                           total_pages: int, current_page: int = 0, 
                           finished_book: bool = False) -> UserProgress:
        """
        Create or update user progress for a document.
        
        Args:
            user_id: Unique user identifier
            document_id: Unique document identifier
            document_name: Human-readable document name
            total_pages: Total number of pages in the document
            current_page: Current page the user is on
            finished_book: Whether the user has finished the book
            
        Returns:
            UserProgress object
        """
        progress = UserProgress(
            user_id=user_id,
            document_id=document_id,
            document_name=document_name,
            current_page=current_page,
            total_pages=total_pages,
            finished_book=finished_book
        )
        
        # Store in cache and persistent storage
        self.active_progress[f"{user_id}_{document_id}"] = progress
        self._save_progress(progress)
        
        return progress
    
    def update_user_progress(self, user_id: str, document_id: str, 
                           page: int, finished_book: bool = False) -> Optional[UserProgress]:
        """
        Update user's reading progress.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            page: New page number
            finished_book: Whether the user has finished the book
            
        Returns:
            Updated UserProgress object, or None if not found
        """
        progress_key = f"{user_id}_{document_id}"
        
        if progress_key in self.active_progress:
            progress = self.active_progress[progress_key]
            progress.update_progress(page, finished_book)
            self._save_progress(progress)
            return progress
        
        # Try to load from storage
        progress = self._load_progress(user_id, document_id)
        if progress:
            progress.update_progress(page, finished_book)
            self.active_progress[progress_key] = progress
            self._save_progress(progress)
            return progress
        
        return None
    
    def get_user_progress(self, user_id: str, document_id: str) -> Optional[UserProgress]:
        """
        Get user's current progress for a document.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            
        Returns:
            UserProgress object, or None if not found
        """
        progress_key = f"{user_id}_{document_id}"
        
        # Check cache first
        if progress_key in self.active_progress:
            return self.active_progress[progress_key]
        
        # Load from storage
        progress = self._load_progress(user_id, document_id)
        if progress:
            self.active_progress[progress_key] = progress
        
        return progress
    
    def get_spoiler_limit(self, user_id: str, document_id: str) -> Optional[int]:
        """
        Get the page limit for spoiler protection for a user and document.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            
        Returns:
            Page number limit, or None if no spoiler protection needed
        """
        progress = self.get_user_progress(user_id, document_id)
        if progress:
            return progress.get_spoiler_limit()
        return None
    
    def is_spoiler_protected(self, user_id: str, document_id: str) -> bool:
        """
        Check if spoiler protection is active for a user and document.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            
        Returns:
            True if spoiler protection is active
        """
        progress = self.get_user_progress(user_id, document_id)
        if progress:
            return progress.is_spoiler_protected()
        return False
    
    def create_document_metadata(self, document_id: str, document_name: str, 
                               file_path: str, file_type: str, total_pages: int,
                               total_chunks: int = 0, processing_time: Optional[float] = None,
                               author: Optional[str] = None, genre: Optional[str] = None) -> DocumentMetadata:
        """
        Create document metadata.
        
        Args:
            document_id: Unique document identifier
            document_name: Human-readable document name
            file_path: Path to the document file
            file_type: Type of document (pdf, epub, etc.)
            total_pages: Total number of pages
            total_chunks: Total number of chunks created
            processing_time: Time taken to process the document
            author: Document author
            genre: Document genre
            
        Returns:
            DocumentMetadata object
        """
        metadata = DocumentMetadata(
            document_id=document_id,
            document_name=document_name,
            file_path=file_path,
            file_type=file_type,
            total_pages=total_pages,
            total_chunks=total_chunks,
            processing_time=processing_time,
            author=author,
            genre=genre
        )
        
        self.document_metadata[document_id] = metadata
        self._save_document_metadata(metadata)
        
        return metadata
    
    def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        Get document metadata.
        
        Args:
            document_id: Document identifier
            
        Returns:
            DocumentMetadata object, or None if not found
        """
        if document_id in self.document_metadata:
            return self.document_metadata[document_id]
        
        # Try to load from storage
        metadata = self._load_document_metadata(document_id)
        if metadata:
            self.document_metadata[document_id] = metadata
        
        return metadata
    
    def get_user_documents(self, user_id: str) -> List[UserProgress]:
        """
        Get all documents for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of UserProgress objects
        """
        user_documents = []
        
        # Check cache
        for progress_key, progress in self.active_progress.items():
            if progress.user_id == user_id:
                user_documents.append(progress)
        
        # Check storage for additional documents
        user_dir = self.storage_path / user_id
        if user_dir.exists():
            for file_path in user_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    progress = UserProgress.from_dict(data)
                    if progress not in user_documents:
                        user_documents.append(progress)
                except Exception as e:
                    print(f"Error loading progress from {file_path}: {e}")
        
        return user_documents
    
    def delete_user_progress(self, user_id: str, document_id: str) -> bool:
        """
        Delete user progress for a document.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            
        Returns:
            True if deleted, False if not found
        """
        progress_key = f"{user_id}_{document_id}"
        
        # Remove from cache
        if progress_key in self.active_progress:
            del self.active_progress[progress_key]
        
        # Remove from storage
        progress_file = self.storage_path / user_id / f"{document_id}.json"
        if progress_file.exists():
            progress_file.unlink()
            return True
        
        return False
    
    def _save_progress(self, progress: UserProgress) -> None:
        """Save user progress to persistent storage."""
        user_dir = self.storage_path / progress.user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        progress_file = user_dir / f"{progress.document_id}.json"
        with open(progress_file, 'w') as f:
            json.dump(progress.to_dict(), f, indent=2)
    
    def _load_progress(self, user_id: str, document_id: str) -> Optional[UserProgress]:
        """Load user progress from persistent storage."""
        progress_file = self.storage_path / user_id / f"{document_id}.json"
        
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    data = json.load(f)
                return UserProgress.from_dict(data)
            except Exception as e:
                print(f"Error loading progress from {progress_file}: {e}")
        
        return None
    
    def _save_document_metadata(self, metadata: DocumentMetadata) -> None:
        """Save document metadata to persistent storage."""
        metadata_file = self.storage_path / f"metadata_{metadata.document_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def _load_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """Load document metadata from persistent storage."""
        metadata_file = self.storage_path / f"metadata_{document_id}.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                return DocumentMetadata.from_dict(data)
            except Exception as e:
                print(f"Error loading metadata from {metadata_file}: {e}")
        
        return None
    
    def get_progress_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of user's reading progress.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with progress summary
        """
        documents = self.get_user_documents(user_id)
        
        total_documents = len(documents)
        finished_documents = sum(1 for doc in documents if doc.finished_book)
        in_progress_documents = sum(1 for doc in documents if doc.reading_status == ReadingStatus.IN_PROGRESS)
        
        return {
            "user_id": user_id,
            "total_documents": total_documents,
            "finished_documents": finished_documents,
            "in_progress_documents": in_progress_documents,
            "not_started_documents": total_documents - finished_documents - in_progress_documents,
            "documents": [
                {
                    "document_id": doc.document_id,
                    "document_name": doc.document_name,
                    "current_page": doc.current_page,
                    "total_pages": doc.total_pages,
                    "progress_percentage": doc.get_progress_percentage(),
                    "reading_status": doc.get_reading_status_text(),
                    "finished_book": doc.finished_book,
                    "last_updated": doc.last_updated.isoformat()
                }
                for doc in documents
            ]
        }
