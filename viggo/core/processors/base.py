# viggo/core/processors/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os
from pathlib import Path


class DocumentProcessor(ABC):
    """
    Abstract base class for document processors.
    
    Each document format (PDF, EPUB, DOCX, etc.) should have its own processor
    that implements this interface.
    """
    
    def __init__(self):
        self.supported_extensions = self._get_supported_extensions()
    
    @abstractmethod
    def _get_supported_extensions(self) -> List[str]:
        """Return list of file extensions this processor supports."""
        pass
    
    @abstractmethod
    def extract_text(self, file_path: str) -> List[Dict]:
        """
        Extract text content from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of dictionaries with 'page' and 'content' keys
            For non-paginated formats, page numbers can be sequential or based on chapters/sections
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported or corrupted
        """
        pass
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate that the file exists and has a supported extension.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid, False otherwise
        """
        if not os.path.exists(file_path):
            return False
        
        file_extension = Path(file_path).suffix.lower()
        return file_extension in self.supported_extensions
    
    def get_file_info(self, file_path: str) -> Dict:
        """
        Get basic information about the file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_path_obj = Path(file_path)
        return {
            "filename": file_path_obj.name,
            "extension": file_path_obj.suffix.lower(),
            "size_bytes": os.path.getsize(file_path),
            "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2)
        }
    
    def process_document(self, file_path: str) -> List[Dict]:
        """
        Main entry point for processing a document.
        Validates the file and extracts text content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of dictionaries with 'page' and 'content' keys
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported or corrupted
        """
        if not self.validate_file(file_path):
            file_extension = Path(file_path).suffix.lower()
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {', '.join(self.supported_extensions)}"
            )
        
        return self.extract_text(file_path)
