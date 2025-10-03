# viggo/core/processors/factory.py
from typing import Optional, List
from pathlib import Path
from .base import DocumentProcessor
from .pdf_processor import PDFProcessor
from .epub_processor import EPUBProcessor


class DocumentProcessorFactory:
    """
    Factory class for creating appropriate document processors based on file type.
    """
    
    def __init__(self):
        self._processors = {
            ".pdf": PDFProcessor(),
            ".epub": EPUBProcessor(),
        }
    
    def get_processor(self, file_path: str) -> Optional[DocumentProcessor]:
        """
        Get the appropriate processor for a given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            DocumentProcessor instance or None if format not supported
        """
        file_extension = Path(file_path).suffix.lower()
        return self._processors.get(file_extension)
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of all supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        return list(self._processors.keys())
    
    def is_supported(self, file_path: str) -> bool:
        """
        Check if a file format is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if format is supported, False otherwise
        """
        return self.get_processor(file_path) is not None
    
    def process_document(self, file_path: str) -> List[dict]:
        """
        Process a document using the appropriate processor.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of dictionaries with 'page' and 'content' keys
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        processor = self.get_processor(file_path)
        if not processor:
            file_extension = Path(file_path).suffix.lower()
            supported_formats = self.get_supported_extensions()
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
        
        return processor.process_document(file_path)
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get file information using the appropriate processor.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        processor = self.get_processor(file_path)
        if not processor:
            file_extension = Path(file_path).suffix.lower()
            supported_formats = self.get_supported_extensions()
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
        
        return processor.get_file_info(file_path)
