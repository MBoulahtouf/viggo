"""
Concrete implementations of document processors following SOLID principles.
"""

import os
from typing import List, Dict, Any, Optional
from viggo.core.services.interfaces.document_processor import (
    DocumentProcessor, DocumentProcessorFactory, DocumentMetadata, DocumentPage
)
from viggo.core.processors import DocumentProcessorFactory as LegacyFactory


class PDFDocumentProcessor(DocumentProcessor):
    """Concrete implementation for PDF document processing."""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def can_process(self, file_path: str) -> bool:
        """Check if this processor can handle the given file."""
        return any(file_path.lower().endswith(ext) for ext in self.supported_extensions)
    
    def process_document(self, file_path: str) -> List[DocumentPage]:
        """Process a PDF document and return its pages."""
        # Use existing PDF processor implementation
        legacy_factory = LegacyFactory()
        processor = legacy_factory.get_processor(file_path)
        
        if not processor:
            raise ValueError(f"Cannot process PDF file: {file_path}")
        
        # Get pages data from existing implementation
        pages_data = processor.process_document(file_path)
        
        # Convert to new format
        document_pages = []
        for page_data in pages_data:
            page = DocumentPage(
                page_number=page_data.get('page', 0),
                content=page_data.get('content', ''),
                metadata=page_data.get('metadata', {})
            )
            document_pages.append(page)
        
        return document_pages
    
    def get_pdf_info(self, file_path: str) -> Dict[str, Any]:
        """Get PDF-specific information."""
        # Use existing PDF processor implementation
        legacy_factory = LegacyFactory()
        processor = legacy_factory.get_processor(file_path)
        
        if not processor:
            raise ValueError(f"Cannot process PDF file: {file_path}")
        
        # Get PDF info from existing implementation
        return processor.get_pdf_info(file_path)
    
    def get_document_metadata(self, file_path: str) -> DocumentMetadata:
        """Get metadata for a PDF document."""
        import time
        
        # Use existing implementation
        legacy_factory = LegacyFactory()
        processor = legacy_factory.get_processor(file_path)
        
        if not processor:
            raise ValueError(f"Cannot process PDF file: {file_path}")
        
        # Get file info
        file_info = processor.get_file_info(file_path)
        
        return DocumentMetadata(
            filename=os.path.basename(file_path),
            file_path=file_path,
            file_type='pdf',
            page_count=file_info.get('page_count', 0),
            word_count=file_info.get('word_count', 0),
            processing_timestamp=time.time(),
            additional_metadata=file_info
        )


class EPUBDocumentProcessor(DocumentProcessor):
    """Concrete implementation for EPUB document processing."""
    
    def __init__(self):
        self.supported_extensions = ['.epub']
    
    def can_process(self, file_path: str) -> bool:
        """Check if this processor can handle the given file."""
        return any(file_path.lower().endswith(ext) for ext in self.supported_extensions)
    
    def process_document(self, file_path: str) -> List[DocumentPage]:
        """Process an EPUB document and return its pages."""
        # Use existing EPUB processor implementation
        legacy_factory = LegacyFactory()
        processor = legacy_factory.get_processor(file_path)
        
        if not processor:
            raise ValueError(f"Cannot process EPUB file: {file_path}")
        
        # Get pages data from existing implementation
        pages_data = processor.process_document(file_path)
        
        # Convert to new format
        document_pages = []
        for page_data in pages_data:
            page = DocumentPage(
                page_number=page_data.get('page', 0),
                content=page_data.get('content', ''),
                metadata=page_data.get('metadata', {})
            )
            document_pages.append(page)
        
        return document_pages
    
    def get_document_metadata(self, file_path: str) -> DocumentMetadata:
        """Get metadata for an EPUB document."""
        import time
        
        # Use existing implementation
        legacy_factory = LegacyFactory()
        processor = legacy_factory.get_processor(file_path)
        
        if not processor:
            raise ValueError(f"Cannot process EPUB file: {file_path}")
        
        # Get EPUB info
        epub_info = processor.get_epub_info(file_path)
        
        return DocumentMetadata(
            filename=os.path.basename(file_path),
            file_path=file_path,
            file_type='epub',
            page_count=epub_info.get('page_count', 0),
            word_count=epub_info.get('word_count', 0),
            processing_timestamp=time.time(),
            additional_metadata=epub_info
        )


class ConcreteDocumentProcessorFactory(DocumentProcessorFactory):
    """Concrete implementation of document processor factory."""
    
    def __init__(self):
        self.processors = [
            PDFDocumentProcessor(),
            EPUBDocumentProcessor()
        ]
    
    def get_processor(self, file_path: str) -> Optional[DocumentProcessor]:
        """Get appropriate processor for file."""
        for processor in self.processors:
            if processor.can_process(file_path):
                return processor
        return None
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        extensions = []
        for processor in self.processors:
            extensions.extend(processor.supported_extensions)
        return extensions
    
    def is_supported(self, file_path: str) -> bool:
        """Check if file format is supported."""
        return self.get_processor(file_path) is not None
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information using the appropriate processor."""
        processor = self.get_processor(file_path)
        if not processor:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Get metadata from the processor
        metadata = processor.get_document_metadata(file_path)
        
        return {
            'filename': metadata.filename,
            'file_type': metadata.file_type,
            'page_count': metadata.page_count,
            'word_count': metadata.word_count,
            'processing_timestamp': metadata.processing_timestamp,
            'additional_metadata': metadata.additional_metadata
        }