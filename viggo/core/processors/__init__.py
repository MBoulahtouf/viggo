# viggo/core/processors/__init__.py
from .base import DocumentProcessor
from .pdf_processor import PDFProcessor
from .epub_processor import EPUBProcessor
from .factory import DocumentProcessorFactory

__all__ = ["DocumentProcessor", "PDFProcessor", "EPUBProcessor", "DocumentProcessorFactory"]
