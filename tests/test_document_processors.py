# tests/test_document_processors.py
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from viggo.core.processors import DocumentProcessorFactory, PDFProcessor, EPUBProcessor
from viggo.core.processors.base import DocumentProcessor


class TestDocumentProcessorFactory:
    """Test the DocumentProcessorFactory class."""
    
    def test_get_processor_pdf(self):
        """Test getting PDF processor."""
        factory = DocumentProcessorFactory()
        processor = factory.get_processor("test.pdf")
        assert isinstance(processor, PDFProcessor)
    
    def test_get_processor_epub(self):
        """Test getting EPUB processor."""
        factory = DocumentProcessorFactory()
        processor = factory.get_processor("test.epub")
        assert isinstance(processor, EPUBProcessor)
    
    def test_get_processor_unsupported(self):
        """Test getting processor for unsupported format."""
        factory = DocumentProcessorFactory()
        processor = factory.get_processor("test.txt")
        assert processor is None
    
    def test_get_supported_extensions(self):
        """Test getting supported extensions."""
        factory = DocumentProcessorFactory()
        extensions = factory.get_supported_extensions()
        assert ".pdf" in extensions
        assert ".epub" in extensions
    
    def test_is_supported(self):
        """Test checking if format is supported."""
        factory = DocumentProcessorFactory()
        assert factory.is_supported("test.pdf") is True
        assert factory.is_supported("test.epub") is True
        assert factory.is_supported("test.txt") is False
    
    def test_process_document_unsupported_format(self):
        """Test processing unsupported document format."""
        factory = DocumentProcessorFactory()
        with pytest.raises(ValueError, match="Unsupported file format"):
            factory.process_document("test.txt")


class TestPDFProcessor:
    """Test the PDFProcessor class."""
    
    def test_supported_extensions(self):
        """Test supported extensions."""
        processor = PDFProcessor()
        assert processor.supported_extensions == [".pdf"]
    
    def test_validate_file_valid(self):
        """Test validating valid PDF file."""
        processor = PDFProcessor()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"dummy pdf content")
            tmp_path = tmp.name
        
        try:
            assert processor.validate_file(tmp_path) is True
        finally:
            os.unlink(tmp_path)
    
    def test_validate_file_invalid_extension(self):
        """Test validating file with invalid extension."""
        processor = PDFProcessor()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"dummy content")
            tmp_path = tmp.name
        
        try:
            assert processor.validate_file(tmp_path) is False
        finally:
            os.unlink(tmp_path)
    
    def test_validate_file_nonexistent(self):
        """Test validating nonexistent file."""
        processor = PDFProcessor()
        assert processor.validate_file("nonexistent.pdf") is False
    
    @patch('viggo.core.processors.pdf_processor.PdfReader')
    def test_extract_text_success(self, mock_pdf_reader):
        """Test successful text extraction from PDF."""
        # Mock PDF reader and pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        mock_reader = Mock()
        mock_reader.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader
        
        processor = PDFProcessor()
        result = processor.extract_text("test.pdf")
        
        assert len(result) == 2
        assert result[0]["page"] == 1
        assert result[0]["content"] == "Page 1 content"
        assert result[1]["page"] == 2
        assert result[1]["content"] == "Page 2 content"
    
    @patch('viggo.core.processors.pdf_processor.PdfReader')
    def test_extract_text_empty_pages(self, mock_pdf_reader):
        """Test text extraction with empty pages."""
        # Mock PDF reader with empty pages
        mock_page = Mock()
        mock_page.extract_text.return_value = ""
        
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        processor = PDFProcessor()
        
        with pytest.raises(ValueError, match="No readable text content found"):
            processor.extract_text("test.pdf")
    
    @patch('viggo.core.processors.pdf_processor.PdfReader')
    def test_extract_text_file_not_found(self, mock_pdf_reader):
        """Test text extraction with file not found."""
        mock_pdf_reader.side_effect = FileNotFoundError("File not found")
        
        processor = PDFProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.extract_text("nonexistent.pdf")


class TestEPUBProcessor:
    """Test the EPUBProcessor class."""
    
    def test_supported_extensions(self):
        """Test supported extensions."""
        processor = EPUBProcessor()
        assert processor.supported_extensions == [".epub"]
    
    def test_validate_file_valid(self):
        """Test validating valid EPUB file."""
        processor = EPUBProcessor()
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
            tmp.write(b"dummy epub content")
            tmp_path = tmp.name
        
        try:
            assert processor.validate_file(tmp_path) is True
        finally:
            os.unlink(tmp_path)
    
    def test_validate_file_invalid_extension(self):
        """Test validating file with invalid extension."""
        processor = EPUBProcessor()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"dummy content")
            tmp_path = tmp.name
        
        try:
            assert processor.validate_file(tmp_path) is False
        finally:
            os.unlink(tmp_path)
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        processor = EPUBProcessor()
        
        # Test removing excessive whitespace
        dirty_text = "  This   is    a   test  \n\n\n  with   spaces  "
        clean_text = processor._clean_text(dirty_text)
        assert clean_text == "This is a test\nwith spaces"  # Updated to match new behavior
        
        # Test removing standalone page numbers
        text_with_page_numbers = "Chapter 1\n123\nThis is content\n456\nMore content"
        clean_text = processor._clean_text(text_with_page_numbers)
        # The regex only removes standalone numbers, not numbers within text
        assert "This is content" in clean_text
        assert "More content" in clean_text
    
    def test_extract_chapter_title(self):
        """Test chapter title extraction."""
        processor = EPUBProcessor()
        
        # Test normal chapter name
        title = processor._extract_chapter_title("chapter_01.xhtml")
        assert title == "chapter 01"
        
        # Test chapter with dashes
        title = processor._extract_chapter_title("chapter-02.html")
        assert title == "chapter 02"
        
        # Test empty input
        title = processor._extract_chapter_title("")
        assert title == ""
    
    @patch('viggo.core.processors.epub_processor.epub.read_epub')
    def test_extract_text_success(self, mock_read_epub):
        """Test successful text extraction from EPUB."""
        # Mock EPUB book and items
        mock_item = Mock()
        mock_item.get_type.return_value = 9  # ITEM_DOCUMENT
        mock_item.get_content.return_value = b"<html><body>Chapter content with enough text to pass minimum length requirement</body></html>"
        mock_item.get_name.return_value = "chapter_01.xhtml"
        mock_item.get_id.return_value = "chapter_01"
        
        mock_book = Mock()
        mock_book.get_items.return_value = [mock_item]
        mock_book.spine = [("chapter_01", None)]  # Mock spine
        mock_book.toc = []  # Mock empty TOC
        mock_book.get_item_by_id.return_value = mock_item  # Mock item lookup
        mock_read_epub.return_value = mock_book
        
        processor = EPUBProcessor()
        result = processor.extract_text("test.epub")
        
        assert len(result) == 1
        assert result[0]["page"] == 1
        assert "Chapter content" in result[0]["content"]
        assert result[0]["chapter_title"] == "chapter 01"
        assert result[0]["item_id"] == "chapter_01"
        assert "word_count" in result[0]
        assert "char_count" in result[0]
    
    @patch('viggo.core.processors.epub_processor.epub.read_epub')
    def test_extract_text_no_content(self, mock_read_epub):
        """Test text extraction with no readable content."""
        # Mock EPUB book with no document items
        mock_book = Mock()
        mock_book.get_items.return_value = []
        mock_book.spine = []  # Mock empty spine
        mock_book.toc = []  # Mock empty TOC
        mock_read_epub.return_value = mock_book
        
        processor = EPUBProcessor()
        
        with pytest.raises(ValueError, match="No readable text content found"):
            processor.extract_text("test.epub")
    
    @patch('viggo.core.processors.epub_processor.epub.read_epub')
    def test_extract_text_file_not_found(self, mock_read_epub):
        """Test text extraction with file not found."""
        mock_read_epub.side_effect = FileNotFoundError("File not found")
        
        processor = EPUBProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.extract_text("nonexistent.epub")


class TestDocumentProcessorBase:
    """Test the base DocumentProcessor class."""
    
    def test_get_file_info(self):
        """Test getting file information."""
        processor = PDFProcessor()  # Use concrete implementation
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"dummy content")
            tmp_path = tmp.name
        
        try:
            file_info = processor.get_file_info(tmp_path)
            
            assert "filename" in file_info
            assert "extension" in file_info
            assert "size_bytes" in file_info
            assert "size_mb" in file_info
            assert file_info["extension"] == ".pdf"
            assert file_info["size_bytes"] > 0
        finally:
            os.unlink(tmp_path)
    
    def test_get_file_info_nonexistent(self):
        """Test getting file info for nonexistent file."""
        processor = PDFProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.get_file_info("nonexistent.pdf")
    
    def test_process_document_validation_failure(self):
        """Test processing document with validation failure."""
        processor = PDFProcessor()
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"dummy content")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                processor.process_document(tmp_path)
        finally:
            os.unlink(tmp_path)
