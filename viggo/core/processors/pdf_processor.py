# viggo/core/processors/pdf_processor.py
from typing import List, Dict
from pypdf import PdfReader
from .base import DocumentProcessor


class PDFProcessor(DocumentProcessor):
    """
    Processor for PDF documents using pypdf library.
    """
    
    def _get_supported_extensions(self) -> List[str]:
        """Return list of supported PDF file extensions."""
        return [".pdf"]
    
    def extract_text(self, file_path: str) -> List[Dict]:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of dictionaries with 'page' and 'content' keys
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If PDF is corrupted or cannot be read
        """
        try:
            reader = PdfReader(file_path)
            pages_data = []
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():  # Only include pages with actual content
                    pages_data.append({
                        "page": i + 1,
                        "content": text.strip()
                    })
                    print(f"[DEBUG] Extracted text from PDF page {i+1}: {text[:200]}...")
            
            if not pages_data:
                raise ValueError("No readable text content found in PDF")
            
            return pages_data
            
        except Exception as e:
            if "File not found" in str(e) or "No such file" in str(e):
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            else:
                raise ValueError(f"Error reading PDF file {file_path}: {str(e)}")
    
    def get_pdf_info(self, file_path: str) -> Dict:
        """
        Get additional PDF-specific information.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF metadata
        """
        try:
            reader = PdfReader(file_path)
            metadata = reader.metadata
            
            return {
                **self.get_file_info(file_path),
                "num_pages": len(reader.pages),
                "title": metadata.get("/Title", "") if metadata else "",
                "author": metadata.get("/Author", "") if metadata else "",
                "subject": metadata.get("/Subject", "") if metadata else "",
                "creator": metadata.get("/Creator", "") if metadata else "",
                "producer": metadata.get("/Producer", "") if metadata else "",
                "creation_date": str(metadata.get("/CreationDate", "")) if metadata else "",
                "modification_date": str(metadata.get("/ModDate", "")) if metadata else ""
            }
        except Exception as e:
            # Return basic file info if PDF metadata cannot be read
            return self.get_file_info(file_path)
