# viggo/core/processors/epub_processor.py
from typing import List, Dict, Optional, Tuple
import re
import os
from pathlib import Path
from ebooklib import epub
from bs4 import BeautifulSoup
from .base import DocumentProcessor


class EPUBProcessor(DocumentProcessor):
    """
    Processor for EPUB documents using ebooklib library.
    """
    
    def _get_supported_extensions(self) -> List[str]:
        """Return list of supported EPUB file extensions."""
        return [".epub"]
    
    def extract_text(self, file_path: str) -> List[Dict]:
        """
        Extract text content from an EPUB file with enhanced processing.
        
        Args:
            file_path: Path to the EPUB file
            
        Returns:
            List of dictionaries with 'page', 'content', 'chapter_title', and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If EPUB is corrupted or cannot be read
        """
        try:
            book = epub.read_epub(file_path)
            pages_data = []
            
            # Get table of contents for better chapter organization
            toc = self._extract_table_of_contents(book)
            
            # Get all items that contain text content, ordered by spine
            spine_items = self._get_ordered_spine_items(book)
            
            for page_counter, item in enumerate(spine_items, 1):
                if item.get_type() == 9:  # ITEM_DOCUMENT = 9
                    # Extract text from HTML content with enhanced processing
                    content, chapter_title, chapter_metadata = self._extract_enhanced_content(item, toc)
                    
                    if content and content.strip():
                        # Clean up the text with advanced filtering
                        cleaned_content = self._clean_text_advanced(content)
                        
                        if cleaned_content and len(cleaned_content.strip()) > 50:  # Minimum content length
                            pages_data.append({
                                "page": page_counter,
                                "content": cleaned_content,
                                "chapter_title": chapter_title,
                                "item_id": item.get_id(),
                                "chapter_metadata": chapter_metadata,
                                "word_count": len(cleaned_content.split()),
                                "char_count": len(cleaned_content)
                            })
                            print(f"[DEBUG] Extracted text from EPUB chapter {page_counter} ({chapter_title}): {cleaned_content[:200]}...")
            
            if not pages_data:
                raise ValueError("No readable text content found in EPUB")
            
            print(f"[DEBUG] Successfully processed {len(pages_data)} chapters from EPUB")
            return pages_data
            
        except Exception as e:
            if "File not found" in str(e) or "No such file" in str(e):
                raise FileNotFoundError(f"EPUB file not found: {file_path}")
            else:
                raise ValueError(f"Error reading EPUB file {file_path}: {str(e)}")
    
    def _extract_table_of_contents(self, book: epub.EpubBook) -> Dict[str, str]:
        """
        Extract table of contents for better chapter organization.
        
        Args:
            book: EPUB book object
            
        Returns:
            Dictionary mapping item IDs to chapter titles
        """
        toc = {}
        try:
            # Try to get NCX table of contents
            if hasattr(book, 'toc') and book.toc:
                for item in book.toc:
                    if hasattr(item, 'href') and hasattr(item, 'title'):
                        # Extract item ID from href
                        item_id = item.href.split('#')[0] if '#' in item.href else item.href
                        toc[item_id] = item.title
        except Exception as e:
            print(f"[DEBUG] Could not extract TOC: {e}")
        
        return toc
    
    def _get_ordered_spine_items(self, book: epub.EpubBook) -> List:
        """
        Get items in reading order from spine.
        
        Args:
            book: EPUB book object
            
        Returns:
            List of items in reading order
        """
        spine_items = []
        try:
            # Get items in spine order (reading order)
            for item_id, _ in book.spine:
                item = book.get_item_by_id(item_id)
                if item:
                    spine_items.append(item)
        except Exception as e:
            print(f"[DEBUG] Could not get spine items, falling back to all items: {e}")
            # Fallback to all document items
            spine_items = [item for item in book.get_items() if item.get_type() == 9]
        
        return spine_items
    
    def _extract_enhanced_content(self, item, toc: Dict[str, str]) -> Tuple[str, str, Dict]:
        """
        Extract content with enhanced processing and metadata.
        
        Args:
            item: EPUB item object
            toc: Table of contents mapping
            
        Returns:
            Tuple of (content, chapter_title, metadata)
        """
        # Extract text from HTML content
        content = self._extract_text_from_html(item.get_content())
        
        # Get chapter title from TOC or item name
        item_id = item.get_id()
        chapter_title = toc.get(item_id, self._extract_chapter_title(item.get_name()))
        
        # Extract additional metadata
        chapter_metadata = {
            "item_id": item_id,
            "item_name": item.get_name(),
            "mime_type": item.get_type(),
            "size": len(item.get_content()) if item.get_content() else 0
        }
        
        return content, chapter_title, chapter_metadata
    
    def _extract_text_from_html(self, html_content: bytes) -> str:
        """
        Extract text content from HTML bytes with enhanced processing.
        
        Args:
            html_content: HTML content as bytes
            
        Returns:
            Extracted text content
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
            
            # Remove elements with common navigation/UI classes
            for element in soup.find_all(class_=re.compile(r'(nav|menu|header|footer|sidebar|toc)', re.I)):
                element.decompose()
            
            # Extract text with better formatting
            # Preserve paragraph breaks
            for p in soup.find_all('p'):
                p.append('\n\n')
            
            # Preserve line breaks
            for br in soup.find_all('br'):
                br.replace_with('\n')
            
            # Get text content
            text = soup.get_text()
            return text
            
        except Exception as e:
            print(f"[WARNING] Error parsing HTML content: {e}")
            # Fallback: try to decode as text directly
            try:
                return html_content.decode('utf-8')
            except UnicodeDecodeError:
                return html_content.decode('latin-1', errors='ignore')
    
    def _clean_text_advanced(self, text: str) -> str:
        """
        Advanced text cleaning for EPUB content.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove common EPUB artifacts and navigation elements
        text = self._remove_epub_artifacts(text)
        
        # Normalize whitespace while preserving paragraph structure
        text = self._normalize_whitespace(text)
        
        # Remove empty lines and normalize
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _remove_epub_artifacts(self, text: str) -> str:
        """
        Remove common EPUB artifacts and navigation elements.
        
        Args:
            text: Raw text
            
        Returns:
            Text with artifacts removed
        """
        # Remove standalone page numbers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove chapter markers
        text = re.sub(r'^\s*Chapter\s+\d+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^\s*Part\s+\d+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove common navigation text
        nav_patterns = [
            r'^\s*(Table of Contents|Contents|Index|Bibliography|References)\s*$',
            r'^\s*(Previous|Next|Back|Continue)\s*$',
            r'^\s*(Page \d+ of \d+)\s*$',
            r'^\s*(\d+)\s*$',  # Standalone numbers
        ]
        
        for pattern in nav_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)  # Normalize ellipses
        text = re.sub(r'[-]{3,}', '---', text)  # Normalize dashes
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace while preserving paragraph structure.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Preserve paragraph breaks (double newlines)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Normalize single newlines to spaces (within paragraphs)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        
        # Remove excessive spaces
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """
        Legacy text cleaning method for backward compatibility.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        return self._clean_text_advanced(text)
    
    def _extract_chapter_title(self, item_name: str) -> str:
        """
        Extract chapter title from item name.
        
        Args:
            item_name: EPUB item name/path
            
        Returns:
            Chapter title or empty string
        """
        if not item_name:
            return ""
        
        # Extract filename without extension
        import os
        filename = os.path.basename(item_name)
        title = os.path.splitext(filename)[0]
        
        # Clean up the title
        title = re.sub(r'[_-]', ' ', title)
        title = re.sub(r'\s+', ' ', title)
        
        return title.strip()
    
    def get_epub_info(self, file_path: str) -> Dict:
        """
        Get comprehensive EPUB-specific information.
        
        Args:
            file_path: Path to the EPUB file
            
        Returns:
            Dictionary with EPUB metadata
        """
        try:
            book = epub.read_epub(file_path)
            metadata = book.get_metadata('DC', {})
            
            # Extract common metadata fields with better handling
            title = self._extract_metadata_field(metadata, 'title')
            author = self._extract_metadata_field(metadata, 'creator')
            language = self._extract_metadata_field(metadata, 'language')
            publisher = self._extract_metadata_field(metadata, 'publisher')
            publication_date = self._extract_metadata_field(metadata, 'date')
            subject = self._extract_metadata_field(metadata, 'subject')
            description = self._extract_metadata_field(metadata, 'description')
            rights = self._extract_metadata_field(metadata, 'rights')
            
            # Get additional metadata
            identifier = self._extract_metadata_field(metadata, 'identifier')
            format_info = self._extract_metadata_field(metadata, 'format')
            
            # Count chapters/sections and get content statistics
            spine_items = self._get_ordered_spine_items(book)
            num_chapters = len(spine_items)
            
            # Calculate estimated word count
            estimated_words = 0
            for item in spine_items:
                if item.get_type() == 9:
                    content = self._extract_text_from_html(item.get_content())
                    cleaned = self._clean_text_advanced(content)
                    estimated_words += len(cleaned.split())
            
            # Get table of contents info
            toc = self._extract_table_of_contents(book)
            
            return {
                **self.get_file_info(file_path),
                "num_chapters": num_chapters,
                "estimated_word_count": estimated_words,
                "title": title,
                "author": author,
                "language": language,
                "publisher": publisher,
                "publication_date": publication_date,
                "subject": subject,
                "description": description,
                "rights": rights,
                "identifier": identifier,
                "format": format_info,
                "toc_entries": len(toc),
                "has_toc": len(toc) > 0,
                "epub_version": getattr(book, 'version', 'Unknown')
            }
            
        except Exception as e:
            print(f"[WARNING] Error reading EPUB metadata: {e}")
            # Return basic file info if EPUB metadata cannot be read
            return self.get_file_info(file_path)
    
    def _extract_metadata_field(self, metadata: Dict, field_name: str) -> str:
        """
        Safely extract metadata field with proper handling.
        
        Args:
            metadata: Metadata dictionary
            field_name: Field name to extract
            
        Returns:
            Extracted field value or empty string
        """
        try:
            if field_name in metadata and metadata[field_name]:
                value = metadata[field_name]
                if isinstance(value, list) and value:
                    return str(value[0])
                elif isinstance(value, str):
                    return value
            return ""
        except Exception:
            return ""
