"""
Concrete implementation of content filter service following SOLID principles.
"""

import re

from viggo.core.services.interfaces.content_filter import (
    ContentType,
    IContentFilterService,
)


class ContentFilterService(IContentFilterService):
    """
    Service for filtering content during Azure Cognitive Search indexing.
    Prevents non-lore content from being indexed in the first place.
    """

    def __init__(self):
        # Patterns that indicate metadata content
        self.metadata_patterns = [
            r'^Published:\s*',
            r'^Categorie\(s\):\s*',
            r'^Source:\s*',
            r'^Copyright:\s*',
            r'^Also available\s+',
            r'^OceanofPDF',
            r'^Feedbooks',
            r'^Wikipedia',
            r'^Strictly for personal use',
            r'^Life\+70',
            r'^Available for countries',
            r'^Howard Phillips Lovecraft$',
            r'^Lovecraft$',
            r'^Time$',
        ]

        # Lovecraft bibliography entries (not story content)
        self.bibliography_patterns = [
            r'^The Call of Cthulhu',
            r'^At the Mountains of Madness',
            r'^The Dunwich Horror',
            r'^The Shadow out of Time',
            r'^The Shadow Over Innsmouth',
            r'^The Haunter of the Dark',
            r'^The Colour Out of Space',
            r'^The Whisperer in Darkness',
            r'^Supernatural Horror in Literature',
            r'^Dreams in the Witch-House',
            r'^The Statement of Randolph Carter',
            r'^The Silver Key',
            r'^The Tree',
            r'^The Temple',
            r'^What the Moon Brings',
            r'^Howard Phillips Lovecraft Poetry',
            r'^Loved this book',
            r'^Similar users also downloaded',
            r'^Food for the mind',
        ]

        # Publisher and technical metadata
        self.publisher_patterns = [
            r'^OceanofPDF\.com',
            r'^Feedbooks\.com',
            r'^www\.feedbooks\.com',
            r'^Food for the mind',
            r'^Strictly for personal use',
            r'^do not use this file',
        ]

        # Technical patterns (file paths, chunk IDs, etc.)
        self.technical_patterns = [
            r'^/.*\.pdf$',
            r'.*_page\d+_chunk\d+.*',
            r'^chunk_id\s*:',
            r'^Source:',
        ]

        # Preface/intro patterns
        self.preface_patterns = [
            r'^About Lovecraft:',
            r'^Note: This book is brought to you by',
            r'^Howard Phillips Lovecraft was an American author',
        ]

        # Compile all patterns for efficiency
        self.compiled_patterns = {
            ContentType.METADATA: [re.compile(pattern, re.IGNORECASE) for pattern in self.metadata_patterns],
            ContentType.BIBLIOGRAPHY: [re.compile(pattern, re.IGNORECASE) for pattern in self.bibliography_patterns],
            ContentType.PUBLISHER_INFO: [re.compile(pattern, re.IGNORECASE) for pattern in self.publisher_patterns],
            ContentType.TECHNICAL: [re.compile(pattern, re.IGNORECASE) for pattern in self.technical_patterns],
            ContentType.PREFACE: [re.compile(pattern, re.IGNORECASE) for pattern in self.preface_patterns],
        }

        # Keywords that indicate story content
        self.story_indicators = [
            'said', 'thought', 'went', 'came', 'looked', 'felt', 'heard', 'saw',
            'walked', 'ran', 'stood', 'sat', 'lived', 'died', 'spoke', 'whispered',
            'house', 'room', 'door', 'window', 'street', 'town', 'city', 'forest',
            'night', 'day', 'morning', 'evening', 'mist', 'fog', 'dark', 'light',
            'strange', 'ancient', 'old', 'mysterious', 'terrible', 'horrible'
        ]

    def classify_content_type(self, content: str, page_number: int = 0) -> ContentType:
        """
        Classify content type based on patterns and context.
        
        Args:
            content: Text content to classify
            page_number: Page number for context
            
        Returns:
            ContentType classification
        """
        content = content.strip()

        # Skip empty or very short content
        if len(content) < 20:
            return ContentType.METADATA

        # Check against all pattern types
        for content_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.match(content):
                    return content_type

        # Check for metadata indicators in content
        metadata_indicators = [
            'Published:', 'Categorie(s):', 'Source:', 'Copyright:',
            'Also available', 'OceanofPDF', 'Feedbooks', 'Wikipedia',
            'Strictly for personal use', 'Life+70', 'Available for countries'
        ]

        content_lower = content.lower()
        metadata_score = sum(1 for indicator in metadata_indicators
                           if indicator.lower() in content_lower)

        # If more than 2 metadata indicators, likely metadata
        if metadata_score >= 2:
            return ContentType.METADATA

        # Check if content is mostly bibliography
        bibliography_score = sum(1 for pattern in self.bibliography_patterns
                               if re.search(pattern, content, re.IGNORECASE))

        if bibliography_score > 0:
            return ContentType.BIBLIOGRAPHY

        # Check for story indicators
        story_score = sum(1 for indicator in self.story_indicators
                         if indicator in content_lower)

        # If content has good story indicators, it's likely story content
        if story_score >= 2:
            return ContentType.STORY_CONTENT

        # Early pages are often metadata/preface
        if page_number <= 2:
            return ContentType.METADATA

        # Default to story content if no clear classification
        return ContentType.STORY_CONTENT

    def should_index_content(self, content: str, page_number: int = 0) -> bool:
        """
        Determine if content should be indexed in Azure Cognitive Search.
        
        Args:
            content: Text content to evaluate
            page_number: Page number for context
            
        Returns:
            True if content should be indexed, False otherwise
        """
        content_type = self.classify_content_type(content, page_number)

        # Only index story content
        return content_type == ContentType.STORY_CONTENT

    def filter_chunks_for_indexing(self, chunks: list[dict]) -> tuple[list[dict], dict[str, int]]:
        """
        Filter chunks to only include those that should be indexed.
        
        Args:
            chunks: List of chunks to filter
            
        Returns:
            Tuple of (filtered_chunks, filter_stats)
        """
        filtered_chunks = []
        filter_stats = {
            'total_chunks': len(chunks),
            'filtered_out': 0,
            'metadata': 0,
            'bibliography': 0,
            'publisher_info': 0,
            'technical': 0,
            'preface': 0,
            'story_content': 0
        }

        for chunk in chunks:
            content = chunk.get('content', '')
            page_number = chunk.get('page', 0)

            content_type = self.classify_content_type(content, page_number)
            filter_stats[content_type.value] += 1

            if self.should_index_content(content, page_number):
                filtered_chunks.append(chunk)
            else:
                filter_stats['filtered_out'] += 1

        return filtered_chunks, filter_stats

    def get_indexing_filter_expression(self) -> str:
        """
        Get Azure Cognitive Search filter expression to exclude non-lore content.
        This can be used in search queries to filter results.
        
        Returns:
            OData filter expression string
        """
        # Example filter expressions for different content types
        filter_expressions = [
            "content_type eq 'story_content'",
            "not content_type eq 'metadata'",
            "not content_type eq 'bibliography'",
            "not content_type eq 'publisher_info'",
            "not content_type eq 'technical'",
            "not content_type eq 'preface'"
        ]

        return " and ".join(filter_expressions)

    def add_content_type_to_chunk(self, chunk: dict) -> dict:
        """
        Add content type classification to a chunk.
        
        Args:
            chunk: Chunk dictionary to enhance
            
        Returns:
            Enhanced chunk with content_type field
        """
        content = chunk.get('content', '')
        page_number = chunk.get('page', 0)

        content_type = self.classify_content_type(content, page_number)
        chunk['content_type'] = content_type.value

        return chunk

    def get_filtering_stats(self, chunks: list[dict]) -> dict[str, any]:
        """
        Get detailed statistics about content filtering.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with filtering statistics
        """
        stats = {
            'total_chunks': len(chunks),
            'content_types': {},
            'page_distribution': {},
            'word_count_stats': {'total': 0, 'story_content': 0}
        }

        for chunk in chunks:
            content = chunk.get('content', '')
            page_number = chunk.get('page', 0)
            word_count = len(content.split())

            content_type = self.classify_content_type(content, page_number)
            type_name = content_type.value

            # Count by content type
            stats['content_types'][type_name] = stats['content_types'].get(type_name, 0) + 1

            # Track page distribution
            page_range = f"pages_{((page_number-1)//5)*5+1}-{((page_number-1)//5)*5+5}"
            stats['page_distribution'][page_range] = stats['page_distribution'].get(page_range, 0) + 1

            # Word count stats
            stats['word_count_stats']['total'] += word_count
            if content_type == ContentType.STORY_CONTENT:
                stats['word_count_stats']['story_content'] += word_count

        return stats
