"""
Concrete implementation of enhanced entity extractor following SOLID principles.
"""

import re
import spacy
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from difflib import SequenceMatcher
from spacy.tokens import Doc, Span

from viggo.core.utils.entity_utils import filter_and_map_entities, get_allowed_labels
from viggo.core.services.interfaces.entity_extractor import (
    IContentFilter, IEntityDeduplicator, IEntityDisambiguator, IEnhancedEntityExtractor
)


class ContentFilter(IContentFilter):
    """Filters out non-story content like metadata, prefaces, and publisher info."""
    
    def __init__(self):
        # Patterns that indicate non-story content
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
            r'^What the Moon Brings',
            r'^The Temple',
            r'^Howard Phillips Lovecraft Poetry',
            r'^Loved this book',
            r'^Similar users also downloaded',
            r'^Food for the mind',
        ]
        
        # File path and technical patterns
        self.technical_patterns = [
            r'^/.*\.pdf$',
            r'.*_page\d+_chunk\d+.*',
            r'.*\.pdf.*',
            r'^chunk_id\s*:',
        ]
        
        # Combine all patterns
        self.all_patterns = self.metadata_patterns + self.bibliography_patterns + self.technical_patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.all_patterns]
    
    def is_story_content(self, content: str) -> bool:
        """
        Determine if content contains actual story material.
        
        Args:
            content: Text content to evaluate
            
        Returns:
            True if content appears to be story material, False otherwise
        """
        content = content.strip()
        
        # Skip empty or very short content
        if len(content) < 20:
            return False
        
        # Check against exclusion patterns
        for pattern in self.compiled_patterns:
            if pattern.match(content):
                return False
        
        # Check for metadata indicators
        metadata_indicators = [
            'Published:', 'Categorie(s):', 'Source:', 'Copyright:',
            'Also available', 'OceanofPDF', 'Feedbooks', 'Wikipedia',
            'Strictly for personal use', 'Life+70', 'Available for countries'
        ]
        
        content_lower = content.lower()
        metadata_score = sum(1 for indicator in metadata_indicators 
                           if indicator.lower() in content_lower)
        
        # If more than 2 metadata indicators, likely not story content
        if metadata_score >= 2:
            return False
        
        # Check if content is mostly bibliography
        bibliography_score = sum(1 for pattern in self.bibliography_patterns
                               if re.search(pattern, content, re.IGNORECASE))
        
        if bibliography_score > 0:
            return False
        
        return True
    
    def should_process_chunk(self, chunk_content: str, page_number: int) -> bool:
        """
        Determine if a chunk should be processed for entity extraction.
        
        Args:
            chunk_content: The content of the chunk
            page_number: Page number (early pages often contain metadata)
            
        Returns:
            True if chunk should be processed, False otherwise
        """
        # Skip first few pages which often contain metadata
        if page_number <= 2:
            return False
        
        return self.is_story_content(chunk_content)


class EntityDeduplicator(IEntityDeduplicator):
    """Handles entity deduplication and normalization."""
    
    def __init__(self):
        # Known entity aliases and variations
        self.entity_aliases = {
            'thomas olney': 'olney',
            'olney': 'olney',
            'granny orne': 'granny orne',
            'kingsport': 'kingsport',
            'arkham': 'arkham',
            'central hill': 'central hill',
            'elder ones': 'the elder ones',
            'the elder ones': 'the elder ones',
            'congregational hospital': 'congregational hospital',
            'nodens': 'nodens',
            'neptune': 'neptune',
            'miskatonic': 'miskatonic',
            'hatheg-kia': 'hatheg-kia',
            'yankees': 'yankees',
            'dragon': 'dragon',
            'great bear': 'the great bear',
            'the great bear': 'the great bear',
        }
        
        # Similarity threshold for fuzzy matching
        self.similarity_threshold = 0.8
    
    def normalize_entity_name(self, entity_text: str) -> str:
        """
        Normalize entity name to a canonical form.
        
        Args:
            entity_text: Raw entity text
            
        Returns:
            Normalized entity name
        """
        # Basic normalization
        normalized = re.sub(r'\s+', ' ', entity_text.strip().lower())
        
        # Apply known aliases
        return self.entity_aliases.get(normalized, normalized)
    
    def find_similar_entities(self, entity_name: str, existing_entities: List[Dict]) -> List[Dict]:
        """
        Find entities similar to the given entity name.
        
        Args:
            entity_name: Entity name to match
            existing_entities: List of existing entities
            
        Returns:
            List of similar entities
        """
        similar = []
        normalized_name = self.normalize_entity_name(entity_name)
        
        for entity in existing_entities:
            existing_name = self.normalize_entity_name(entity['text'])
            
            # Exact match
            if normalized_name == existing_name:
                similar.append(entity)
                continue
            
            # Fuzzy match
            similarity = SequenceMatcher(None, normalized_name, existing_name).ratio()
            if similarity >= self.similarity_threshold:
                similar.append(entity)
        
        return similar
    
    def merge_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Merge similar entities into canonical forms.
        
        Args:
            entities: List of entities to deduplicate
            
        Returns:
            List of deduplicated entities
        """
        merged = []
        processed = set()
        
        for entity in entities:
            entity_name = entity['text']
            normalized_name = self.normalize_entity_name(entity_name)
            
            if normalized_name in processed:
                continue
            
            # Find all similar entities
            similar = self.find_similar_entities(entity_name, entities)
            
            if similar:
                # Use the most common form as canonical
                canonical_entity = max(similar, key=lambda x: len(x['text']))
                canonical_entity['text'] = normalized_name.title()  # Proper case
                merged.append(canonical_entity)
                processed.add(normalized_name)
        
        return merged


class EntityDisambiguator(IEntityDisambiguator):
    """Handles entity type disambiguation and conflict resolution."""
    
    def __init__(self):
        # Known entity type mappings for Lovecraft stories
        self.entity_type_mapping = {
            # Characters
            'olney': 'Character',
            'thomas olney': 'Character',
            'granny orne': 'Character',
            'nodens': 'Character',
            'miskatonic': 'Character',
            'kingsporter': 'Character',
            'lovecraft': 'Character',  # Author name should be character
            'howard phillips lovecraft': 'Character',
            'shirley': 'Character',
            'pownall': 'Character',
            'bernard': 'Character',
            'majesty': 'Character',
            'narragansett bay': 'Location',  # This is actually a location
            'causeway': 'Location',
            'east': 'Location',
            'pinnacle': 'Location',
            'next': 'Location',
            'earth': 'Location',
            'randolph carter': 'Character',
            
            # Locations
            'kingsport': 'Location',
            'arkham': 'Location',
            'central hill': 'Location',
            'tudor': 'Location',
            'poseidon': 'Location',
            'ulthar': 'Location',
            'new england': 'Location',
            'bristol highlands': 'Location',
            'archaic kingsport': 'Location',
            'congregational': 'Location',
            'kadath': 'Location',
            
            # Organizations
            'the elder ones': 'Organization',
            'elder ones': 'Organization',
            'congregational hospital': 'Organization',
            'hatheg-kia': 'Organization',
            'mighty ones': 'Organization',
            'yankees': 'Organization',
            'dragon': 'Organization',
            'the great bear': 'Organization',
            'neptune': 'Organization',
            
            # Books/Works (should be filtered out)
            'necronomicon': 'Work',
            'the temple': 'Work',
        }
        
        # Context clues for disambiguation
        self.character_indicators = ['said', 'thought', 'went', 'came', 'looked', 'felt']
        self.organization_indicators = ['organization', 'group', 'society', 'hospital', 'institution']
        self.location_indicators = ['in', 'at', 'near', 'located', 'place', 'town', 'city']
    
    def disambiguate_entity_type(self, entity_name: str, entity_label: str, context: str) -> str:
        """
        Disambiguate entity type based on context and known mappings.
        
        Args:
            entity_name: Name of the entity
            entity_label: Current label from spaCy
            context: Surrounding text context
            
        Returns:
            Disambiguated entity label
        """
        normalized_name = entity_name.lower().strip()
        
        # Check known mappings first
        if normalized_name in self.entity_type_mapping:
            return self.entity_type_mapping[normalized_name]
        
        # Use context clues for disambiguation
        context_lower = context.lower()
        
        # Character indicators
        if any(indicator in context_lower for indicator in self.character_indicators):
            return 'Character'
        
        # Organization indicators
        if any(indicator in context_lower for indicator in self.organization_indicators):
            return 'Organization'
        
        # Location indicators
        if any(indicator in context_lower for indicator in self.location_indicators):
            return 'Location'
        
        # Default to original label if no disambiguation possible
        return entity_label


class EnhancedEntityExtractor(IEnhancedEntityExtractor):
    """
    Enhanced entity extractor with content filtering, deduplication, and disambiguation.
    """
    
    def __init__(self, nlp_model=None):
        self.nlp = nlp_model or spacy.load("en_core_web_sm")
        self.content_filter = ContentFilter()
        self.deduplicator = EntityDeduplicator()
        self.disambiguator = EntityDisambiguator()
    
    def should_process_content(self, content: str, page_number: int) -> bool:
        """
        Determine if content should be processed for entity extraction.
        
        Args:
            content: Text content
            page_number: Page number
            
        Returns:
            True if content should be processed
        """
        return self.content_filter.should_process_chunk(content, page_number)
    
    def extract_entities_enhanced(self, content: str, page_number: int) -> List[Dict]:
        """
        Extract entities with enhanced filtering and processing.
        
        Args:
            content: Text content to process
            page_number: Page number for context
            
        Returns:
            List of enhanced entities
        """
        # Skip if content shouldn't be processed
        if not self.should_process_content(content, page_number):
            return []
        
        # Extract entities using spaCy
        doc = self.nlp(content)
        entities = filter_and_map_entities(doc, get_allowed_labels())
        
        # Filter out noisy entities
        filtered_entities = self._filter_noisy_entities(entities)
        
        # Disambiguate entity types
        disambiguated_entities = []
        for entity in filtered_entities:
            entity_name = entity['text']
            entity_label = entity['label']
            
            # Get disambiguated label
            disambiguated_label = self.disambiguator.disambiguate_entity_type(
                entity_name, entity_label, content
            )
            
            entity['label'] = disambiguated_label
            disambiguated_entities.append(entity)
        
        # Deduplicate similar entities
        deduplicated_entities = self.deduplicator.merge_entities(disambiguated_entities)
        
        return deduplicated_entities
    
    def _filter_noisy_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Filter out noisy entities that aren't relevant to the story.
        
        Args:
            entities: List of entities to filter
            
        Returns:
            List of filtered entities
        """
        filtered = []
        
        for entity in entities:
            entity_text = entity['text'].strip()
            entity_label = entity.get('label', '')
            
            # Skip very short or very long entities
            if len(entity_text) < 2 or len(entity_text) > 50:
                continue
            
            # Skip entities that are mostly numbers or special characters
            if re.match(r'^[\d\s\-_\.]+$', entity_text):
                continue
            
            # Skip Work entities (books, etc.)
            if entity_label == 'Work':
                continue
            
            # Skip common noise patterns
            noise_patterns = [
                r'^[A-Z]{2,}$',  # All caps (likely acronyms)
                r'^\d+$',        # Pure numbers
                r'^[^\w\s]+$',   # Only special characters
            ]
            
            if any(re.match(pattern, entity_text) for pattern in noise_patterns):
                continue
            
            # Skip entities that look like bibliography entries
            bibliography_indicators = [
                'download', 'book', 'similar', 'users', 'loved', 'food', 'mind'
            ]
            
            if any(indicator in entity_text.lower() for indicator in bibliography_indicators):
                continue
            
            filtered.append(entity)
        
        return filtered
    
    def process_chunks_enhanced(self, chunks: List[Dict]) -> List[Dict]:
        """
        Process multiple chunks with enhanced entity extraction and global deduplication.
        
        Args:
            chunks: List of chunks to process
            
        Returns:
            List of processed chunks with enhanced entities
        """
        processed_chunks = []
        all_entities = []  # Collect all entities for global deduplication
        
        # First pass: extract entities from all chunks
        for chunk in chunks:
            content = chunk.get('content', '')
            page_number = chunk.get('page', 0)
            
            # Extract enhanced entities
            enhanced_entities = self.extract_entities_enhanced(content, page_number)
            
            # Update chunk with enhanced entities
            chunk['entities'] = enhanced_entities
            chunk['relationships'] = []  # Clear relationships for now
            
            # Collect entities for global deduplication
            all_entities.extend(enhanced_entities)
            
            # Only keep chunks with relevant entities
            if enhanced_entities:
                processed_chunks.append(chunk)
        
        # Second pass: global deduplication
        print(f"🔍 Global deduplication: {len(all_entities)} entities before, ", end="")
        globally_deduplicated = self.deduplicator.merge_entities(all_entities)
        print(f"{len(globally_deduplicated)} entities after")
        
        # Third pass: update chunks with globally deduplicated entities
        entity_map = {self.deduplicator.normalize_entity_name(e['text']): e for e in globally_deduplicated}
        
        for chunk in processed_chunks:
            original_entities = chunk.get('entities', [])
            deduplicated_chunk_entities = []
            
            for entity in original_entities:
                normalized_name = self.deduplicator.normalize_entity_name(entity['text'])
                if normalized_name in entity_map:
                    deduplicated_chunk_entities.append(entity_map[normalized_name])
            
            chunk['entities'] = deduplicated_chunk_entities
        
        return processed_chunks
