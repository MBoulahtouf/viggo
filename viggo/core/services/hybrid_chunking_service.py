"""
Hybrid Chunking Service for Viggo - Implements hierarchical chunking strategy
to reduce noise and improve retrieval accuracy in book lore knowledge exploration.

This service implements:
1. Pre-chunking: Chapter/section-level indexing for broad context
2. Post-chunking: Dynamic passage splitting for detailed answers
3. Overlapping chunks: For critical lore sections to reduce noise
4. Hierarchical indexing: Multi-level exploration (book → chapter → passage)
"""

import re
import spacy
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from viggo.core.services.content_filter_service import ContentFilterService
from viggo.core.services.enhanced_entity_extractor import EnhancedEntityExtractor


class ChunkLevel(Enum):
    """Hierarchical chunk levels for multi-granularity retrieval."""
    BOOK = "book"
    CHAPTER = "chapter"
    SECTION = "section"
    PASSAGE = "passage"
    SENTENCE = "sentence"


class ChunkType(Enum):
    """Types of chunks for different retrieval strategies."""
    FULL_CHAPTER = "full_chapter"
    PARAGRAPH_GROUP = "paragraph_group"
    OVERLAPPING_PASSAGE = "overlapping_passage"
    STANDARD_PASSAGE = "standard_passage"
    CRITICAL_LORE = "critical_lore"
    DIALOGUE_BLOCK = "dialogue_block"
    NARRATIVE_BLOCK = "narrative_block"


@dataclass
class ChunkMetadata:
    """Metadata for a chunk with hierarchical information."""
    level: ChunkLevel
    chunk_type: ChunkType
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    word_count: int = 0
    char_count: int = 0
    page_number: int = 0
    chapter_title: str = ""
    section_title: str = ""
    entities: List[Dict] = None
    relationships: List[Dict] = None
    content_type: str = "story_content"
    lore_significance: float = 0.0  # 0.0 to 1.0, higher = more important for lore
    overlap_ratio: float = 0.0  # For overlapping chunks
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.entities is None:
            self.entities = []
        if self.relationships is None:
            self.relationships = []


@dataclass
class ChunkingConfig:
    """Configuration for hybrid chunking strategy."""
    # Pre-chunking settings
    max_chapter_words: int = 2000
    min_chapter_words: int = 100
    
    # Post-chunking settings
    max_passage_words: int = 400
    min_passage_words: int = 50
    passage_overlap_ratio: float = 0.2  # 20% overlap
    
    # Overlapping chunk settings
    critical_lore_threshold: float = 0.7  # Entities + context score
    max_overlap_chunks: int = 3
    
    # Hierarchical settings
    enable_hierarchical: bool = True
    max_children_per_parent: int = 10
    
    # Content filtering
    enable_content_filtering: bool = True
    skip_metadata_pages: int = 2  # Skip first N pages


class HybridChunkingService:
    """
    Hybrid chunking service that implements hierarchical chunking strategy
    to reduce noise and improve retrieval accuracy.
    """
    
    def __init__(self, nlp_model=None, config: ChunkingConfig = None):
        self.nlp = nlp_model or spacy.load("en_core_web_sm")
        self.config = config or ChunkingConfig()
        self.content_filter = ContentFilterService()
        self.enhanced_extractor = EnhancedEntityExtractor(self.nlp)
        
        # Chunk storage with hierarchical relationships
        self.chunks_by_level = defaultdict(list)
        self.chunk_hierarchy = {}  # parent_id -> children_ids
        self.chunk_metadata = {}  # chunk_id -> ChunkMetadata
        
        # Lore significance patterns
        self.lore_indicators = [
            'ancient', 'mysterious', 'forbidden', 'eldritch', 'cosmic',
            'horror', 'terror', 'dread', 'fear', 'unknown', 'strange',
            'supernatural', 'occult', 'magic', 'ritual', 'summon',
            'entity', 'creature', 'monster', 'demon', 'god', 'deity'
        ]
        
        # Dialogue patterns
        self.dialogue_patterns = [
            r'"[^"]*"',  # Quoted text
            r'"[^"]*"',  # Alternative quotes
            r'"[^"]*"',  # Single quotes
        ]
    
    def chunk_document_hierarchical(self, document_store: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Main entry point for hierarchical document chunking.
        
        Args:
            document_store: List of document pages with content
            
        Returns:
            Dictionary with chunks organized by level
        """
        print("🏗️ Starting hierarchical document chunking...")
        
        # Step 1: Pre-chunk into chapters/sections
        chapter_chunks = self._pre_chunk_document(document_store)
        print(f"📚 Pre-chunked into {len(chapter_chunks)} chapters/sections")
        
        # Step 2: Post-chunk chapters into passages
        passage_chunks = self._post_chunk_chapters(chapter_chunks)
        print(f"📄 Post-chunked into {len(passage_chunks)} passages")
        
        # Step 3: Create overlapping chunks for critical lore
        overlapping_chunks = self._create_overlapping_chunks(passage_chunks)
        print(f"🔄 Created {len(overlapping_chunks)} overlapping chunks")
        
        # Step 4: Build hierarchical relationships
        self._build_hierarchical_relationships(chapter_chunks, passage_chunks, overlapping_chunks)
        
        # Step 5: Organize by level
        organized_chunks = {
            ChunkLevel.CHAPTER.value: chapter_chunks,
            ChunkLevel.PASSAGE.value: passage_chunks,
            ChunkLevel.SENTENCE.value: overlapping_chunks
        }
        
        # Step 6: Calculate chunk statistics
        stats = self._calculate_chunking_statistics(organized_chunks)
        
        print(f"✅ Hierarchical chunking complete:")
        print(f"   Chapters: {len(chapter_chunks)}")
        print(f"   Passages: {len(passage_chunks)}")
        print(f"   Overlapping: {len(overlapping_chunks)}")
        print(f"   Total chunks: {sum(len(chunks) for chunks in organized_chunks.values())}")
        
        return {
            "chunks": organized_chunks,
            "statistics": stats,
            "hierarchy": self.chunk_hierarchy,
            "metadata": self.chunk_metadata
        }
    
    def _pre_chunk_document(self, document_store: List[Dict]) -> List[Dict]:
        """
        Pre-chunk document into chapters/sections for broad context retrieval.
        
        Args:
            document_store: List of document pages
            
        Returns:
            List of chapter-level chunks
        """
        chapter_chunks = []
        
        for doc_page in document_store:
            # Skip metadata pages
            if self.config.enable_content_filtering and doc_page.get("page", 0) <= self.config.skip_metadata_pages:
                continue
            
            content = doc_page.get("content", "")
            if not content or len(content.strip()) < self.config.min_chapter_words:
                continue
            
            # Determine if this should be a full chapter or split
            word_count = len(content.split())
            
            if word_count <= self.config.max_chapter_words:
                # Keep as single chapter
                chapter_chunk = self._create_chapter_chunk(doc_page, content, word_count)
                chapter_chunks.append(chapter_chunk)
            else:
                # Split into sections
                sections = self._split_into_sections(doc_page, content)
                chapter_chunks.extend(sections)
        
        return chapter_chunks
    
    def _create_chapter_chunk(self, doc_page: Dict, content: str, word_count: int) -> Dict:
        """Create a single chapter chunk."""
        chunk_id = f"chapter_{doc_page.get('page', 0)}_{len(self.chunk_metadata)}"
        
        # Extract entities and relationships
        entities = self.enhanced_extractor.extract_entities_enhanced(content, doc_page.get("page", 0))
        relationships = self._extract_relationships(content, entities)
        
        # Calculate lore significance
        lore_significance = self._calculate_lore_significance(content, entities)
        
        # Create chunk metadata
        metadata = ChunkMetadata(
            level=ChunkLevel.CHAPTER,
            chunk_type=ChunkType.FULL_CHAPTER,
            word_count=word_count,
            char_count=len(content),
            page_number=doc_page.get("page", 0),
            chapter_title=doc_page.get("chapter_title", ""),
            entities=entities,
            relationships=relationships,
            lore_significance=lore_significance
        )
        
        # Store metadata
        self.chunk_metadata[chunk_id] = metadata
        
        return {
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
            "level": ChunkLevel.CHAPTER.value,
            "chunk_type": ChunkType.FULL_CHAPTER.value
        }
    
    def _split_into_sections(self, doc_page: Dict, content: str) -> List[Dict]:
        """Split large content into sections."""
        sections = []
        
        # Try to split by natural boundaries (paragraphs, sections)
        paragraphs = content.split('\n\n')
        current_section = ""
        current_word_count = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_words = len(paragraph.split())
            
            # If adding this paragraph would exceed max, finalize current section
            if current_word_count + paragraph_words > self.config.max_chapter_words and current_section:
                section_chunk = self._create_section_chunk(doc_page, current_section, current_word_count)
                sections.append(section_chunk)
                current_section = paragraph
                current_word_count = paragraph_words
            else:
                if current_section:
                    current_section += "\n\n" + paragraph
                else:
                    current_section = paragraph
                current_word_count += paragraph_words
        
        # Add final section
        if current_section:
            section_chunk = self._create_section_chunk(doc_page, current_section, current_word_count)
            sections.append(section_chunk)
        
        return sections
    
    def _create_section_chunk(self, doc_page: Dict, content: str, word_count: int) -> Dict:
        """Create a section chunk."""
        chunk_id = f"section_{doc_page.get('page', 0)}_{len(self.chunk_metadata)}"
        
        # Extract entities and relationships
        entities = self.enhanced_extractor.extract_entities_enhanced(content, doc_page.get("page", 0))
        relationships = self._extract_relationships(content, entities)
        
        # Calculate lore significance
        lore_significance = self._calculate_lore_significance(content, entities)
        
        # Create chunk metadata
        metadata = ChunkMetadata(
            level=ChunkLevel.SECTION,
            chunk_type=ChunkType.PARAGRAPH_GROUP,
            word_count=word_count,
            char_count=len(content),
            page_number=doc_page.get("page", 0),
            chapter_title=doc_page.get("chapter_title", ""),
            entities=entities,
            relationships=relationships,
            lore_significance=lore_significance
        )
        
        # Store metadata
        self.chunk_metadata[chunk_id] = metadata
        
        return {
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
            "level": ChunkLevel.SECTION.value,
            "chunk_type": ChunkType.PARAGRAPH_GROUP.value
        }
    
    def _post_chunk_chapters(self, chapter_chunks: List[Dict]) -> List[Dict]:
        """
        Post-chunk chapters into smaller passages for detailed retrieval.
        
        Args:
            chapter_chunks: List of chapter-level chunks
            
        Returns:
            List of passage-level chunks
        """
        passage_chunks = []
        
        for chapter_chunk in chapter_chunks:
            content = chapter_chunk["content"]
            chapter_id = chapter_chunk["id"]
            word_count = len(content.split())
            
            # If chapter is small enough, keep as single passage
            if word_count <= self.config.max_passage_words:
                passage_chunk = self._create_passage_chunk(
                    chapter_chunk, content, word_count, parent_id=chapter_id
                )
                passage_chunks.append(passage_chunk)
            else:
                # Split into passages with overlap
                passages = self._split_into_passages(chapter_chunk, content)
                passage_chunks.extend(passages)
        
        return passage_chunks
    
    def _split_into_passages(self, chapter_chunk: Dict, content: str) -> List[Dict]:
        """Split chapter content into overlapping passages."""
        passages = []
        chapter_id = chapter_chunk["id"]
        
        # Use sentence-based splitting for better context preservation
        doc = self.nlp(content)
        sentences = [sent.text for sent in doc.sents]
        
        current_passage = ""
        current_word_count = 0
        passage_start_idx = 0
        
        for i, sentence in enumerate(sentences):
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed max, finalize current passage
            if current_word_count + sentence_words > self.config.max_passage_words and current_passage:
                passage_chunk = self._create_passage_chunk(
                    chapter_chunk, current_passage, current_word_count, 
                    parent_id=chapter_id, start_idx=passage_start_idx, end_idx=i-1
                )
                passages.append(passage_chunk)
                
                # Start new passage with overlap
                overlap_sentences = self._calculate_overlap_sentences(sentences, i-1)
                current_passage = " ".join(overlap_sentences + [sentence])
                current_word_count = len(current_passage.split())
                passage_start_idx = max(0, i - len(overlap_sentences))
            else:
                if current_passage:
                    current_passage += " " + sentence
                else:
                    current_passage = sentence
                current_word_count += sentence_words
        
        # Add final passage
        if current_passage:
            passage_chunk = self._create_passage_chunk(
                chapter_chunk, current_passage, current_word_count,
                parent_id=chapter_id, start_idx=passage_start_idx, end_idx=len(sentences)-1
            )
            passages.append(passage_chunk)
        
        return passages
    
    def _calculate_overlap_sentences(self, sentences: List[str], end_idx: int) -> List[str]:
        """Calculate overlap sentences for passage continuity."""
        overlap_count = max(1, int(len(sentences) * self.config.passage_overlap_ratio))
        start_idx = max(0, end_idx - overlap_count + 1)
        return sentences[start_idx:end_idx+1]
    
    def _create_passage_chunk(self, chapter_chunk: Dict, content: str, word_count: int, 
                            parent_id: str, start_idx: int = 0, end_idx: int = 0) -> Dict:
        """Create a passage chunk."""
        chunk_id = f"passage_{parent_id}_{len(self.chunk_metadata)}"
        
        # Extract entities and relationships
        entities = self.enhanced_extractor.extract_entities_enhanced(content, chapter_chunk["metadata"].page_number)
        relationships = self._extract_relationships(content, entities)
        
        # Calculate lore significance
        lore_significance = self._calculate_lore_significance(content, entities)
        
        # Determine chunk type
        chunk_type = self._determine_passage_type(content, entities)
        
        # Create chunk metadata
        metadata = ChunkMetadata(
            level=ChunkLevel.PASSAGE,
            chunk_type=chunk_type,
            parent_id=parent_id,
            word_count=word_count,
            char_count=len(content),
            page_number=chapter_chunk["metadata"].page_number,
            chapter_title=chapter_chunk["metadata"].chapter_title,
            entities=entities,
            relationships=relationships,
            lore_significance=lore_significance,
            overlap_ratio=self.config.passage_overlap_ratio if start_idx > 0 else 0.0
        )
        
        # Store metadata
        self.chunk_metadata[chunk_id] = metadata
        
        return {
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
            "level": ChunkLevel.PASSAGE.value,
            "chunk_type": chunk_type.value,
            "parent_id": parent_id
        }
    
    def _determine_passage_type(self, content: str, entities: List[Dict]) -> ChunkType:
        """Determine the type of passage based on content analysis."""
        content_lower = content.lower()
        
        # Check for dialogue
        dialogue_count = sum(1 for pattern in self.dialogue_patterns 
                           if re.search(pattern, content))
        
        if dialogue_count > 0:
            return ChunkType.DIALOGUE_BLOCK
        
        # Check for critical lore content
        lore_score = self._calculate_lore_significance(content, entities)
        if lore_score > self.config.critical_lore_threshold:
            return ChunkType.CRITICAL_LORE
        
        # Check for narrative indicators
        narrative_indicators = ['he said', 'she said', 'he thought', 'she thought', 
                              'he went', 'she went', 'he looked', 'she looked']
        if any(indicator in content_lower for indicator in narrative_indicators):
            return ChunkType.NARRATIVE_BLOCK
        
        return ChunkType.STANDARD_PASSAGE
    
    def _create_overlapping_chunks(self, passage_chunks: List[Dict]) -> List[Dict]:
        """
        Create overlapping chunks for critical lore sections to reduce noise.
        
        Args:
            passage_chunks: List of passage chunks
            
        Returns:
            List of overlapping chunks
        """
        overlapping_chunks = []
        
        # Find critical lore passages
        critical_passages = [
            chunk for chunk in passage_chunks 
            if chunk["metadata"].lore_significance > self.config.critical_lore_threshold
        ]
        
        for passage in critical_passages[:self.config.max_overlap_chunks]:
            # Create overlapping chunk with extended context
            overlapping_chunk = self._create_overlapping_chunk(passage, passage_chunks)
            if overlapping_chunk:
                overlapping_chunks.append(overlapping_chunk)
        
        return overlapping_chunks
    
    def _create_overlapping_chunk(self, passage: Dict, all_passages: List[Dict]) -> Optional[Dict]:
        """Create an overlapping chunk with extended context."""
        passage_id = passage["id"]
        content = passage["content"]
        
        # Find neighboring passages for context
        neighboring_content = self._get_neighboring_context(passage_id, all_passages)
        
        if not neighboring_content:
            return None
        
        # Combine with original content
        extended_content = f"{neighboring_content}\n\n{content}\n\n{neighboring_content}"
        
        chunk_id = f"overlap_{passage_id}_{len(self.chunk_metadata)}"
        
        # Extract entities and relationships
        entities = self.enhanced_extractor.extract_entities_enhanced(extended_content, passage["metadata"].page_number)
        relationships = self._extract_relationships(extended_content, entities)
        
        # Create chunk metadata
        metadata = ChunkMetadata(
            level=ChunkLevel.SENTENCE,
            chunk_type=ChunkType.OVERLAPPING_PASSAGE,
            parent_id=passage_id,
            word_count=len(extended_content.split()),
            char_count=len(extended_content),
            page_number=passage["metadata"].page_number,
            chapter_title=passage["metadata"].chapter_title,
            entities=entities,
            relationships=relationships,
            lore_significance=passage["metadata"].lore_significance,
            overlap_ratio=0.5  # 50% overlap
        )
        
        # Store metadata
        self.chunk_metadata[chunk_id] = metadata
        
        return {
            "id": chunk_id,
            "content": extended_content,
            "metadata": metadata,
            "level": ChunkLevel.SENTENCE.value,
            "chunk_type": ChunkType.OVERLAPPING_PASSAGE.value,
            "parent_id": passage_id
        }
    
    def _get_neighboring_context(self, passage_id: str, all_passages: List[Dict]) -> str:
        """Get neighboring passages for context."""
        # Find the passage index
        passage_idx = None
        for i, passage in enumerate(all_passages):
            if passage["id"] == passage_id:
                passage_idx = i
                break
        
        if passage_idx is None:
            return ""
        
        # Get neighboring passages
        context_parts = []
        
        # Previous passage
        if passage_idx > 0:
            prev_passage = all_passages[passage_idx - 1]
            context_parts.append(prev_passage["content"][-200:])  # Last 200 chars
        
        # Next passage
        if passage_idx < len(all_passages) - 1:
            next_passage = all_passages[passage_idx + 1]
            context_parts.append(next_passage["content"][:200])  # First 200 chars
        
        return " ".join(context_parts)
    
    def _build_hierarchical_relationships(self, chapter_chunks: List[Dict], 
                                        passage_chunks: List[Dict], 
                                        overlapping_chunks: List[Dict]):
        """Build hierarchical relationships between chunks."""
        # Clear existing relationships
        self.chunk_hierarchy.clear()
        
        # Build chapter -> passage relationships
        for passage in passage_chunks:
            parent_id = passage.get("parent_id")
            if parent_id:
                if parent_id not in self.chunk_hierarchy:
                    self.chunk_hierarchy[parent_id] = []
                self.chunk_hierarchy[parent_id].append(passage["id"])
                
                # Update metadata
                if parent_id in self.chunk_metadata:
                    self.chunk_metadata[parent_id].children_ids.append(passage["id"])
        
        # Build passage -> overlapping relationships
        for overlap in overlapping_chunks:
            parent_id = overlap.get("parent_id")
            if parent_id:
                if parent_id not in self.chunk_hierarchy:
                    self.chunk_hierarchy[parent_id] = []
                self.chunk_hierarchy[parent_id].append(overlap["id"])
                
                # Update metadata
                if parent_id in self.chunk_metadata:
                    self.chunk_metadata[parent_id].children_ids.append(overlap["id"])
    
    def _calculate_lore_significance(self, content: str, entities: List[Dict]) -> float:
        """
        Calculate lore significance score for content.
        
        Args:
            content: Text content
            entities: List of entities in the content
            
        Returns:
            Lore significance score (0.0 to 1.0)
        """
        content_lower = content.lower()
        
        # Count lore indicators
        lore_count = sum(1 for indicator in self.lore_indicators 
                        if indicator in content_lower)
        
        # Count significant entities
        significant_entities = sum(1 for entity in entities 
                                 if entity.get("label") in ["Character", "Location", "Organization"])
        
        # Calculate base score
        lore_score = min(1.0, (lore_count * 0.1) + (significant_entities * 0.2))
        
        # Boost for dialogue (often contains important lore)
        if any(pattern in content for pattern in ['"', '"', "'"]):
            lore_score *= 1.2
        
        # Boost for descriptive passages
        descriptive_words = ['ancient', 'mysterious', 'strange', 'terrible', 'horrible']
        if any(word in content_lower for word in descriptive_words):
            lore_score *= 1.3
        
        return min(1.0, lore_score)
    
    def _extract_relationships(self, content: str, entities: List[Dict]) -> List[Dict]:
        """Extract relationships between entities in content."""
        relationships = []
        
        if len(entities) < 2:
            return relationships
        
        # Simple relationship extraction based on proximity
        doc = self.nlp(content)
        
        for sent in doc.sents:
            sent_entities = [ent for ent in entities 
                           if ent["text"] in sent.text]
            
            if len(sent_entities) >= 2:
                # Create relationships between entities in the same sentence
                for i in range(len(sent_entities)):
                    for j in range(i + 1, len(sent_entities)):
                        relationships.append({
                            "source": sent_entities[i]["text"],
                            "target": sent_entities[j]["text"],
                            "type": "RELATED_TO",
                            "context": sent.text
                        })
        
        return relationships
    
    def _calculate_chunking_statistics(self, organized_chunks: Dict[str, List[Dict]]) -> Dict:
        """Calculate statistics about the chunking process."""
        stats = {
            "total_chunks": 0,
            "chunks_by_level": {},
            "chunks_by_type": {},
            "word_count_stats": {},
            "lore_significance_stats": {},
            "overlap_stats": {}
        }
        
        for level, chunks in organized_chunks.items():
            stats["chunks_by_level"][level] = len(chunks)
            stats["total_chunks"] += len(chunks)
            
            # Count by type
            for chunk in chunks:
                chunk_type = chunk.get("chunk_type", "unknown")
                stats["chunks_by_type"][chunk_type] = stats["chunks_by_type"].get(chunk_type, 0) + 1
                
                # Word count stats
                word_count = chunk["metadata"].word_count
                if level not in stats["word_count_stats"]:
                    stats["word_count_stats"][level] = {"min": word_count, "max": word_count, "avg": 0}
                else:
                    stats["word_count_stats"][level]["min"] = min(stats["word_count_stats"][level]["min"], word_count)
                    stats["word_count_stats"][level]["max"] = max(stats["word_count_stats"][level]["max"], word_count)
                
                # Lore significance stats
                lore_sig = chunk["metadata"].lore_significance
                if level not in stats["lore_significance_stats"]:
                    stats["lore_significance_stats"][level] = {"min": lore_sig, "max": lore_sig, "avg": 0}
                else:
                    stats["lore_significance_stats"][level]["min"] = min(stats["lore_significance_stats"][level]["min"], lore_sig)
                    stats["lore_significance_stats"][level]["max"] = max(stats["lore_significance_stats"][level]["max"], lore_sig)
        
        # Calculate averages
        for level in stats["word_count_stats"]:
            chunks = organized_chunks.get(level, [])
            if chunks:
                avg_words = sum(chunk["metadata"].word_count for chunk in chunks) / len(chunks)
                stats["word_count_stats"][level]["avg"] = round(avg_words, 2)
        
        for level in stats["lore_significance_stats"]:
            chunks = organized_chunks.get(level, [])
            if chunks:
                avg_lore = sum(chunk["metadata"].lore_significance for chunk in chunks) / len(chunks)
                stats["lore_significance_stats"][level]["avg"] = round(avg_lore, 3)
        
        return stats
    
    def get_chunks_by_level(self, level: ChunkLevel) -> List[Dict]:
        """Get chunks at a specific hierarchical level."""
        return self.chunks_by_level.get(level.value, [])
    
    def get_chunk_children(self, chunk_id: str) -> List[Dict]:
        """Get child chunks of a specific chunk."""
        children_ids = self.chunk_hierarchy.get(chunk_id, [])
        return [self.chunk_metadata[child_id] for child_id in children_ids 
                if child_id in self.chunk_metadata]
    
    def get_chunk_parent(self, chunk_id: str) -> Optional[ChunkMetadata]:
        """Get parent chunk of a specific chunk."""
        if chunk_id in self.chunk_metadata:
            parent_id = self.chunk_metadata[chunk_id].parent_id
            if parent_id and parent_id in self.chunk_metadata:
                return self.chunk_metadata[parent_id]
        return None
    
    def get_critical_lore_chunks(self, threshold: float = 0.7) -> List[Dict]:
        """Get chunks with high lore significance."""
        critical_chunks = []
        
        for chunk_id, metadata in self.chunk_metadata.items():
            if metadata.lore_significance >= threshold:
                # Find the actual chunk data
                for level_chunks in self.chunks_by_level.values():
                    for chunk in level_chunks:
                        if chunk["id"] == chunk_id:
                            critical_chunks.append(chunk)
                            break
        
        return critical_chunks
    
    def get_chunking_summary(self) -> Dict:
        """Get a summary of the chunking process."""
        return {
            "total_chunks": sum(len(chunks) for chunks in self.chunks_by_level.values()),
            "chunks_by_level": {level: len(chunks) for level, chunks in self.chunks_by_level.items()},
            "hierarchy_depth": len(self.chunks_by_level),
            "critical_lore_chunks": len(self.get_critical_lore_chunks()),
            "config": {
                "max_chapter_words": self.config.max_chapter_words,
                "max_passage_words": self.config.max_passage_words,
                "passage_overlap_ratio": self.config.passage_overlap_ratio,
                "critical_lore_threshold": self.config.critical_lore_threshold
            }
        }
