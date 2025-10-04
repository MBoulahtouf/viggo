"""
Concrete implementations of chunking services following SOLID principles.
"""

from typing import List, Dict, Any, Optional
from viggo.core.services.interfaces.chunking import (
    ChunkingStrategy, ChunkingService, ChunkingResult, Chunk, ChunkMetadata, ChunkLevel
)
from viggo.core.services.hybrid_chunking_service import HybridChunkingService, ChunkingConfig


class HybridChunkingStrategy(ChunkingStrategy):
    """Concrete implementation of hybrid chunking strategy."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.hybrid_chunking = HybridChunkingService(config=self.config)
    
    def chunk_document(self, pages: List[Dict[str, Any]]) -> ChunkingResult:
        """Chunk a document using hybrid chunking strategy."""
        # Use existing hybrid chunking implementation
        result = self.hybrid_chunking.chunk_document_hierarchical(pages)
        
        # Convert to new format
        chunks_by_level = {}
        metadata = {}
        hierarchy = {}
        
        for level_name, level_chunks in result["chunks"].items():
            level = ChunkLevel(level_name)
            chunks = []
            
            for chunk_data in level_chunks:
                # Create chunk metadata
                chunk_metadata = ChunkMetadata(
                    chunk_id=chunk_data["id"],
                    level=level,
                    page_number=chunk_data["metadata"].page_number,
                    word_count=chunk_data["metadata"].word_count,
                    char_count=chunk_data["metadata"].char_count,
                    chapter_title=chunk_data["metadata"].chapter_title,
                    content_type=chunk_data["metadata"].content_type,
                    lore_significance=chunk_data["metadata"].lore_significance,
                    entities=chunk_data["metadata"].entities,
                    relationships=chunk_data["metadata"].relationships,
                    parent_id=chunk_data["metadata"].parent_id
                )
                
                # Create chunk
                chunk = Chunk(
                    id=chunk_data["id"],
                    content=chunk_data["content"],
                    level=level,
                    metadata=chunk_metadata
                )
                
                chunks.append(chunk)
                metadata[chunk_data["id"]] = chunk_metadata
            
            chunks_by_level[level] = chunks
        
        return ChunkingResult(
            chunks=chunks_by_level,
            metadata=metadata,
            hierarchy=result["hierarchy"],
            statistics=result["statistics"]
        )
    
    def get_strategy_name(self) -> str:
        """Get the name of this chunking strategy."""
        return "hybrid_chunking"


class StandardChunkingStrategy(ChunkingStrategy):
    """Concrete implementation of standard chunking strategy."""
    
    def __init__(self, max_chunk_size: int = 500, overlap_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
    
    def chunk_document(self, pages: List[Dict[str, Any]]) -> ChunkingResult:
        """Chunk a document using standard strategy."""
        chunks_by_level = {ChunkLevel.PASSAGE: []}
        metadata = {}
        hierarchy = {}
        chunk_counter = 0
        
        for page_data in pages:
            content = page_data.get('content', '')
            page_number = page_data.get('page', 0)
            
            # Simple sentence-based chunking
            sentences = content.split('. ')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < self.max_chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        # Create chunk
                        chunk_id = f"standard_chunk_{chunk_counter}"
                        chunk_metadata = ChunkMetadata(
                            chunk_id=chunk_id,
                            level=ChunkLevel.PASSAGE,
                            page_number=page_number,
                            word_count=len(current_chunk.split()),
                            char_count=len(current_chunk),
                            content_type="story_content",
                            lore_significance=0.5
                        )
                        
                        chunk = Chunk(
                            id=chunk_id,
                            content=current_chunk.strip(),
                            level=ChunkLevel.PASSAGE,
                            metadata=chunk_metadata
                        )
                        
                        chunks_by_level[ChunkLevel.PASSAGE].append(chunk)
                        metadata[chunk_id] = chunk_metadata
                        chunk_counter += 1
                    
                    current_chunk = sentence + ". "
            
            # Add final chunk
            if current_chunk:
                chunk_id = f"standard_chunk_{chunk_counter}"
                chunk_metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    level=ChunkLevel.PASSAGE,
                    page_number=page_number,
                    word_count=len(current_chunk.split()),
                    char_count=len(current_chunk),
                    content_type="story_content",
                    lore_significance=0.5
                )
                
                chunk = Chunk(
                    id=chunk_id,
                    content=current_chunk.strip(),
                    level=ChunkLevel.PASSAGE,
                    metadata=chunk_metadata
                )
                
                chunks_by_level[ChunkLevel.PASSAGE].append(chunk)
                metadata[chunk_id] = chunk_metadata
                chunk_counter += 1
        
        return ChunkingResult(
            chunks=chunks_by_level,
            metadata=metadata,
            hierarchy=hierarchy,
            statistics={
                "total_chunks": chunk_counter,
                "strategy": "standard",
                "max_chunk_size": self.max_chunk_size
            }
        )
    
    def get_strategy_name(self) -> str:
        """Get the name of this chunking strategy."""
        return "standard_chunking"


class ConcreteChunkingService(ChunkingService):
    """Concrete implementation of chunking service."""
    
    def __init__(self, default_strategy: Optional[ChunkingStrategy] = None):
        self.current_strategy = default_strategy or HybridChunkingStrategy()
        self.available_strategies = {
            "hybrid": HybridChunkingStrategy,
            "standard": StandardChunkingStrategy
        }
    
    def chunk_document(self, pages: List[Dict[str, Any]]) -> ChunkingResult:
        """Chunk a document using the configured strategy."""
        return self.current_strategy.chunk_document(pages)
    
    def set_strategy(self, strategy: ChunkingStrategy) -> None:
        """Set the chunking strategy to use."""
        self.current_strategy = strategy
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available chunking strategies."""
        return list(self.available_strategies.keys())
    
    def create_strategy(self, strategy_name: str, **kwargs) -> Optional[ChunkingStrategy]:
        """Create a chunking strategy by name."""
        if strategy_name in self.available_strategies:
            strategy_class = self.available_strategies[strategy_name]
            return strategy_class(**kwargs)
        return None
