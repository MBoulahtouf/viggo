"""
Entity-Chunk Linking Service for Viggo

This service creates the missing link between Neo4j entities and specific chunks,
enabling users to find passages where entities are mentioned with context-aware retrieval.
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re

from viggo.core.services.graph_service import GraphService
from viggo.core.services.hybrid_chunking_service import ChunkLevel, ChunkMetadata


class ContextType(Enum):
    """Types of context where entities appear."""
    DESCRIPTION = "description"  # Entity is being described
    ACTION = "action"           # Entity is performing an action
    RELATIONSHIP = "relationship"  # Entity's relationship to others
    ATMOSPHERIC = "atmospheric"    # Entity in atmospheric context
    DIALOGUE = "dialogue"       # Entity mentioned in dialogue
    REFERENCE = "reference"     # Entity referenced indirectly


@dataclass
class EntityChunkLink:
    """Link between an entity and a specific chunk with context information."""
    entity_name: str
    entity_label: str
    chunk_id: str
    chunk_level: ChunkLevel
    page_number: int
    context_type: ContextType
    context_score: float  # 0.0 to 1.0, how relevant this context is
    entity_positions: List[int]  # Character positions where entity appears
    surrounding_text: str  # Text around entity mentions
    lore_significance: float  # From chunk metadata


@dataclass
class EntityContextResult:
    """Result of entity context analysis."""
    entity_name: str
    entity_label: str
    total_mentions: int
    context_types: Dict[ContextType, int]
    chunks_by_context: Dict[ContextType, List[EntityChunkLink]]
    spoiler_safe_chunks: List[EntityChunkLink]  # Chunks within user's page limit


class EntityChunkLinker:
    """
    Service that links entities to chunks and provides context-aware retrieval.
    """
    
    def __init__(self, graph_service: GraphService):
        self.graph_service = graph_service
        
        # Context analysis patterns
        self.context_patterns = {
            ContextType.DESCRIPTION: [
                r'\b(?:is|was|are|were|appears|seems|looks?|feels?)\s+',
                r'\b(?:described|known|called|named)\s+as\s+',
                r'\b(?:located|situated|found)\s+(?:in|at|on)\s+',
                r'\b(?:characterized|defined|identified)\s+by\s+'
            ],
            ContextType.ACTION: [
                r'\b(?:said|spoke|whispered|shouted|exclaimed)\s+',
                r'\b(?:went|walked|ran|moved|traveled)\s+',
                r'\b(?:looked|saw|observed|noticed|watched)\s+',
                r'\b(?:felt|touched|grasped|held)\s+',
                r'\b(?:thought|considered|pondered|wondered)\s+'
            ],
            ContextType.RELATIONSHIP: [
                r'\b(?:with|and|together|alongside)\s+',
                r'\b(?:related|connected|associated)\s+(?:to|with)\s+',
                r'\b(?:friend|ally|enemy|rival)\s+(?:of|to)\s+',
                r'\b(?:member|part|belongs?)\s+(?:of|to)\s+'
            ],
            ContextType.ATMOSPHERIC: [
                r'\b(?:mysterious|strange|ancient|forbidden|eldritch)\s+',
                r'\b(?:dark|shadowy|ominous|eerie|haunting)\s+',
                r'\b(?:mist|fog|clouds|shadows|darkness)\s+',
                r'\b(?:supernatural|occult|magical|otherworldly)\s+'
            ],
            ContextType.DIALOGUE: [
                r'"[^"]*',  # Quoted text
                r'"[^"]*',  # Alternative quotes
                r'"[^"]*'   # Single quotes
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for context_type, patterns in self.context_patterns.items():
            self.compiled_patterns[context_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def create_entity_chunk_links(self, chunks_with_metadata: List[Dict]) -> List[EntityChunkLink]:
        """
        Create links between entities and chunks with context analysis.
        
        Args:
            chunks_with_metadata: List of chunks with metadata from hybrid chunking
            
        Returns:
            List of EntityChunkLink objects
        """
        entity_chunk_links = []
        
        for chunk in chunks_with_metadata:
            chunk_id = chunk.get("id", f"chunk_{len(entity_chunk_links)}")
            chunk_level = ChunkLevel(chunk.get("level", "passage"))
            page_number = chunk.get("page", 0)
            content = chunk.get("content", "")
            entities = chunk.get("entities", [])
            lore_significance = chunk.get("lore_significance", 0.0)
            
            # Create links for each entity in this chunk
            for entity in entities:
                entity_name = entity.get("text", "")
                entity_label = entity.get("label", "")
                
                if not entity_name:
                    continue
                
                # Find entity positions in chunk content
                entity_positions = self._find_entity_positions(content, entity_name)
                
                if not entity_positions:
                    continue
                
                # Analyze context type for each mention
                for position in entity_positions:
                    context_type = self._analyze_context_type(content, position, entity_name)
                    context_score = self._calculate_context_score(content, position, context_type)
                    surrounding_text = self._extract_surrounding_text(content, position)
                    
                    link = EntityChunkLink(
                        entity_name=entity_name,
                        entity_label=entity_label,
                        chunk_id=chunk_id,
                        chunk_level=chunk_level,
                        page_number=page_number,
                        context_type=context_type,
                        context_score=context_score,
                        entity_positions=[position],
                        surrounding_text=surrounding_text,
                        lore_significance=lore_significance
                    )
                    
                    entity_chunk_links.append(link)
        
        return entity_chunk_links
    
    def get_entity_context(self, entity_name: str, user_page_limit: Optional[int] = None) -> EntityContextResult:
        """
        Get context-aware information about an entity with spoiler protection.
        
        Args:
            entity_name: Name of the entity to analyze
            user_page_limit: Optional page limit for spoiler protection
            
        Returns:
            EntityContextResult with context analysis
        """
        # Get entity details from Neo4j
        entity_data = self.graph_service.get_entity_details(entity_name)
        if not entity_data:
            return EntityContextResult(
                entity_name=entity_name,
                entity_label="Unknown",
                total_mentions=0,
                context_types={},
                chunks_by_context={},
                spoiler_safe_chunks=[]
            )
        
        entity_label = entity_data.get("labels", ["Unknown"])[0] if entity_data.get("labels") else "Unknown"
        
        # Get all chunks mentioning this entity (this would need to be stored/retrieved)
        # For now, we'll simulate this - in production, you'd query your chunk storage
        entity_links = self._get_entity_links_from_storage(entity_name)
        
        # Filter by spoiler protection
        spoiler_safe_links = [
            link for link in entity_links 
            if user_page_limit is None or link.page_number <= user_page_limit
        ]
        
        # Analyze context types
        context_types = {}
        chunks_by_context = {}
        
        for link in spoiler_safe_links:
            context_type = link.context_type
            context_types[context_type] = context_types.get(context_type, 0) + 1
            
            if context_type not in chunks_by_context:
                chunks_by_context[context_type] = []
            chunks_by_context[context_type].append(link)
        
        return EntityContextResult(
            entity_name=entity_name,
            entity_label=entity_label,
            total_mentions=len(spoiler_safe_links),
            context_types=context_types,
            chunks_by_context=chunks_by_context,
            spoiler_safe_chunks=spoiler_safe_links
        )
    
    def find_entity_passages(self, entity_name: str, context_type: Optional[ContextType] = None, 
                           user_page_limit: Optional[int] = None) -> List[EntityChunkLink]:
        """
        Find specific passages where an entity is mentioned with optional context filtering.
        
        Args:
            entity_name: Name of the entity
            context_type: Optional context type filter
            user_page_limit: Optional page limit for spoiler protection
            
        Returns:
            List of EntityChunkLink objects
        """
        entity_links = self._get_entity_links_from_storage(entity_name)
        
        # Apply filters
        filtered_links = entity_links
        
        if context_type:
            filtered_links = [link for link in filtered_links if link.context_type == context_type]
        
        if user_page_limit is not None:
            filtered_links = [link for link in filtered_links if link.page_number <= user_page_limit]
        
        # Sort by context score and lore significance
        filtered_links.sort(key=lambda x: (x.context_score, x.lore_significance), reverse=True)
        
        return filtered_links
    
    def get_entity_relationships_in_context(self, entity_name: str, 
                                          user_page_limit: Optional[int] = None) -> Dict[str, List[EntityChunkLink]]:
        """
        Get entity relationships with context from chunks.
        
        Args:
            entity_name: Name of the entity
            user_page_limit: Optional page limit for spoiler protection
            
        Returns:
            Dictionary mapping related entities to their context chunks
        """
        # Get relationships from Neo4j
        relationships = self.graph_service.get_related_info_for_entity(entity_name)
        
        related_entities = {}
        
        for rel_info in relationships:
            related_entity = rel_info.get("related_node", {}).get("name", "")
            if not related_entity:
                continue
            
            # Find chunks where both entities appear together
            entity_links = self.find_entity_passages(entity_name, ContextType.RELATIONSHIP, user_page_limit)
            related_links = self.find_entity_passages(related_entity, ContextType.RELATIONSHIP, user_page_limit)
            
            # Find overlapping chunks (both entities mentioned)
            overlapping_chunks = []
            entity_chunk_ids = {link.chunk_id for link in entity_links}
            related_chunk_ids = {link.chunk_id for link in related_links}
            
            for chunk_id in entity_chunk_ids.intersection(related_chunk_ids):
                chunk_links = [link for link in entity_links if link.chunk_id == chunk_id]
                overlapping_chunks.extend(chunk_links)
            
            if overlapping_chunks:
                related_entities[related_entity] = overlapping_chunks
        
        return related_entities
    
    def _find_entity_positions(self, content: str, entity_name: str) -> List[int]:
        """Find all positions where entity appears in content."""
        positions = []
        content_lower = content.lower()
        entity_lower = entity_name.lower()
        
        start = 0
        while True:
            pos = content_lower.find(entity_lower, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        return positions
    
    def _analyze_context_type(self, content: str, position: int, entity_name: str) -> ContextType:
        """Analyze the context type around an entity mention."""
        # Extract context window around the entity
        context_window = 100  # characters before and after
        start = max(0, position - context_window)
        end = min(len(content), position + len(entity_name) + context_window)
        context_text = content[start:end]
        
        # Check each context type
        for context_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(context_text):
                    return context_type
        
        # Default to reference if no specific context found
        return ContextType.REFERENCE
    
    def _calculate_context_score(self, content: str, position: int, context_type: ContextType) -> float:
        """Calculate how relevant this context is (0.0 to 1.0)."""
        base_score = 0.5
        
        # Boost score based on context type
        context_boosts = {
            ContextType.DESCRIPTION: 0.3,
            ContextType.ACTION: 0.2,
            ContextType.RELATIONSHIP: 0.25,
            ContextType.ATMOSPHERIC: 0.2,
            ContextType.DIALOGUE: 0.15,
            ContextType.REFERENCE: 0.0
        }
        
        return min(1.0, base_score + context_boosts.get(context_type, 0.0))
    
    def _extract_surrounding_text(self, content: str, position: int, window: int = 50) -> str:
        """Extract text surrounding an entity mention."""
        start = max(0, position - window)
        end = min(len(content), position + window)
        return content[start:end].strip()
    
    def _get_entity_links_from_storage(self, entity_name: str) -> List[EntityChunkLink]:
        """
        Get entity links from storage. In production, this would query your chunk storage.
        For now, this is a placeholder that would be implemented based on your storage system.
        """
        # This would typically query your chunk storage (FAISS, database, etc.)
        # to find all chunks mentioning this entity
        return []
    
    def store_entity_chunk_links(self, entity_chunk_links: List[EntityChunkLink]) -> bool:
        """
        Store entity-chunk links for future retrieval.
        
        Args:
            entity_chunk_links: List of links to store
            
        Returns:
            True if successful, False otherwise
        """
        # This would store the links in your chosen storage system
        # (Neo4j, database, or alongside your chunk storage)
        # For now, this is a placeholder
        return True
