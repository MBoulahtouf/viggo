"""
Enhanced RAG Integration Service

This service integrates the entity-chunk linking architecture with the existing
hybrid chunking and spoiler safeguard system.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from viggo.core.services.rag_service import RAGService
from viggo.core.services.graph_service import GraphService
from viggo.core.services.entity_chunk_linker import EntityChunkLinker, EntityChunkLink, ContextType
from viggo.core.services.context_aware_retriever import ContextAwareRetriever, ContextAwareResult, QueryIntent
from viggo.core.services.hybrid_chunking_service import HybridChunkingService, ChunkLevel


@dataclass
class EnhancedRAGResult:
    """Enhanced RAG result with entity-chunk linking."""
    query: str
    answer: str
    source_pages: List[int]
    search_method: str
    
    # Entity-chunk linking data
    entities_found: List[str]
    entity_contexts: Dict[str, ContextAwareResult]
    chunk_links: List[EntityChunkLink]
    
    # Metadata
    spoiler_protected: bool
    user_page_limit: Optional[int]
    processing_time: float


class EnhancedRAGIntegration:
    """
    Integration service that combines hybrid chunking with entity-chunk linking.
    """
    
    def __init__(self, rag_service: RAGService, graph_service: GraphService):
        self.rag_service = rag_service
        self.graph_service = graph_service
        self.entity_chunk_linker = EntityChunkLinker(graph_service)
        self.context_aware_retriever = ContextAwareRetriever(rag_service, graph_service)
        
        # Storage for entity-chunk links (in production, this would be persistent)
        self.entity_chunk_links: List[EntityChunkLink] = []
        self.links_created = False
    
    def process_document_with_entity_linking(self, file_path: str) -> Dict:
        """
        Process document with hybrid chunking and create entity-chunk links.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with processing results and entity-chunk links
        """
        import time
        start_time = time.time()
        
        # Step 1: Process document with hybrid chunking
        print(f"🏗️ Processing document with hybrid chunking: {file_path}")
        chunking_result = self.rag_service.process_document_hybrid_chunking(file_path)
        
        # Step 2: Create entity-chunk links
        print(f"🔗 Creating entity-chunk links...")
        chunks_with_metadata = chunking_result.get('chunks_with_metadata', [])
        
        if chunks_with_metadata:
            entity_chunk_links = self.entity_chunk_linker.create_entity_chunk_links(chunks_with_metadata)
            self.entity_chunk_links = entity_chunk_links
            self.links_created = True
            
            print(f"✅ Created {len(entity_chunk_links)} entity-chunk links")
            
            # Store links (in production, this would be persistent storage)
            self.entity_chunk_linker.store_entity_chunk_links(entity_chunk_links)
        else:
            print("⚠️ No chunks available for entity linking")
            entity_chunk_links = []
        
        processing_time = time.time() - start_time
        
        return {
            "file_path": file_path,
            "chunking_result": chunking_result,
            "entity_chunk_links": len(entity_chunk_links),
            "processing_time": processing_time,
            "links_created": self.links_created
        }
    
    def query_with_entity_context(self, query: str, user_page_limit: Optional[int] = None) -> EnhancedRAGResult:
        """
        Perform query with entity-chunk linking and context analysis.
        
        Args:
            query: User query
            user_page_limit: Optional page limit for spoiler protection
            
        Returns:
            EnhancedRAGResult with entity context
        """
        import time
        start_time = time.time()
        
        # Step 1: Try context-aware retrieval first
        if self.links_created:
            try:
                context_result = self.context_aware_retriever.retrieve_with_context(query, user_page_limit)
                
                if context_result.entity_name:
                    # Context-aware retrieval succeeded
                    processing_time = time.time() - start_time
                    
                    return EnhancedRAGResult(
                        query=query,
                        answer=context_result.answer,
                        source_pages=context_result.source_pages,
                        search_method=context_result.retrieval_method,
                        entities_found=[context_result.entity_name],
                        entity_contexts={context_result.entity_name: context_result},
                        chunk_links=context_result.relevant_chunks,
                        spoiler_protected=context_result.spoiler_protected,
                        user_page_limit=user_page_limit,
                        processing_time=processing_time
                    )
            except Exception as e:
                print(f"⚠️ Context-aware retrieval failed: {e}")
        
        # Step 2: Fallback to regular RAG query
        print("🔄 Falling back to regular RAG query")
        rag_result = self.rag_service.perform_rag_query(query, user_page_limit)
        
        # Step 3: Try to extract entities from the answer and find related chunks
        entities_found = self._extract_entities_from_query(query)
        entity_contexts = {}
        chunk_links = []
        
        for entity_name in entities_found:
            try:
                entity_context = self.context_aware_retriever.retrieve_with_context(
                    f"Tell me about {entity_name}", user_page_limit
                )
                entity_contexts[entity_name] = entity_context
                chunk_links.extend(entity_context.relevant_chunks)
            except Exception as e:
                print(f"⚠️ Could not get context for entity {entity_name}: {e}")
        
        processing_time = time.time() - start_time
        
        return EnhancedRAGResult(
            query=query,
            answer=rag_result.get("answer", ""),
            source_pages=rag_result.get("source_pages", []),
            search_method=rag_result.get("search_method", "rag_fallback"),
            entities_found=entities_found,
            entity_contexts=entity_contexts,
            chunk_links=chunk_links,
            spoiler_protected=user_page_limit is not None,
            user_page_limit=user_page_limit,
            processing_time=processing_time
        )
    
    def find_entity_passages(self, entity_name: str, context_type: Optional[ContextType] = None,
                           user_page_limit: Optional[int] = None) -> List[EntityChunkLink]:
        """
        Find specific passages where an entity is mentioned.
        
        Args:
            entity_name: Name of the entity
            context_type: Optional context type filter
            user_page_limit: Optional page limit for spoiler protection
            
        Returns:
            List of EntityChunkLink objects
        """
        if not self.links_created:
            print("⚠️ Entity-chunk links not created yet. Process a document first.")
            return []
        
        return self.context_aware_retriever.find_entity_passages(entity_name, context_type, user_page_limit)
    
    def get_entity_context_analysis(self, entity_name: str, user_page_limit: Optional[int] = None) -> Dict:
        """
        Get comprehensive context analysis for an entity.
        
        Args:
            entity_name: Name of the entity
            user_page_limit: Optional page limit for spoiler protection
            
        Returns:
            Dictionary with context analysis
        """
        if not self.links_created:
            return {"error": "Entity-chunk links not created yet"}
        
        # Get context analysis
        context_analysis = self.entity_chunk_linker.get_entity_context(entity_name, user_page_limit)
        
        # Get relationships with context
        relationships = self.context_aware_retriever.get_entity_relationships_with_context(entity_name, user_page_limit)
        
        return {
            "entity_name": entity_name,
            "entity_label": context_analysis.entity_label,
            "total_mentions": context_analysis.total_mentions,
            "context_types": {ct.value: count for ct, count in context_analysis.context_types.items()},
            "chunks_by_context": {
                ct.value: len(chunks) for ct, chunks in context_analysis.chunks_by_context.items()
            },
            "relationships": {
                related_entity: len(chunks) for related_entity, chunks in relationships.items()
            },
            "spoiler_protected": user_page_limit is not None,
            "user_page_limit": user_page_limit
        }
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Get a specific chunk by its ID.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            Chunk data if found, None otherwise
        """
        # This would typically query your chunk storage
        # For now, we'll search in the current chunks
        for chunk in self.rag_service.all_chunks_with_metadata:
            if chunk.get("chunk_id") == chunk_id:
                return chunk
        return None
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential entity names from query."""
        # Simple entity extraction - in production, use more sophisticated NLP
        import re
        
        # Look for capitalized words
        words = query.split()
        entities = []
        
        for word in words:
            if word[0].isupper() and len(word) > 2:
                # Clean up punctuation
                clean_word = re.sub(r'[^\w\s]', '', word)
                if clean_word:
                    entities.append(clean_word)
        
        return entities
    
    def get_entity_chunk_links_summary(self) -> Dict:
        """Get summary of entity-chunk links."""
        if not self.links_created:
            return {"links_created": False}
        
        # Count entities
        entity_counts = {}
        context_type_counts = {}
        
        for link in self.entity_chunk_links:
            entity_counts[link.entity_name] = entity_counts.get(link.entity_name, 0) + 1
            context_type_counts[link.context_type.value] = context_type_counts.get(link.context_type.value, 0) + 1
        
        return {
            "links_created": True,
            "total_links": len(self.entity_chunk_links),
            "unique_entities": len(entity_counts),
            "entity_counts": entity_counts,
            "context_type_counts": context_type_counts,
            "top_entities": sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
