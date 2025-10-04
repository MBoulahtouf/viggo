"""
Context-Aware Retrieval Service for Viggo

This service provides user-centric access to entity mentions with context analysis,
integrating with the existing spoiler safeguard and hybrid chunking architecture.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from viggo.core.services.entity_chunk_linker import (
    EntityChunkLinker, EntityContextResult, ContextType, EntityChunkLink
)
from viggo.core.services.graph_service import GraphService
from viggo.core.services.rag_service import RAGService


class QueryIntent(Enum):
    """Types of user query intents."""
    FIND_ENTITY_MENTIONS = "find_mentions"      # "Where is X mentioned?"
    GET_ENTITY_DESCRIPTION = "get_description"  # "What is X?"
    EXPLORE_ENTITY_RELATIONSHIPS = "explore_relationships"  # "Who is X related to?"
    GET_ENTITY_CONTEXT = "get_context"          # "Tell me about X in the story"
    FIND_SPECIFIC_PASSAGE = "find_passage"      # "Show me the passage where X does Y"


@dataclass
class ContextAwareResult:
    """Result of context-aware retrieval."""
    query: str
    intent: QueryIntent
    entity_name: str
    entity_label: str
    user_page_limit: Optional[int]
    spoiler_protected: bool
    
    # Results
    context_analysis: EntityContextResult
    relevant_chunks: List[EntityChunkLink]
    answer: str
    source_pages: List[int]
    
    # Metadata
    total_mentions: int
    context_types_found: List[ContextType]
    retrieval_method: str


class ContextAwareRetriever:
    """
    Service that provides context-aware retrieval with user-centric access patterns.
    """
    
    def __init__(self, rag_service: RAGService, graph_service: GraphService):
        self.rag_service = rag_service
        self.graph_service = graph_service
        self.entity_chunk_linker = EntityChunkLinker(graph_service)
        
        # Query intent patterns
        self.intent_patterns = {
            QueryIntent.FIND_ENTITY_MENTIONS: [
                r"where\s+(?:is|are)\s+(\w+)\s+mentioned",
                r"show\s+me\s+where\s+(\w+)\s+appears",
                r"find\s+mentions\s+of\s+(\w+)",
                r"where\s+does\s+(\w+)\s+come\s+up"
            ],
            QueryIntent.GET_ENTITY_DESCRIPTION: [
                r"what\s+(?:is|are)\s+(\w+)",
                r"who\s+(?:is|are)\s+(\w+)",
                r"describe\s+(\w+)",
                r"tell\s+me\s+about\s+(\w+)"
            ],
            QueryIntent.EXPLORE_ENTITY_RELATIONSHIPS: [
                r"who\s+(?:is|are)\s+(\w+)\s+(?:related|connected)\s+to",
                r"what\s+(?:is|are)\s+(\w+)\s+(?:relationship|connection)",
                r"(\w+)\s+(?:and|with)\s+",
                r"(\w+)\s+(?:friend|ally|enemy|rival)"
            ],
            QueryIntent.GET_ENTITY_CONTEXT: [
                r"(\w+)\s+(?:in|during|throughout)\s+(?:the\s+)?story",
                r"(\w+)\s+(?:role|part|importance)",
                r"(\w+)\s+(?:significance|meaning)"
            ],
            QueryIntent.FIND_SPECIFIC_PASSAGE: [
                r"show\s+me\s+(?:the\s+)?passage\s+where\s+(\w+)",
                r"find\s+(?:the\s+)?part\s+where\s+(\w+)",
                r"(\w+)\s+(?:does|says|thinks|feels)\s+",
                r"when\s+(\w+)\s+(?:does|says|thinks|feels)"
            ]
        }
    
    def retrieve_with_context(self, query: str, user_page_limit: Optional[int] = None) -> ContextAwareResult:
        """
        Perform context-aware retrieval with spoiler protection.
        
        Args:
            query: User query
            user_page_limit: Optional page limit for spoiler protection
            
        Returns:
            ContextAwareResult with context analysis
        """
        # Step 1: Analyze query intent and extract entity
        intent, entity_name = self._analyze_query_intent(query)
        
        if not entity_name:
            # Fallback to regular RAG query
            return self._fallback_to_rag(query, user_page_limit)
        
        # Step 2: Get entity context with spoiler protection
        context_analysis = self.entity_chunk_linker.get_entity_context(entity_name, user_page_limit)
        
        # Step 3: Get relevant chunks based on intent
        relevant_chunks = self._get_chunks_by_intent(intent, entity_name, context_analysis, user_page_limit)
        
        # Step 4: Generate contextual answer
        answer = self._generate_contextual_answer(query, intent, entity_name, context_analysis, relevant_chunks)
        
        # Step 5: Extract source pages
        source_pages = sorted(list(set(chunk.page_number for chunk in relevant_chunks)))
        
        return ContextAwareResult(
            query=query,
            intent=intent,
            entity_name=entity_name,
            entity_label=context_analysis.entity_label,
            user_page_limit=user_page_limit,
            spoiler_protected=user_page_limit is not None,
            context_analysis=context_analysis,
            relevant_chunks=relevant_chunks,
            answer=answer,
            source_pages=source_pages,
            total_mentions=context_analysis.total_mentions,
            context_types_found=list(context_analysis.context_types.keys()),
            retrieval_method="context_aware"
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
        return self.entity_chunk_linker.find_entity_passages(entity_name, context_type, user_page_limit)
    
    def get_entity_relationships_with_context(self, entity_name: str, 
                                            user_page_limit: Optional[int] = None) -> Dict[str, List[EntityChunkLink]]:
        """
        Get entity relationships with context from chunks.
        
        Args:
            entity_name: Name of the entity
            user_page_limit: Optional page limit for spoiler protection
            
        Returns:
            Dictionary mapping related entities to their context chunks
        """
        return self.entity_chunk_linker.get_entity_relationships_in_context(entity_name, user_page_limit)
    
    def _analyze_query_intent(self, query: str) -> Tuple[QueryIntent, Optional[str]]:
        """Analyze query intent and extract entity name."""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                import re
                match = re.search(pattern, query_lower)
                if match:
                    entity_name = match.group(1).strip()
                    return intent, entity_name
        
        # Try to extract entity name using simple patterns
        entity_name = self._extract_entity_name_simple(query)
        if entity_name:
            return QueryIntent.GET_ENTITY_CONTEXT, entity_name
        
        return QueryIntent.GET_ENTITY_CONTEXT, None
    
    def _extract_entity_name_simple(self, query: str) -> Optional[str]:
        """Simple entity name extraction."""
        # Look for capitalized words (potential entity names)
        import re
        words = query.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                # Clean up punctuation
                clean_word = re.sub(r'[^\w\s]', '', word)
                if clean_word:
                    return clean_word
        return None
    
    def _get_chunks_by_intent(self, intent: QueryIntent, entity_name: str, 
                            context_analysis: EntityContextResult, 
                            user_page_limit: Optional[int]) -> List[EntityChunkLink]:
        """Get relevant chunks based on query intent."""
        if intent == QueryIntent.FIND_ENTITY_MENTIONS:
            # Return all mentions, sorted by context score
            return sorted(context_analysis.spoiler_safe_chunks, 
                         key=lambda x: x.context_score, reverse=True)
        
        elif intent == QueryIntent.GET_ENTITY_DESCRIPTION:
            # Prioritize descriptive context
            descriptive_chunks = context_analysis.chunks_by_context.get(ContextType.DESCRIPTION, [])
            if descriptive_chunks:
                return descriptive_chunks[:5]  # Top 5 descriptive chunks
            else:
                return context_analysis.spoiler_safe_chunks[:5]
        
        elif intent == QueryIntent.EXPLORE_ENTITY_RELATIONSHIPS:
            # Prioritize relationship context
            relationship_chunks = context_analysis.chunks_by_context.get(ContextType.RELATIONSHIP, [])
            if relationship_chunks:
                return relationship_chunks[:5]
            else:
                return context_analysis.spoiler_safe_chunks[:5]
        
        elif intent == QueryIntent.FIND_SPECIFIC_PASSAGE:
            # Return chunks with highest context scores
            return sorted(context_analysis.spoiler_safe_chunks, 
                         key=lambda x: (x.context_score, x.lore_significance), reverse=True)[:3]
        
        else:  # GET_ENTITY_CONTEXT
            # Return a mix of context types
            all_chunks = context_analysis.spoiler_safe_chunks
            return sorted(all_chunks, key=lambda x: (x.context_score, x.lore_significance), reverse=True)[:5]
    
    def _generate_contextual_answer(self, query: str, intent: QueryIntent, entity_name: str,
                                  context_analysis: EntityContextResult, 
                                  relevant_chunks: List[EntityChunkLink]) -> str:
        """Generate contextual answer based on intent and chunks."""
        if not relevant_chunks:
            return f"I couldn't find any mentions of {entity_name} in the story."
        
        # Build context from chunks
        context_parts = []
        for chunk in relevant_chunks[:3]:  # Limit to top 3 chunks
            context_parts.append(f"Page {chunk.page_number}: {chunk.surrounding_text}")
        
        full_context = "\n\n".join(context_parts)
        
        # Generate answer based on intent
        if intent == QueryIntent.FIND_ENTITY_MENTIONS:
            prompt = f"""Based on the following context, tell me where {entity_name} is mentioned in the story:

Context:
{full_context}

Provide a concise answer about where {entity_name} appears, including page numbers."""
        
        elif intent == QueryIntent.GET_ENTITY_DESCRIPTION:
            prompt = f"""Based on the following context, describe {entity_name}:

Context:
{full_context}

Provide a description of {entity_name} based on the story context."""
        
        elif intent == QueryIntent.EXPLORE_ENTITY_RELATIONSHIPS:
            prompt = f"""Based on the following context, tell me about {entity_name}'s relationships:

Context:
{full_context}

Describe {entity_name}'s relationships with other characters or entities."""
        
        else:
            prompt = f"""Based on the following context, answer this question about {entity_name}:

Question: {query}

Context:
{full_context}

Provide a helpful answer based on the story context."""
        
        # Use the existing LLM generation from RAG service
        try:
            return self.rag_service._generate_answer_with_llm(query, full_context)
        except:
            # Fallback to simple answer
            return f"{entity_name} appears in {len(relevant_chunks)} passages, including pages {', '.join(map(str, sorted(set(chunk.page_number for chunk in relevant_chunks))))}."
    
    def _fallback_to_rag(self, query: str, user_page_limit: Optional[int]) -> ContextAwareResult:
        """Fallback to regular RAG query when context-aware retrieval fails."""
        rag_result = self.rag_service.perform_rag_query(query, user_page_limit)
        
        return ContextAwareResult(
            query=query,
            intent=QueryIntent.GET_ENTITY_CONTEXT,
            entity_name="",
            entity_label="",
            user_page_limit=user_page_limit,
            spoiler_protected=user_page_limit is not None,
            context_analysis=EntityContextResult("", "", 0, {}, {}, []),
            relevant_chunks=[],
            answer=rag_result.get("answer", ""),
            source_pages=rag_result.get("source_pages", []),
            total_mentions=0,
            context_types_found=[],
            retrieval_method="rag_fallback"
        )
