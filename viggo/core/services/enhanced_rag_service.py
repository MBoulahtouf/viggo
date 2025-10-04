"""
Enhanced RAG Service for Viggo - Integrates hybrid chunking strategy
with multi-level retrieval for improved lore knowledge exploration.

This service implements:
1. Hierarchical retrieval (book → chapter → passage → sentence)
2. Dynamic chunking based on query complexity
3. Noise reduction through content filtering and entity enhancement
4. Multi-source fusion with weighted scoring
"""

import os
import pickle
import asyncio
from typing import List, Dict, Tuple, Optional, Set
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2, write_index, read_index
from groq import Groq

from viggo.core.config import settings
from viggo.core.services.hybrid_chunking_service import (
    HybridChunkingService, ChunkLevel, ChunkType, ChunkingConfig
)
from viggo.core.services.hybrid_retriever import HybridRetriever
from viggo.core.services.graph_service import GraphService
from viggo.core.services.hybrid_search_service import HybridSearchService
from viggo.core.processors import DocumentProcessorFactory


class EnhancedRAGService:
    """
    Enhanced RAG service that implements hierarchical chunking and retrieval
    for improved book lore knowledge exploration.
    """
    
    def __init__(self, graph_service: GraphService = None, model_name: str = "all-MiniLM-L6-v2"):
        self.graph_service = graph_service
        self.model = SentenceTransformer(model_name)
        self.groq_client = Groq(api_key=settings.groq_api_key)
        self.document_processor_factory = DocumentProcessorFactory()
        
        # Initialize hybrid chunking service
        self.chunking_config = ChunkingConfig()
        self.hybrid_chunking = HybridChunkingService(config=self.chunking_config)
        
        # Initialize other services
        self.hybrid_search_service = HybridSearchService(model_name)
        self.hybrid_retriever = None  # Will be initialized after document processing
        
        # Storage for hierarchical chunks and indices
        self.hierarchical_chunks = {}  # level -> chunks
        self.hierarchical_indices = {}  # level -> FAISS index
        self.chunk_metadata = {}  # chunk_id -> metadata
        self.chunk_hierarchy = {}  # parent_id -> children_ids
        
        # File paths for persistence
        self.index_base_path = "enhanced_faiss_index"
        self.chunks_base_path = "enhanced_chunks_data"
        
        # Load existing data if available
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing hierarchical chunks and indices if available."""
        try:
            # Load hierarchical chunks
            chunks_file = f"{self.chunks_base_path}.pkl"
            if os.path.exists(chunks_file):
                with open(chunks_file, 'rb') as f:
                    data = pickle.load(f)
                    self.hierarchical_chunks = data.get("chunks", {})
                    self.chunk_metadata = data.get("metadata", {})
                    self.chunk_hierarchy = data.get("hierarchy", {})
                print(f"✅ Loaded existing hierarchical chunks: {sum(len(chunks) for chunks in self.hierarchical_chunks.values())} total")
            
            # Load hierarchical indices
            for level in ChunkLevel:
                index_file = f"{self.index_base_path}_{level.value}.bin"
                if os.path.exists(index_file):
                    self.hierarchical_indices[level.value] = read_index(index_file)
                    print(f"✅ Loaded FAISS index for {level.value} level")
            
            # Initialize hybrid retriever if data is available
            if self.hierarchical_chunks:
                self._initialize_hybrid_retriever()
                
        except Exception as e:
            print(f"⚠️ Could not load existing data: {e}")
    
    def _save_data(self):
        """Save hierarchical chunks and indices to disk."""
        try:
            # Save hierarchical chunks
            chunks_file = f"{self.chunks_base_path}.pkl"
            with open(chunks_file, 'wb') as f:
                pickle.dump({
                    "chunks": self.hierarchical_chunks,
                    "metadata": self.chunk_metadata,
                    "hierarchy": self.chunk_hierarchy
                }, f)
            
            # Save hierarchical indices
            for level, index in self.hierarchical_indices.items():
                index_file = f"{self.index_base_path}_{level}.bin"
                write_index(index, index_file)
            
            print("✅ Saved enhanced RAG data to disk")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def _initialize_hybrid_retriever(self):
        """Initialize hybrid retriever with hierarchical data."""
        if self.hierarchical_chunks:
            # Use passage-level chunks for hybrid retriever
            passage_chunks = self.hierarchical_chunks.get(ChunkLevel.PASSAGE.value, [])
            if passage_chunks:
                # Convert to format expected by hybrid retriever
                all_chunks_with_metadata = []
                for chunk in passage_chunks:
                    all_chunks_with_metadata.append({
                        "content": chunk["content"],
                        "page": chunk["metadata"].page_number,
                        "entities": chunk["metadata"].entities,
                        "entity_labels": [entity["label"] for entity in chunk["metadata"].entities],
                        "chapter_title": chunk["metadata"].chapter_title,
                        "chunk_type": chunk["chunk_type"],
                        "document_metadata": {}
                    })
                
                # Create FAISS index for passages
                if all_chunks_with_metadata:
                    documents = [chunk["content"] for chunk in all_chunks_with_metadata]
                    embeddings = self.model.encode(documents)
                    index = IndexFlatL2(embeddings.shape[1])
                    index.add(embeddings)
                    
                    self.hybrid_retriever = HybridRetriever(
                        vector_index=index,
                        all_chunks_with_metadata=all_chunks_with_metadata,
                        model_name="all-MiniLM-L6-v2"
                    )
                    
                    if self.graph_service:
                        self.hybrid_retriever.graph_service = self.graph_service
                    
                    print("✅ Initialized hybrid retriever with hierarchical data")
    
    def process_document_enhanced(self, file_path: str) -> Dict:
        """
        Process document with enhanced hierarchical chunking strategy.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with processing results and statistics
        """
        print(f"🔍 Processing document with enhanced hierarchical chunking: {file_path}")
        
        # Step 1: Extract text using document processor
        all_pages_data = self.document_processor_factory.process_document(file_path)
        
        # Step 2: Apply hierarchical chunking
        chunking_result = self.hybrid_chunking.chunk_document_hierarchical(all_pages_data)
        
        # Step 3: Build hierarchical FAISS indices
        self._build_hierarchical_indices(chunking_result["chunks"])
        
        # Step 4: Store data
        self.hierarchical_chunks = chunking_result["chunks"]
        self.chunk_metadata = chunking_result["metadata"]
        self.chunk_hierarchy = chunking_result["hierarchy"]
        
        # Step 5: Initialize hybrid retriever
        self._initialize_hybrid_retriever()
        
        # Step 6: Save to disk
        self._save_data()
        
        # Step 7: Index in Azure Cognitive Search
        self._index_in_azure_search(chunking_result["chunks"])
        
        print(f"✅ Enhanced document processing complete:")
        print(f"   Total chunks: {chunking_result['statistics']['total_chunks']}")
        print(f"   Chapters: {len(chunking_result['chunks'].get(ChunkLevel.CHAPTER.value, []))}")
        print(f"   Passages: {len(chunking_result['chunks'].get(ChunkLevel.PASSAGE.value, []))}")
        print(f"   Overlapping: {len(chunking_result['chunks'].get(ChunkLevel.SENTENCE.value, []))}")
        
        return {
            "file_path": file_path,
            "chunking_result": chunking_result,
            "hierarchical_indices": {level: index.ntotal for level, index in self.hierarchical_indices.items()},
            "processing_stats": chunking_result["statistics"]
        }
    
    def _build_hierarchical_indices(self, hierarchical_chunks: Dict[str, List[Dict]]):
        """Build FAISS indices for each hierarchical level."""
        self.hierarchical_indices.clear()
        
        for level, chunks in hierarchical_chunks.items():
            if not chunks:
                continue
            
            # Extract documents and metadata
            documents = [chunk["content"] for chunk in chunks]
            
            if not documents:
                continue
            
            # Generate embeddings
            embeddings = self.model.encode(documents)
            
            # Create FAISS index
            index = IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            
            self.hierarchical_indices[level] = index
            
            print(f"✅ Built FAISS index for {level} level: {len(documents)} chunks")
    
    def _index_in_azure_search(self, hierarchical_chunks: Dict[str, List[Dict]]):
        """Index hierarchical chunks in Azure Cognitive Search."""
        try:
            # Create index if it doesn't exist
            if not self.hybrid_search_service.create_index():
                print("Failed to create Azure Cognitive Search index")
                return
            
            # Prepare documents for indexing (use passage-level chunks)
            passage_chunks = hierarchical_chunks.get(ChunkLevel.PASSAGE.value, [])
            if not passage_chunks:
                print("No passage chunks to index")
                return
            
            search_docs = []
            for chunk in passage_chunks:
                metadata = chunk["metadata"]
                search_doc = {
                    "content": chunk["content"],
                    "page": metadata.page_number,
                    "word_count": metadata.word_count,
                    "char_count": metadata.char_count,
                    "entities": [entity["text"] for entity in metadata.entities],
                    "entity_labels": [entity["label"] for entity in metadata.entities],
                    "chapter_title": metadata.chapter_title,
                    "chunk_type": chunk["chunk_type"],
                    "content_type": metadata.content_type,
                    "lore_significance": metadata.lore_significance,
                    "document_metadata": {}
                }
                search_docs.append(search_doc)
            
            # Index documents
            success = self.hybrid_search_service.index_documents(search_docs)
            
            if success:
                print(f"✅ Successfully indexed {len(search_docs)} passage chunks in Azure Cognitive Search")
            else:
                print("❌ Failed to index documents in Azure Cognitive Search")
                
        except Exception as e:
            print(f"❌ Error indexing in Azure Search: {e}")
    
    def query_hierarchical(self, query: str, level: ChunkLevel = ChunkLevel.PASSAGE, 
                          top_k: int = 5, page_filter: Optional[int] = None) -> Dict:
        """
        Perform hierarchical query with dynamic level selection.
        
        Args:
            query: User query
            level: Hierarchical level to search
            top_k: Number of results to return
            page_filter: Optional page number filter
            
        Returns:
            Dictionary with query results and metadata
        """
        print(f"🔍 Hierarchical query: '{query}' at {level.value} level")
        
        # Step 1: Determine optimal search level based on query complexity
        optimal_level = self._determine_optimal_level(query, level)
        print(f"📊 Optimal search level: {optimal_level.value}")
        
        # Step 2: Search at optimal level
        level_results = self._search_at_level(query, optimal_level, top_k, page_filter)
        
        # Step 3: If needed, drill down to more specific levels
        if optimal_level != ChunkLevel.SENTENCE and len(level_results) < top_k:
            drill_down_results = self._drill_down_search(query, optimal_level, top_k - len(level_results), page_filter)
            level_results.extend(drill_down_results)
        
        # Step 4: If needed, expand to broader context
        if len(level_results) < top_k:
            expanded_results = self._expand_context_search(query, optimal_level, top_k - len(level_results), page_filter)
            level_results.extend(expanded_results)
        
        # Step 5: Rank and deduplicate results
        ranked_results = self._rank_and_deduplicate_results(level_results)
        
        # Step 6: Generate answer with hierarchical context
        answer = self._generate_hierarchical_answer(query, ranked_results)
        
        return {
            "query": query,
            "answer": answer,
            "results": ranked_results[:top_k],
            "search_level": optimal_level.value,
            "total_results": len(ranked_results),
            "search_metadata": {
                "levels_searched": [optimal_level.value],
                "drill_down_performed": len(level_results) < top_k,
                "context_expansion_performed": len(level_results) < top_k
            }
        }
    
    def _determine_optimal_level(self, query: str, requested_level: ChunkLevel) -> ChunkLevel:
        """Determine optimal search level based on query characteristics."""
        query_lower = query.lower()
        
        # Check for specific entity queries (use passage level)
        entity_indicators = ['who is', 'what is', 'where is', 'when did', 'how did']
        if any(indicator in query_lower for indicator in entity_indicators):
            return ChunkLevel.PASSAGE
        
        # Check for broad context queries (use chapter level)
        context_indicators = ['tell me about', 'explain', 'describe', 'overview', 'summary']
        if any(indicator in query_lower for indicator in context_indicators):
            return ChunkLevel.CHAPTER
        
        # Check for detailed analysis queries (use sentence level)
        detail_indicators = ['exactly', 'precisely', 'specifically', 'in detail', 'word for word']
        if any(indicator in query_lower for indicator in detail_indicators):
            return ChunkLevel.SENTENCE
        
        # Default to requested level
        return requested_level
    
    def _search_at_level(self, query: str, level: ChunkLevel, top_k: int, 
                        page_filter: Optional[int] = None) -> List[Dict]:
        """Search at a specific hierarchical level."""
        level_chunks = self.hierarchical_chunks.get(level.value, [])
        level_index = self.hierarchical_indices.get(level.value)
        
        if not level_chunks or not level_index:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])
            
            # Search FAISS index
            distances, indices = level_index.search(query_embedding, top_k)
            
            # Get results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(level_chunks):
                    chunk = level_chunks[idx]
                    metadata = chunk["metadata"]
                    
                    # Apply page filter if specified
                    if page_filter and metadata.page_number > page_filter:
                        continue
                    
                    results.append({
                        "content": chunk["content"],
                        "page": metadata.page_number,
                        "score": 1.0 - distance,  # Convert distance to similarity
                        "level": level.value,
                        "chunk_type": chunk["chunk_type"],
                        "lore_significance": metadata.lore_significance,
                        "entities": metadata.entities,
                        "chapter_title": metadata.chapter_title,
                        "chunk_id": chunk["id"]
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching at {level.value} level: {e}")
            return []
    
    def _drill_down_search(self, query: str, current_level: ChunkLevel, 
                          top_k: int, page_filter: Optional[int] = None) -> List[Dict]:
        """Drill down to more specific levels for additional results."""
        drill_down_results = []
        
        # Define drill-down hierarchy
        drill_down_map = {
            ChunkLevel.CHAPTER: ChunkLevel.PASSAGE,
            ChunkLevel.PASSAGE: ChunkLevel.SENTENCE,
            ChunkLevel.SECTION: ChunkLevel.PASSAGE
        }
        
        next_level = drill_down_map.get(current_level)
        if next_level:
            drill_down_results = self._search_at_level(query, next_level, top_k, page_filter)
        
        return drill_down_results
    
    def _expand_context_search(self, query: str, current_level: ChunkLevel, 
                              top_k: int, page_filter: Optional[int] = None) -> List[Dict]:
        """Expand to broader context levels for additional results."""
        expand_results = []
        
        # Define context expansion hierarchy
        expand_map = {
            ChunkLevel.PASSAGE: ChunkLevel.CHAPTER,
            ChunkLevel.SENTENCE: ChunkLevel.PASSAGE,
            ChunkLevel.SECTION: ChunkLevel.CHAPTER
        }
        
        broader_level = expand_map.get(current_level)
        if broader_level:
            expand_results = self._search_at_level(query, broader_level, top_k, page_filter)
        
        return expand_results
    
    def _rank_and_deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Rank and deduplicate results from multiple levels."""
        if not results:
            return []
        
        # Remove duplicates based on content similarity
        unique_results = []
        seen_content = set()
        
        for result in results:
            content_key = result["content"][:100]  # Use first 100 chars as key
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
        
        # Sort by combined score (similarity + lore significance)
        def combined_score(result):
            similarity_score = result.get("score", 0.0)
            lore_score = result.get("lore_significance", 0.0)
            return similarity_score * 0.7 + lore_score * 0.3
        
        ranked_results = sorted(unique_results, key=combined_score, reverse=True)
        
        return ranked_results
    
    def _generate_hierarchical_answer(self, query: str, results: List[Dict]) -> str:
        """Generate answer using hierarchical context."""
        if not results:
            return "No relevant information found in the document."
        
        # Create hierarchical context
        context_parts = []
        
        # Group results by level
        results_by_level = {}
        for result in results:
            level = result.get("level", "unknown")
            if level not in results_by_level:
                results_by_level[level] = []
            results_by_level[level].append(result)
        
        # Build context from different levels
        for level in [ChunkLevel.CHAPTER.value, ChunkLevel.PASSAGE.value, ChunkLevel.SENTENCE.value]:
            if level in results_by_level:
                level_results = results_by_level[level]
                context_parts.append(f"\n{level.upper()} CONTEXT:")
                
                for i, result in enumerate(level_results[:3]):  # Limit to top 3 per level
                    context_parts.append(f"{i+1}. Page {result.get('page', 'N/A')}: {result['content'][:300]}...")
        
        # Combine context
        full_context = "\n".join(context_parts)
        
        # Generate answer with LLM
        prompt = f"""You are Viggo, a lore expert. Answer the following question using the provided hierarchical context from the book:

Question: {query}

Hierarchical Context:
{full_context}

Instructions:
- Use the most relevant context from the appropriate hierarchical level
- If multiple levels provide information, synthesize them coherently
- Prioritize information from higher lore significance scores
- Provide specific page references when possible
- Maintain the narrative flow and lore consistency

Answer:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer."
    
    def get_chunking_statistics(self) -> Dict:
        """Get comprehensive chunking statistics."""
        if not self.hierarchical_chunks:
            return {"error": "No chunks available"}
        
        stats = {
            "total_chunks": sum(len(chunks) for chunks in self.hierarchical_chunks.values()),
            "chunks_by_level": {level: len(chunks) for level, chunks in self.hierarchical_chunks.items()},
            "indices_available": list(self.hierarchical_indices.keys()),
            "chunking_config": {
                "max_chapter_words": self.chunking_config.max_chapter_words,
                "max_passage_words": self.chunking_config.max_passage_words,
                "passage_overlap_ratio": self.chunking_config.passage_overlap_ratio,
                "critical_lore_threshold": self.chunking_config.critical_lore_threshold
            }
        }
        
        # Add detailed statistics if available
        if hasattr(self.hybrid_chunking, 'get_chunking_summary'):
            stats.update(self.hybrid_chunking.get_chunking_summary())
        
        return stats
    
    def get_critical_lore_chunks(self, threshold: float = 0.7) -> List[Dict]:
        """Get chunks with high lore significance."""
        critical_chunks = []
        
        for level, chunks in self.hierarchical_chunks.items():
            for chunk in chunks:
                if chunk["metadata"].lore_significance >= threshold:
                    critical_chunks.append({
                        "id": chunk["id"],
                        "content": chunk["content"],
                        "level": level,
                        "lore_significance": chunk["metadata"].lore_significance,
                        "page": chunk["metadata"].page_number,
                        "chapter_title": chunk["metadata"].chapter_title,
                        "entities": chunk["metadata"].entities
                    })
        
        # Sort by lore significance
        critical_chunks.sort(key=lambda x: x["lore_significance"], reverse=True)
        
        return critical_chunks
    
    def clear_data(self):
        """Clear all stored data and indices."""
        self.hierarchical_chunks.clear()
        self.hierarchical_indices.clear()
        self.chunk_metadata.clear()
        self.chunk_hierarchy.clear()
        self.hybrid_retriever = None
        
        # Remove files
        try:
            chunks_file = f"{self.chunks_base_path}.pkl"
            if os.path.exists(chunks_file):
                os.remove(chunks_file)
            
            for level in ChunkLevel:
                index_file = f"{self.index_base_path}_{level.value}.bin"
                if os.path.exists(index_file):
                    os.remove(index_file)
            
            print("✅ Cleared all enhanced RAG data")
        except Exception as e:
            print(f"❌ Error clearing data: {e}")
