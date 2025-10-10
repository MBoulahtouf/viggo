"""
Concrete implementation of RAG service following SOLID principles.
"""

import time
from typing import Any

from viggo.core.services.interfaces.generation import (
    GenerationContext,
)
from viggo.core.services.interfaces.rag import (
    IndexingResult,
    RAGConfig,
    RAGResult,
    RAGService,
)
from viggo.core.services.interfaces.retrieval import QueryContext


class ConcreteRAGService(RAGService):
    """Concrete implementation of RAG service following SOLID principles."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.document_processor_factory = config.document_processor_factory
        self.chunking_service = config.chunking_service
        self.hybrid_retriever = config.hybrid_retriever
        self.generation_service = config.generation_service
        self.vector_storage = config.vector_storage
        self.graph_storage = config.graph_storage
        self.cache_storage = config.cache_storage

        # Initialize retrievers with storage backends
        self._initialize_retrievers()

    def _initialize_retrievers(self):
        """Initialize retrievers with appropriate storage backends."""
        # This would be done by the orchestrator, but we can set up basic connections here
        pass

    def index_document(self, document_path: str) -> IndexingResult:
        """Index a document for retrieval."""
        start_time = time.time()

        try:
            # Step 1: Process document
            processor = self.document_processor_factory.get_processor(document_path)
            if not processor:
                return IndexingResult(
                    document_path=document_path,
                    chunks_created=0,
                    entities_extracted=0,
                    relationships_found=0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=f"Unsupported file format: {document_path}"
                )

            # Get document pages
            pages = processor.process_document(document_path)
            if not pages:
                return IndexingResult(
                    document_path=document_path,
                    chunks_created=0,
                    entities_extracted=0,
                    relationships_found=0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="No pages found in document"
                )

            # Convert pages to expected format
            pages_data = []
            for page in pages:
                pages_data.append({
                    'content': page.content,
                    'page': page.page_number,
                    'metadata': page.metadata
                })

            # Step 2: Chunk document
            chunking_result = self.chunking_service.chunk_document(pages_data)

            # Step 3: Store in vector storage
            self._store_chunks_in_vector_storage(chunking_result)

            # Step 4: Store entities in graph storage
            entities_count, relationships_count = self._store_entities_in_graph_storage(
                document_path, chunking_result
            )

            processing_time = time.time() - start_time

            return IndexingResult(
                document_path=document_path,
                chunks_created=chunking_result.statistics.get('total_chunks', 0),
                entities_extracted=entities_count,
                relationships_found=relationships_count,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            return IndexingResult(
                document_path=document_path,
                chunks_created=0,
                entities_extracted=0,
                relationships_found=0,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def query(self, query: str, context: QueryContext | None = None) -> RAGResult:
        """Query the RAG system."""
        start_time = time.time()

        try:
            # Create query context if not provided
            if context is None:
                context = QueryContext(query=query)

            # Step 1: Retrieve relevant content
            retrieval_results = self.hybrid_retriever.retrieve_hybrid(context)

            if not retrieval_results:
                # Check if retrievers are available
                available_sources = self.hybrid_retriever.get_available_sources() if hasattr(self.hybrid_retriever, 'get_available_sources') else []

                # Try a simpler fallback approach
                if hasattr(self.hybrid_retriever, 'vector_index') and self.hybrid_retriever.vector_index is not None:
                    # Direct semantic search as fallback
                    try:
                        from sentence_transformers import SentenceTransformer
                        model = SentenceTransformer("all-MiniLM-L6-v2")
                        query_embedding = model.encode([query])

                        if hasattr(self.hybrid_retriever, 'all_chunks_with_metadata') and self.hybrid_retriever.all_chunks_with_metadata:
                            # Simple keyword matching as last resort
                            query_words = query.lower().split()
                            for chunk in self.hybrid_retriever.all_chunks_with_metadata[:10]:  # Check first 10 chunks
                                content = chunk.get('content', '').lower()
                                if any(word in content for word in query_words):
                                    # Create a simple result
                                    from viggo.core.services.interfaces.retrieval import (
                                        RetrievalResult,
                                        RetrievalSource,
                                    )
                                    retrieval_results = [RetrievalResult(
                                        content=chunk['content'],
                                        score=0.8,
                                        source=RetrievalSource.SEMANTIC,
                                        page_number=chunk.get('page', 1),
                                        chunk_id=chunk.get('id', 'fallback')
                                    )]
                                    break
                    except Exception as e:
                        print(f"Fallback retrieval failed: {e}")

                if not retrieval_results:
                    return RAGResult(
                        query=query,
                        answer="No relevant information found in the document. Please try re-uploading the document or ask a different question.",
                        source_pages=[],
                        confidence_score=0.0,
                        processing_time=time.time() - start_time,
                        metadata={
                            "retrieval_results": 0,
                            "available_sources": available_sources,
                            "debug_info": "No retrievers returned results"
                        }
                    )

            # Step 2: Prepare generation context
            retrieved_content = []
            source_pages = set()
            citations = []

            for result in retrieval_results:
                retrieved_content.append({
                    'content': result.content,
                    'page': result.page_number,
                    'source': result.source.value,
                    'score': result.score,
                    'metadata': result.metadata
                })

                if result.page_number:
                    source_pages.add(result.page_number)

                if result.chunk_id:
                    citations.append(f"Chunk {result.chunk_id}")

            # Step 3: Generate response
            generation_context = GenerationContext(
                query=query,
                retrieved_content=retrieved_content,
                user_context=context.additional_filters
            )

            generation_result = self.generation_service.generate_response(generation_context)

            processing_time = time.time() - start_time

            return RAGResult(
                query=query,
                answer=generation_result.generated_text,
                source_pages=sorted(list(source_pages)),
                confidence_score=generation_result.confidence_score,
                processing_time=processing_time,
                metadata={
                    "retrieval_results": len(retrieval_results),
                    "generation_model": generation_result.model_used.value,
                    "sources_used": [r.source.value for r in retrieval_results]
                },
                citations=citations
            )

        except Exception as e:
            return RAGResult(
                query=query,
                answer=f"Error processing query: {str(e)}",
                source_pages=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    def update_document(self, document_path: str) -> IndexingResult:
        """Update an existing document index."""
        # For now, treat update as re-indexing
        # In a more sophisticated implementation, we would:
        # 1. Compare document versions
        # 2. Update only changed chunks
        # 3. Maintain incremental indexing
        return self.index_document(document_path)

    def delete_document(self, document_path: str) -> bool:
        """Delete a document from the index."""
        try:
            # This would require tracking which chunks belong to which document
            # For now, this is a placeholder implementation
            print(f"Document deletion not yet implemented for: {document_path}")
            return False
        except Exception as e:
            print(f"Error deleting document {document_path}: {e}")
            return False

    def get_system_status(self) -> dict[str, Any]:
        """Get the status of the RAG system."""
        try:
            status = {
                "vector_storage": {
                    "available": self.vector_storage is not None,
                    "vector_count": self.vector_storage.get_vector_count() if self.vector_storage else 0
                },
                "graph_storage": {
                    "available": self.graph_storage is not None
                },
                "cache_storage": {
                    "available": self.cache_storage is not None,
                    "stats": self.cache_storage.get_stats() if self.cache_storage else {}
                },
                "retrievers": {
                    "available_sources": self.hybrid_retriever.get_available_sources(),
                    "performance_stats": self.hybrid_retriever.get_performance_stats()
                },
                "generators": {
                    "available_models": self.generation_service.get_available_models()
                },
                "chunking": {
                    "available_strategies": self.chunking_service.get_available_strategies()
                },
                "document_processors": {
                    "supported_extensions": self.document_processor_factory.get_supported_extensions()
                }
            }

            return status

        except Exception as e:
            return {"error": str(e)}

    def clear_index(self) -> bool:
        """Clear all indexed data."""
        try:
            success = True

            # Clear vector storage
            if self.vector_storage:
                success &= self.vector_storage.clear_vectors()

            # Clear graph storage
            if self.graph_storage:
                success &= self.graph_storage.clear_graph()

            # Clear cache storage
            if self.cache_storage:
                success &= self.cache_storage.clear()

            return success

        except Exception as e:
            print(f"Error clearing index: {e}")
            return False

    def _store_chunks_in_vector_storage(self, chunking_result):
        """Store chunks in vector storage."""
        try:
            # Get passage-level chunks for vector storage
            passage_chunks = chunking_result.chunks.get('passage', [])

            if not passage_chunks:
                print("No passage chunks to store in vector storage")
                return

            # Prepare vectors and metadata
            vectors = []
            metadata = []

            # This would require generating embeddings for each chunk
            # For now, this is a placeholder
            print(f"Would store {len(passage_chunks)} chunks in vector storage")

        except Exception as e:
            print(f"Error storing chunks in vector storage: {e}")

    def _store_entities_in_graph_storage(self, document_path: str, chunking_result) -> tuple[int, int]:
        """Store entities in graph storage."""
        try:
            entities_count = 0
            relationships_count = 0

            # Process all chunks for entities
            for level, chunks in chunking_result.chunks.items():
                for chunk in chunks:
                    # Add entities to graph
                    for entity in chunk.metadata.entities:
                        self.graph_storage.add_node(
                            node_id=entity.get('text', ''),
                            labels=[entity.get('label', 'Entity')],
                            properties={'description': entity.get('description', '')}
                        )
                        entities_count += 1

                    # Add relationships to graph
                    for relationship in chunk.metadata.relationships:
                        self.graph_storage.add_relationship(
                            from_node=relationship.get('source', ''),
                            to_node=relationship.get('target', ''),
                            relationship_type=relationship.get('type', 'RELATED_TO'),
                            properties={}
                        )
                        relationships_count += 1

            return entities_count, relationships_count

        except Exception as e:
            print(f"Error storing entities in graph storage: {e}")
            return 0, 0
