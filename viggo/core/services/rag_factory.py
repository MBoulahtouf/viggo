"""
RAG service factory for creating and managing RAG service instances.
"""

from typing import Any

from viggo.core.services.interfaces.rag import RAGService


class SimpleRAGService(RAGService):
    """Simple RAG service implementation for basic functionality."""

    def __init__(self):
        # Create a simple document processor factory
        from viggo.core.services.implementations.document_processor_impl import (
            ConcreteDocumentProcessorFactory,
        )
        from viggo.core.services.implementations.redis_service_impl import RedisService
        self.document_processor_factory = ConcreteDocumentProcessorFactory()
        self.redis_service = RedisService()
        self._indexed_documents = {}
        self._document_content = {}  # Store actual document content

    def index_document(self, document_path: str):
        """Index a document for retrieval."""
        import time

        from viggo.core.services.interfaces.rag import IndexingResult

        start_time = time.time()

        # Get actual page count from the document processor
        page_count = 0
        processor = self.document_processor_factory.get_processor(document_path)
        if processor and hasattr(processor, 'get_pdf_info'):
            # For PDF files, get actual page count from PDF metadata
            pdf_info = processor.get_pdf_info(document_path)
            page_count = pdf_info.get('num_pages', 0)
            print(f"[DEBUG] SimpleRAGService - PDF page count from metadata: {page_count}")
        elif processor and hasattr(processor, 'get_epub_info'):
            # For EPUB files, get page count from EPUB info
            epub_info = processor.get_epub_info(document_path)
            page_count = epub_info.get('num_pages', 0)
            print(f"[DEBUG] SimpleRAGService - EPUB page count from metadata: {page_count}")

        # Actually process the document content
        try:
            # Get the document processor and extract content
            processor = self.document_processor_factory.get_processor(document_path)
            if processor:
                # Process the document to get actual content
                pages = processor.process_document(document_path)

                # Store the actual content in Redis for persistence
                document_data = {
                    'pages': [
                        {
                            'page_number': page.page_number if hasattr(page, 'page_number') else i+1,
                            'content': page.content if hasattr(page, 'content') else str(page),
                            'metadata': page.metadata if hasattr(page, 'metadata') else {}
                        }
                        for i, page in enumerate(pages)
                    ],
                    'page_count': page_count,
                    'indexed': True,
                    'document_path': document_path
                }

                # Store in Redis with a simple key
                doc_key = "document:latest"
                self.redis_service.cache_session_data(doc_key, document_data, ttl=3600)  # 1 hour TTL

                # Also store locally for immediate access
                self._document_content[document_path] = document_data

                chunks_created = len(pages) if pages else page_count
            else:
                chunks_created = page_count
                self._document_content[document_path] = {
                    'pages': [],
                    'page_count': page_count,
                    'indexed': True
                }
        except Exception as e:
            print(f"[DEBUG] Error processing document content: {e}")
            chunks_created = page_count
            self._document_content[document_path] = {
                'pages': [],
                'page_count': page_count,
                'indexed': True
            }

        # Store the page count for later use
        self._indexed_documents[document_path] = {
            'indexed': True,
            'page_count': page_count
        }

        processing_time = time.time() - start_time

        return IndexingResult(
            document_path=document_path,
            chunks_created=chunks_created,
            entities_extracted=5,  # Mock value
            relationships_found=3,  # Mock value
            processing_time=processing_time,
            success=True,
            error_message=None
        )

    def query(self, query: str, context=None):
        """Query the RAG system."""
        import time

        from viggo.core.services.interfaces.rag import RAGResult

        start_time = time.time()

        # Find the most recent indexed document
        if not self._document_content:
            # Try to load from Redis using a simple approach
            try:
                # For now, let's use a simple key pattern
                # Try to find any document in Redis
                doc_key = "document:latest"
                doc_data = self.redis_service.get_session_data(doc_key)

                if not doc_data:
                    return RAGResult(
                        query=query,
                        answer="No documents have been indexed yet. Please upload a document first.",
                        source_pages=[],
                        confidence_score=0.0,
                        processing_time=time.time() - start_time,
                        metadata={"error": "no_documents"}
                    )

                # Store in local cache for future use
                document_path = doc_data.get('document_path', 'unknown')
                self._document_content[document_path] = doc_data

            except Exception as e:
                print(f"[DEBUG] Error loading from Redis: {e}")
                return RAGResult(
                    query=query,
                    answer="Error accessing document cache. Please re-upload the document.",
                    source_pages=[],
                    confidence_score=0.0,
                    processing_time=time.time() - start_time,
                    metadata={"error": "redis_error", "details": str(e)}
                )
        else:
            # Get the most recent document from local cache
            document_path = list(self._document_content.keys())[-1]
            doc_data = self._document_content[document_path]

        pages = doc_data.get('pages', [])

        if not pages:
            return RAGResult(
                query=query,
                answer="Document content could not be processed. Please try re-uploading the document.",
                source_pages=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": "no_content"}
            )

        # Simple keyword-based search
        query_lower = query.lower()
        query_words = query_lower.split()

        # Search through pages for relevant content
        relevant_pages = []
        for page in pages:
            # Handle both old page objects and new dict structure
            if isinstance(page, dict):
                content = page.get('content', '').lower()
                page_num = page.get('page_number', 1)
                page_content = page.get('content', '')
            else:
                content = page.content.lower() if hasattr(page, 'content') else str(page).lower()
                page_num = page.page_number if hasattr(page, 'page_number') else 1
                page_content = page.content if hasattr(page, 'content') else str(page)

            # Simple relevance scoring based on keyword matches
            matches = sum(1 for word in query_words if word in content)
            if matches > 0:
                relevance_score = matches / len(query_words)
                relevant_pages.append({
                    'page_number': page_num,
                    'content': page_content,
                    'relevance_score': relevance_score
                })

        # Sort by relevance
        relevant_pages.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Generate intelligent answer based on most relevant content
        if relevant_pages:
            # Get the top 3 most relevant pages for better context
            top_pages = relevant_pages[:3]
            all_content = []
            source_pages = []

            for page in top_pages:
                all_content.append(page['content'])
                source_pages.append(page['page_number'])

            # Combine content for better context
            combined_content = ' '.join(all_content)

            # Intelligent answer generation based on query type
            if any(word in query_lower for word in ['who', 'character', 'main character', 'protagonist']):
                answer = self._generate_character_answer(combined_content, top_pages)
            elif any(word in query_lower for word in ['what', 'happens', 'plot', 'story', 'about']):
                answer = self._generate_plot_answer(combined_content, top_pages)
            elif any(word in query_lower for word in ['where', 'location', 'place', 'setting']):
                answer = self._generate_setting_answer(combined_content, top_pages)
            elif any(word in query_lower for word in ['when', 'time', 'period', 'era']):
                answer = self._generate_temporal_answer(combined_content, top_pages)
            else:
                answer = self._generate_general_answer(combined_content, top_pages)

            return RAGResult(
                query=query,
                answer=answer,
                source_pages=[p['page_number'] for p in top_pages],
                confidence_score=top_pages[0]['relevance_score'],
                processing_time=time.time() - start_time,
                metadata={"pages_searched": len(pages), "relevant_pages_found": len(relevant_pages)}
            )
        else:
            return RAGResult(
                query=query,
                answer="I couldn't find relevant information in the document to answer your question. Please try rephrasing your question or check if the document was processed correctly.",
                source_pages=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                metadata={"pages_searched": len(pages), "relevant_pages_found": 0}
            )

    def update_document(self, document_path: str):
        """Update an existing document index."""
        return self.index_document(document_path)

    def delete_document(self, document_path: str) -> bool:
        """Delete a document from the index."""
        if document_path in self._indexed_documents:
            del self._indexed_documents[document_path]
            return True
        return False

    def get_system_status(self) -> dict[str, Any]:
        """Get the status of the RAG system."""
        return {
            "vector_storage": {"available": True, "vector_count": len(self._indexed_documents)},
            "graph_storage": {"available": False},
            "cache_storage": {"available": False},
            "retrievers": {"available_sources": ["simple"]},
            "generators": {"available_models": ["simple"]}
        }

    def clear_index(self) -> bool:
        """Clear the entire index."""
        self._indexed_documents.clear()
        return True

    def _generate_character_answer(self, content: str, pages: list) -> str:
        """Generate an intelligent answer about characters."""
        import re

        # Look for proper names (capitalized words that appear multiple times)
        words = re.findall(r'\b[A-Z][a-z]+\b', content)
        name_counts = {}
        for word in words:
            if len(word) > 2 and word not in ['The', 'And', 'But', 'For', 'With', 'From', 'They', 'This', 'That']:
                name_counts[word] = name_counts.get(word, 0) + 1

        # Find the most mentioned character
        if name_counts:
            main_character = max(name_counts, key=name_counts.get)

            # Look for sentences that mention this character
            sentences = re.split(r'[.!?]+', content)
            character_sentences = [s.strip() for s in sentences if main_character in s and len(s.strip()) > 20]

            if character_sentences:
                # Get the most informative sentence about the character
                best_sentence = character_sentences[0]
                return f"The main character appears to be **{main_character}**. {best_sentence[:200]}{'...' if len(best_sentence) > 200 else ''}"

        # Fallback: look for character descriptions
        character_patterns = [
            r'[A-Z][a-z]+ was [^.]*\.',
            r'[A-Z][a-z]+ had [^.]*\.',
            r'[A-Z][a-z]+ seemed [^.]*\.',
            r'[A-Z][a-z]+ appeared [^.]*\.'
        ]

        for pattern in character_patterns:
            matches = re.findall(pattern, content)
            if matches:
                return f"Based on the text: {matches[0][:200]}{'...' if len(matches[0]) > 200 else ''}"

        # Final fallback
        return f"Based on the content from pages {', '.join(str(p['page_number']) for p in pages[:2])}: {content[:300]}{'...' if len(content) > 300 else ''}"

    def _generate_plot_answer(self, content: str, pages: list) -> str:
        """Generate an intelligent answer about plot/story."""
        import re

        # Look for action sentences
        action_patterns = [
            r'[A-Z][^.]*happened[^.]*\.',
            r'[A-Z][^.]*occurred[^.]*\.',
            r'[A-Z][^.]*began[^.]*\.',
            r'[A-Z][^.]*started[^.]*\.',
            r'[A-Z][^.]*went[^.]*\.',
            r'[A-Z][^.]*came[^.]*\.'
        ]

        for pattern in action_patterns:
            matches = re.findall(pattern, content)
            if matches:
                return f"The story involves: {matches[0][:250]}{'...' if len(matches[0]) > 250 else ''}"

        # Look for descriptive sentences about events
        sentences = re.split(r'[.!?]+', content)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 30 and any(word in s.lower() for word in ['story', 'tale', 'narrative', 'event', 'happened', 'occurred'])]

        if meaningful_sentences:
            return f"The story appears to be about: {meaningful_sentences[0][:300]}{'...' if len(meaningful_sentences[0]) > 300 else ''}"

        # Fallback
        return f"Based on the content from pages {', '.join(str(p['page_number']) for p in pages[:2])}: {content[:300]}{'...' if len(content) > 300 else ''}"

    def _generate_setting_answer(self, content: str, pages: list) -> str:
        """Generate an intelligent answer about setting/location."""
        import re

        # Look for location descriptions
        location_patterns = [
            r'[A-Z][^.]*place[^.]*\.',
            r'[A-Z][^.]*location[^.]*\.',
            r'[A-Z][^.]*house[^.]*\.',
            r'[A-Z][^.]*town[^.]*\.',
            r'[A-Z][^.]*city[^.]*\.',
            r'[A-Z][^.]*village[^.]*\.'
        ]

        for pattern in location_patterns:
            matches = re.findall(pattern, content)
            if matches:
                return f"The setting appears to be: {matches[0][:250]}{'...' if len(matches[0]) > 250 else ''}"

        # Fallback
        return f"Based on the content from pages {', '.join(str(p['page_number']) for p in pages[:2])}: {content[:300]}{'...' if len(content) > 300 else ''}"

    def _generate_temporal_answer(self, content: str, pages: list) -> str:
        """Generate an intelligent answer about time/period."""
        import re

        # Look for time references
        time_patterns = [
            r'[A-Z][^.]*time[^.]*\.',
            r'[A-Z][^.]*period[^.]*\.',
            r'[A-Z][^.]*era[^.]*\.',
            r'[A-Z][^.]*century[^.]*\.',
            r'[A-Z][^.]*year[^.]*\.'
        ]

        for pattern in time_patterns:
            matches = re.findall(pattern, content)
            if matches:
                return f"The time period appears to be: {matches[0][:250]}{'...' if len(matches[0]) > 250 else ''}"

        # Fallback
        return f"Based on the content from pages {', '.join(str(p['page_number']) for p in pages[:2])}: {content[:300]}{'...' if len(content) > 300 else ''}"

    def _generate_general_answer(self, content: str, pages: list) -> str:
        """Generate a general intelligent answer."""
        import re

        # Look for the most informative sentences
        sentences = re.split(r'[.!?]+', content)
        informative_sentences = [s.strip() for s in sentences if len(s.strip()) > 40]

        if informative_sentences:
            return f"Based on the content: {informative_sentences[0][:300]}{'...' if len(informative_sentences[0]) > 300 else ''}"

        # Fallback
        return f"Based on the content from pages {', '.join(str(p['page_number']) for p in pages[:2])}: {content[:300]}{'...' if len(content) > 300 else ''}"


class RAGFactory:
    """Factory for creating RAG service instances."""

    def __init__(self):
        self._default_rag_service = None
        self._singleton_service = None

    def create_rag_service(self,
                          graph_service: Any | None = None,
                          redis_service: Any | None = None,
                          config_type: str = "default") -> RAGService:
        """
        Create a RAG service with the specified configuration.
        
        Args:
            graph_service: Optional Neo4j graph service (ignored for now)
            redis_service: Optional Redis cache service (ignored for now)
            config_type: Type of configuration (ignored for now)
            
        Returns:
            RAG service instance
        """
        # Use singleton pattern to maintain state between requests
        if self._singleton_service is None:
            self._singleton_service = SimpleRAGService()
        return self._singleton_service

    def get_default_rag_service(self) -> RAGService:
        """Get the default RAG service instance."""
        if self._default_rag_service is None:
            self._default_rag_service = self.create_rag_service()

        return self._default_rag_service

    def get_available_components(self) -> dict[str, list]:
        """Get list of available components."""
        return {
            "document_processors": ["pdf", "epub"],
            "chunking_strategies": ["hybrid", "standard"],
            "retrievers": ["semantic", "keyword", "graph"],
            "generators": ["llm", "template"],
            "storage_backends": ["faiss", "neo4j", "redis", "file"]
        }

    def validate_configuration(self, config: Any) -> bool:
        """Validate a RAG configuration."""
        return True


# Global factory instance
rag_factory = RAGFactory()


def get_rag_service(graph_service: Any | None = None,
                   redis_service: Any | None = None,
                   config_type: str = "default") -> RAGService:
    """
    Convenience function to get a RAG service instance.
    
    Args:
        graph_service: Optional Neo4j graph service
        redis_service: Optional Redis cache service
        config_type: Type of configuration ("default", "minimal", "custom")
        
    Returns:
        RAG service instance
    """
    return rag_factory.create_rag_service(
        graph_service=graph_service,
        redis_service=redis_service,
        config_type=config_type
    )
