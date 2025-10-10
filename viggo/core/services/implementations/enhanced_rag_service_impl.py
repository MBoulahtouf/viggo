"""
Enhanced RAG service implementation with multi-agent framework integration.
Extends the existing RAG service with advanced query processing and relationship extraction.
"""

import asyncio
import time
from typing import Any

from viggo.core.services.implementations.azure_graph_rag_impl import (
    AzureGraphRAGService,
)
from viggo.core.services.implementations.multi_agent_impl import MultiAgentOrchestrator
from viggo.core.services.implementations.rag_service_impl import ConcreteRAGService
from viggo.core.services.interfaces.rag import (
    IndexingResult,
    RAGConfig,
    RAGResult,
    RAGService,
)
from viggo.core.services.interfaces.retrieval import QueryContext


class EnhancedRAGService(RAGService):
    """
    Enhanced RAG service that integrates multi-agent framework and GraphRAG capabilities.
    Extends the existing ConcreteRAGService with advanced features.
    """

    def __init__(self, config: RAGConfig, graph_rag_service: AzureGraphRAGService | None = None):
        # Initialize base RAG service
        self.base_rag_service = ConcreteRAGService(config)

        # Initialize multi-agent orchestrator
        self.multi_agent_orchestrator = MultiAgentOrchestrator()

        # Initialize GraphRAG service if provided
        self.graph_rag_service = graph_rag_service

        # Store configuration
        self.config = config

        # Track processed documents for GraphRAG
        self.processed_documents = {}

        print("🚀 Enhanced RAG Service initialized with multi-agent framework")

    def index_document(self, document_path: str) -> IndexingResult:
        """Index a document with enhanced processing including GraphRAG."""
        print(f"📚 Indexing document with enhanced processing: {document_path}")

        # First, use the base RAG service for standard indexing
        base_result = self.base_rag_service.index_document(document_path)

        if not base_result.success:
            return base_result

        # If GraphRAG service is available, perform enhanced processing
        if self.graph_rag_service:
            try:
                # Get document content for GraphRAG processing
                document_content = self._get_document_content(document_path)

                if document_content:
                    # Process with GraphRAG (async)
                    asyncio.run(self._process_document_with_graph_rag(document_path, document_content))

                    # Update result with GraphRAG information
                    base_result.metadata = base_result.metadata or {}
                    base_result.metadata['graph_rag_processed'] = True
                    base_result.metadata['entities_extracted'] = len(self.processed_documents.get(document_path, {}).get('entities', []))
                    base_result.metadata['relationships_found'] = len(self.processed_documents.get(document_path, {}).get('relationships', []))
                    base_result.metadata['communities_identified'] = len(self.processed_documents.get(document_path, {}).get('communities', []))

            except Exception as e:
                print(f"⚠️ GraphRAG processing failed for {document_path}: {e}")
                base_result.metadata = base_result.metadata or {}
                base_result.metadata['graph_rag_error'] = str(e)

        return base_result

    async def _process_document_with_graph_rag(self, document_path: str, document_content: list[str]):
        """Process document with GraphRAG pipeline."""
        try:
            print(f"🔍 Processing {document_path} with GraphRAG pipeline...")

            # Stage 1: Extract entities and relationships
            entities, relationships = await self.graph_rag_service.extract_nodes_and_relationships(document_content)

            # Stage 2: Summarize and deduplicate
            summarized_entities, summarized_relationships = await self.graph_rag_service.summarize_nodes_and_relationships(
                entities, relationships
            )

            # Stage 3: Identify communities
            communities = await self.graph_rag_service.identify_entity_communities(
                summarized_entities, summarized_relationships
            )

            # Stage 4: Generate community summaries
            communities_with_summaries = await self.graph_rag_service.generate_community_summaries(
                communities, document_content
            )

            # Stage 5: Store in Neo4j
            await self.graph_rag_service.store_in_neo4j(
                summarized_entities, summarized_relationships, communities_with_summaries
            )

            # Store results for later use
            self.processed_documents[document_path] = {
                'entities': summarized_entities,
                'relationships': summarized_relationships,
                'communities': communities_with_summaries,
                'processed_at': time.time()
            }

            print(f"✅ GraphRAG processing completed for {document_path}")

        except Exception as e:
            print(f"❌ GraphRAG processing failed for {document_path}: {e}")
            raise

    def _get_document_content(self, document_path: str) -> list[str] | None:
        """Get document content as list of text chunks."""
        try:
            # Use the document processor to get content
            processor = self.config.document_processor_factory.get_processor(document_path)
            if processor:
                pages = processor.process_document(document_path)
                return [page.content for page in pages if page.content.strip()]
        except Exception as e:
            print(f"Error getting document content: {e}")

        return None

    def query(self, query: str, context: QueryContext | None = None) -> RAGResult:
        """Enhanced query processing using multi-agent framework."""
        print(f"🤖 Processing query with multi-agent framework: {query}")
        start_time = time.time()

        try:
            # Step 1: Use multi-agent orchestrator to analyze and process query
            agent_results = self.multi_agent_orchestrator.process_query(query, {
                'content': self._get_all_document_content(),
                'semantic_results': [],  # Will be populated by base RAG service
                'graph_results': []      # Will be populated by GraphRAG service
            })

            # Step 2: Get base RAG results
            base_result = self.base_rag_service.query(query, context)

            # Step 3: Enhance with GraphRAG if available
            enhanced_context = {}
            if self.graph_rag_service and self.processed_documents:
                try:
                    # Get all entities and communities from processed documents
                    all_entities = []
                    all_communities = []

                    for doc_data in self.processed_documents.values():
                        all_entities.extend(doc_data.get('entities', []))
                        all_communities.extend(doc_data.get('communities', []))

                    # Query GraphRAG
                    graph_rag_result = asyncio.run(
                        self.graph_rag_service.query_with_graph_rag(query, all_entities, all_communities)
                    )

                    enhanced_context = {
                        'semantic_results': base_result.metadata.get('retrieval_results', []),
                        'graph_results': graph_rag_result.get('community_summaries', []),
                        'entities': graph_rag_result.get('relevant_entities', []),
                        'communities': graph_rag_result.get('relevant_communities', [])
                    }

                except Exception as e:
                    print(f"⚠️ GraphRAG query failed: {e}")
                    enhanced_context = {
                        'semantic_results': base_result.metadata.get('retrieval_results', []),
                        'graph_results': []
                    }
            else:
                enhanced_context = {
                    'semantic_results': base_result.metadata.get('retrieval_results', []),
                    'graph_results': []
                }

            # Step 4: Use multi-agent orchestrator to generate enhanced response
            enhanced_agent_results = self.multi_agent_orchestrator.process_query(query, enhanced_context)

            # Step 5: Combine results
            if 'response' in enhanced_agent_results and enhanced_agent_results['response']:
                # Use multi-agent generated response
                final_answer = enhanced_agent_results['response'].get('response', base_result.answer)
            else:
                # Fallback to base RAG result
                final_answer = base_result.answer

            # Enhance metadata
            enhanced_metadata = base_result.metadata.copy() if base_result.metadata else {}
            enhanced_metadata.update({
                'multi_agent_processed': True,
                'agent_analysis': agent_results.get('analysis', {}),
                'agent_aggregation': enhanced_agent_results.get('aggregation', {}),
                'graph_rag_used': self.graph_rag_service is not None,
                'processing_time_enhanced': time.time() - start_time
            })

            # Calculate enhanced confidence score
            base_confidence = base_result.confidence_score
            agent_confidence = enhanced_agent_results.get('aggregation', {}).get('hybrid_score', 0.5)
            enhanced_confidence = (base_confidence * 0.6) + (agent_confidence * 0.4)

            return RAGResult(
                query=query,
                answer=final_answer,
                source_pages=base_result.source_pages,
                confidence_score=enhanced_confidence,
                processing_time=time.time() - start_time,
                metadata=enhanced_metadata,
                citations=base_result.citations
            )

        except Exception as e:
            print(f"❌ Enhanced query processing failed: {e}")
            # Fallback to base RAG service
            return self.base_rag_service.query(query, context)

    def _get_all_document_content(self) -> str:
        """Get all indexed document content for context."""
        try:
            # This would aggregate content from all indexed documents
            # For now, return empty string as placeholder
            return ""
        except Exception as e:
            print(f"Error getting document content: {e}")
            return ""

    def update_document(self, document_path: str) -> IndexingResult:
        """Update an existing document index."""
        # Remove from processed documents if exists
        if document_path in self.processed_documents:
            del self.processed_documents[document_path]

        # Use base service for update
        return self.base_rag_service.update_document(document_path)

    def delete_document(self, document_path: str) -> bool:
        """Delete a document from the index."""
        # Remove from processed documents if exists
        if document_path in self.processed_documents:
            del self.processed_documents[document_path]

        # Use base service for deletion
        return self.base_rag_service.delete_document(document_path)

    def get_system_status(self) -> dict[str, Any]:
        """Get enhanced system status including multi-agent and GraphRAG status."""
        base_status = self.base_rag_service.get_system_status()

        # Add multi-agent status
        base_status['multi_agent'] = {
            'enabled': True,
            'agents': self.multi_agent_orchestrator.get_agent_status()
        }

        # Add GraphRAG status
        base_status['graph_rag'] = {
            'enabled': self.graph_rag_service is not None,
            'processed_documents': len(self.processed_documents),
            'total_entities': sum(len(doc.get('entities', [])) for doc in self.processed_documents.values()),
            'total_relationships': sum(len(doc.get('relationships', [])) for doc in self.processed_documents.values()),
            'total_communities': sum(len(doc.get('communities', [])) for doc in self.processed_documents.values())
        }

        return base_status

    def clear_index(self) -> bool:
        """Clear all indexed data including GraphRAG data."""
        # Clear processed documents
        self.processed_documents.clear()

        # Use base service for clearing
        return self.base_rag_service.clear_index()

    def get_agent_status(self) -> dict[str, Any]:
        """Get status of multi-agent system."""
        return self.multi_agent_orchestrator.get_agent_status()

    def get_graph_rag_status(self) -> dict[str, Any]:
        """Get status of GraphRAG system."""
        if not self.graph_rag_service:
            return {'enabled': False}

        return {
            'enabled': True,
            'processed_documents': len(self.processed_documents),
            'total_entities': sum(len(doc.get('entities', [])) for doc in self.processed_documents.values()),
            'total_relationships': sum(len(doc.get('relationships', [])) for doc in self.processed_documents.values()),
            'total_communities': sum(len(doc.get('communities', [])) for doc in self.processed_documents.values())
        }
