"""
User-Aware RAG Service for Viggo

This service integrates user progress tracking with the entity-chunk linking architecture,
providing personalized, spoiler-protected responses.
"""

from typing import Optional, Dict, List, Any
import time
from pathlib import Path

from viggo.core.services.rag_service import RAGService
from viggo.core.services.graph_service import GraphService
from viggo.core.services.enhanced_rag_integration import EnhancedRAGIntegration
from viggo.core.services.user_progress_service import UserProgressService
from viggo.core.models.user_progress import UserProgress, DocumentMetadata


class UserAwareRAGService:
    """
    RAG service that is aware of user progress and provides spoiler-protected responses.
    """
    
    def __init__(self, rag_service: RAGService, graph_service: GraphService, 
                 user_progress_service: UserProgressService):
        self.rag_service = rag_service
        self.graph_service = graph_service
        self.user_progress_service = user_progress_service
        self.enhanced_rag = EnhancedRAGIntegration(rag_service, graph_service)
        
        # Track processed documents
        self.processed_documents: Dict[str, bool] = {}
    
    def process_document_for_user(self, file_path: str, user_id: str, 
                                document_name: Optional[str] = None,
                                current_page: int = 0, finished_book: bool = False) -> Dict[str, Any]:
        """
        Process a document and set up user progress tracking.
        
        Args:
            file_path: Path to the document file
            user_id: User identifier
            document_name: Optional custom document name
            current_page: Current page the user is on
            finished_book: Whether the user has finished the book
            
        Returns:
            Dictionary with processing results and user progress
        """
        import time
        start_time = time.time()
        
        # Generate document ID from file path
        document_id = self._generate_document_id(file_path)
        
        # Process document with enhanced RAG (entity-chunk linking)
        print(f"🏗️ Processing document for user {user_id}: {file_path}")
        processing_result = self.enhanced_rag.process_document_with_entity_linking(file_path)
        
        # Get total pages from processing result
        total_pages = self._extract_total_pages(processing_result)
        
        # Create document metadata
        document_name = document_name or Path(file_path).stem
        metadata = self.user_progress_service.create_document_metadata(
            document_id=document_id,
            document_name=document_name,
            file_path=file_path,
            file_type=Path(file_path).suffix[1:].lower(),
            total_pages=total_pages,
            total_chunks=processing_result.get('entity_chunk_links', 0),
            processing_time=processing_result.get('processing_time', 0)
        )
        
        # Create user progress
        user_progress = self.user_progress_service.create_user_progress(
            user_id=user_id,
            document_id=document_id,
            document_name=document_name,
            total_pages=total_pages,
            current_page=current_page,
            finished_book=finished_book
        )
        
        # Mark document as processed
        self.processed_documents[document_id] = True
        
        processing_time = time.time() - start_time
        
        return {
            "document_id": document_id,
            "document_name": document_name,
            "user_id": user_id,
            "processing_result": processing_result,
            "user_progress": user_progress,
            "document_metadata": metadata,
            "total_processing_time": processing_time,
            "spoiler_protected": user_progress.is_spoiler_protected(),
            "spoiler_limit": user_progress.get_spoiler_limit()
        }
    
    def query_with_user_context(self, query: str, user_id: str, document_id: str) -> Dict[str, Any]:
        """
        Perform a query with user context and spoiler protection.
        
        Args:
            query: User query
            user_id: User identifier
            document_id: Document identifier
            
        Returns:
            Dictionary with query results and user context
        """
        import time
        start_time = time.time()
        
        # Get user progress
        user_progress = self.user_progress_service.get_user_progress(user_id, document_id)
        if not user_progress:
            return {
                "error": f"No progress found for user {user_id} and document {document_id}",
                "query": query,
                "user_id": user_id,
                "document_id": document_id
            }
        
        # Get spoiler limit
        spoiler_limit = user_progress.get_spoiler_limit()
        
        # Perform enhanced RAG query with spoiler protection
        rag_result = self.enhanced_rag.query_with_entity_context(query, spoiler_limit)
        
        query_time = time.time() - start_time
        
        return {
            "query": query,
            "user_id": user_id,
            "document_id": document_id,
            "user_progress": {
                "current_page": user_progress.current_page,
                "total_pages": user_progress.total_pages,
                "progress_percentage": user_progress.get_progress_percentage(),
                "reading_status": user_progress.get_reading_status_text(),
                "finished_book": user_progress.finished_book,
                "spoiler_protected": user_progress.is_spoiler_protected()
            },
            "rag_result": {
                "answer": rag_result.answer,
                "source_pages": rag_result.source_pages,
                "search_method": rag_result.search_method,
                "entities_found": rag_result.entities_found,
                "spoiler_protected": rag_result.spoiler_protected
            },
            "query_time": query_time,
            "spoiler_limit": spoiler_limit
        }
    
    def update_user_progress(self, user_id: str, document_id: str, 
                           page: int, finished_book: bool = False) -> Optional[UserProgress]:
        """
        Update user's reading progress.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            page: New page number
            finished_book: Whether the user has finished the book
            
        Returns:
            Updated UserProgress object
        """
        return self.user_progress_service.update_user_progress(user_id, document_id, page, finished_book)
    
    def get_user_progress(self, user_id: str, document_id: str) -> Optional[UserProgress]:
        """
        Get user's current progress.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            
        Returns:
            UserProgress object
        """
        return self.user_progress_service.get_user_progress(user_id, document_id)
    
    def get_user_documents(self, user_id: str) -> List[UserProgress]:
        """
        Get all documents for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of UserProgress objects
        """
        return self.user_progress_service.get_user_documents(user_id)
    
    def find_entity_passages_for_user(self, entity_name: str, user_id: str, document_id: str,
                                    context_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Find entity passages with user context and spoiler protection.
        
        Args:
            entity_name: Name of the entity
            user_id: User identifier
            document_id: Document identifier
            context_type: Optional context type filter
            
        Returns:
            Dictionary with entity passages and user context
        """
        # Get user progress
        user_progress = self.user_progress_service.get_user_progress(user_id, document_id)
        if not user_progress:
            return {
                "error": f"No progress found for user {user_id} and document {document_id}",
                "entity_name": entity_name
            }
        
        # Get spoiler limit
        spoiler_limit = user_progress.get_spoiler_limit()
        
        # Find entity passages
        from viggo.core.services.entity_chunk_linker import ContextType
        context_type_enum = None
        if context_type:
            try:
                context_type_enum = ContextType(context_type)
            except ValueError:
                pass
        
        passages = self.enhanced_rag.find_entity_passages(entity_name, context_type_enum, spoiler_limit)
        
        return {
            "entity_name": entity_name,
            "user_id": user_id,
            "document_id": document_id,
            "user_progress": {
                "current_page": user_progress.current_page,
                "total_pages": user_progress.total_pages,
                "progress_percentage": user_progress.get_progress_percentage(),
                "spoiler_protected": user_progress.is_spoiler_protected()
            },
            "passages": [
                {
                    "chunk_id": passage.chunk_id,
                    "page_number": passage.page_number,
                    "context_type": passage.context_type.value,
                    "context_score": passage.context_score,
                    "surrounding_text": passage.surrounding_text,
                    "lore_significance": passage.lore_significance
                }
                for passage in passages
            ],
            "total_passages": len(passages),
            "spoiler_limit": spoiler_limit
        }
    
    def get_entity_context_for_user(self, entity_name: str, user_id: str, document_id: str) -> Dict[str, Any]:
        """
        Get entity context analysis with user progress.
        
        Args:
            entity_name: Name of the entity
            user_id: User identifier
            document_id: Document identifier
            
        Returns:
            Dictionary with entity context and user progress
        """
        # Get user progress
        user_progress = self.user_progress_service.get_user_progress(user_id, document_id)
        if not user_progress:
            return {
                "error": f"No progress found for user {user_id} and document {document_id}",
                "entity_name": entity_name
            }
        
        # Get spoiler limit
        spoiler_limit = user_progress.get_spoiler_limit()
        
        # Get entity context analysis
        context_analysis = self.enhanced_rag.get_entity_context_analysis(entity_name, spoiler_limit)
        
        return {
            "entity_name": entity_name,
            "user_id": user_id,
            "document_id": document_id,
            "user_progress": {
                "current_page": user_progress.current_page,
                "total_pages": user_progress.total_pages,
                "progress_percentage": user_progress.get_progress_percentage(),
                "spoiler_protected": user_progress.is_spoiler_protected()
            },
            "context_analysis": context_analysis,
            "spoiler_limit": spoiler_limit
        }
    
    def get_user_reading_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of user's reading progress across all documents.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with reading summary
        """
        return self.user_progress_service.get_progress_summary(user_id)
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate a unique document ID from file path."""
        import hashlib
        return hashlib.md5(file_path.encode()).hexdigest()[:16]
    
    def _extract_total_pages(self, processing_result: Dict[str, Any]) -> int:
        """Extract total pages from processing result."""
        # This would typically come from the document processor
        # For now, we'll estimate based on chunks or use a default
        chunks = processing_result.get('chunking_result', {}).get('chunks_with_metadata', [])
        if chunks:
            # Estimate pages based on chunk metadata
            max_page = 0
            for chunk in chunks:
                page = chunk.get('page', 0)
                max_page = max(max_page, page)
            return max_page if max_page > 0 else 16  # Default for Lovecraft story
        
        return 16  # Default fallback
