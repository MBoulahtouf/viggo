# viggo/api/v1/endpoints/query.py
from fastapi import APIRouter, HTTPException, Depends, Query as FastAPIQuery
from viggo.models.schemas import QueryRequest, QueryResponse
from viggo.core.services.rag_service import RAGService
from viggo.dependencies import get_rag_service
from viggo.core.services.hybrid_retriever import HybridRetriever
from viggo.core.services.graph_service import GraphService
from viggo.core.config import settings
import logging
import time
from typing import Optional

router = APIRouter()

@router.post("/", response_model=QueryResponse)
async def query_document(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Query the indexed document using hybrid RAG and return an answer with supporting context."""
    logging.info(f"Received query: {request.question} (page: {request.page_number})")
    
    # Validate request
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    if request.page_number < 0:
        raise HTTPException(status_code=400, detail="Page number must be non-negative.")
    
    # Check if we have indexed documents
    if rag_service.index is None or not rag_service.all_chunks_with_metadata:
        raise HTTPException(status_code=400, detail="No document has been indexed yet. Please upload a PDF first.")
    
    try:
        # Initialize hybrid retriever with all services
        hybrid_retriever = HybridRetriever(
            vector_index=rag_service.index,
            all_chunks_with_metadata=rag_service.all_chunks_with_metadata
        )
        
        # Initialize graph service if available
        try:
            graph_service = GraphService(
                uri=settings.neo4j_uri,
                user=settings.neo4j_user,
                password=settings.neo4j_password
            )
            hybrid_retriever.graph_service = graph_service
        except Exception as e:
            logging.warning(f"Neo4j service not available: {e}")
            hybrid_retriever.graph_service = None
        
        # Perform hybrid retrieval
        start_time = time.time()
        retrieval_result = await hybrid_retriever.retrieve(
            query=request.question,
            top_k=5,
            page_filter=request.page_number
        )
        
        # Generate answer using LLM
        if retrieval_result and retrieval_result.get("results"):
            answer = rag_service.generate_answer_with_context(
                question=request.question,
                context_results=retrieval_result["results"]
            )
        else:
            answer = "I couldn't find relevant information to answer your question. Please try rephrasing or check if the document has been properly indexed."
        
        # Extract source pages
        source_pages = list(set([result.get("page", 0) for result in retrieval_result.get("results", [])]))
        source_pages.sort()
        
        # Log performance
        processing_time = time.time() - start_time
        logging.info(f"Query processed in {processing_time:.2f}s with {len(source_pages)} source pages")
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            source_pages=source_pages
        )
        
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for the query service."""
    return {
        "status": "healthy",
        "service": "query",
        "timestamp": time.time()
    }

@router.get("/stats")
async def get_query_stats(
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get query service statistics."""
    try:
        stats = {
            "indexed_documents": len(rag_service.all_chunks_with_metadata) if rag_service.all_chunks_with_metadata else 0,
            "vector_index_available": rag_service.index is not None,
            "services": {
                "neo4j": "unknown",
                "elasticsearch": "unknown",
                "redis": "unknown"
            }
        }
        
        # Check Neo4j
        try:
            graph_service = GraphService(
                uri=settings.neo4j_uri,
                user=settings.neo4j_user,
                password=settings.neo4j_password
            )
            stats["services"]["neo4j"] = "available"
            graph_service.close()
        except:
            stats["services"]["neo4j"] = "unavailable"
        
        # Check Redis
        try:
            from viggo.core.services.redis_service import redis_service
            if redis_service.is_connected():
                stats["services"]["redis"] = "available"
            else:
                stats["services"]["redis"] = "unavailable"
        except:
            stats["services"]["redis"] = "unavailable"
        
        # Check Elasticsearch
        try:
            from viggo.core.services.hybrid_search_service import HybridSearchService
            search_service = HybridSearchService()
            if search_service.search_client:
                stats["services"]["elasticsearch"] = "available"
            else:
                stats["services"]["elasticsearch"] = "unavailable"
        except:
            stats["services"]["elasticsearch"] = "unavailable"
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")