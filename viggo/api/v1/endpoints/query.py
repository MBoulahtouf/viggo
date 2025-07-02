# viggo/api/v1/endpoints/query.py
from fastapi import APIRouter, HTTPException, Depends
from viggo.models.schemas import QueryRequest, QueryResponse
from viggo.core.services.rag_service import RAGService
from viggo.dependencies import get_rag_service
import logging

router = APIRouter()

@router.post("/", response_model=QueryResponse)
async def query_document(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Query the indexed document using RAG and return an answer with supporting context."""
    logging.info(f"Received query: {request.question} (page: {request.page_number})")
    if not hasattr(request, 'question') or not hasattr(request, 'page_number'):
        raise HTTPException(status_code=400, detail="Malformed request: missing question or page_number.")
    if rag_service.index is None:
        raise HTTPException(status_code=400, detail="No document has been indexed yet. Please upload a PDF first.")
    result = rag_service.perform_rag_query(
        question=request.question,
        page_number=request.page_number,
        vector_index=rag_service.index,
        all_chunks_with_metadata=rag_service.all_chunks_with_metadata
    )
    if result is None:
        raise HTTPException(status_code=404, detail="No content available at or before the specified page.")
    return QueryResponse(**result)