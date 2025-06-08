# strider/api/v1/endpoints/query.py
from fastapi import APIRouter, HTTPException
from strider.core.services import rag_service
from strider.core import state
from strider.models.schemas import QueryRequest, QueryResponse

router = APIRouter()

@router.post("/", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    if state.vector_index is None:
        raise HTTPException(status_code=400, detail="No document has been indexed yet.")

    result = rag_service.perform_rag_query(
        question=request.question,
        page_number=request.page_number,
        vector_index=state.vector_index,
        all_chunks_with_metadata=state.all_chunks_with_metadata
    )

    if result is None:
        raise HTTPException(status_code=404, detail="No content available at or before the specified page.")
        
    return result
