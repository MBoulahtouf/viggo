import io
import os
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pypdf import PdfReader
from viggo.core.config import settings
from viggo.models.schemas import DocumentInfoResponse, DocumentUploadResponse
from viggo.core.services.rag_service import RAGService
from viggo.core.services.graph_service import GraphService
from viggo.dependencies import get_rag_service, get_graph_service
from viggo.core.utils.file_ops import clear_indexes_and_graph

router = APIRouter()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
    graph_service: GraphService = Depends(get_graph_service)
):
    """Upload a PDF, process it for RAG, and load entities/relationships into the graph."""
    clear_indexes_and_graph(rag_service, graph_service)
    file_location = settings.data_dir + "/" + file.filename
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())
    num_chunks, _, all_chunks_with_metadata = rag_service.process_pdf(file_location)
    graph_service.extract_and_load_graph(file.filename, all_chunks_with_metadata)
    return DocumentUploadResponse(filename=file.filename, num_chunks_indexed=num_chunks, message="Document processed and indexed for RAG. Graph processing acknowledged.")

# @router.get("/info", response_model=DocumentInfoResponse)
# async def get_document_info():
#     """Get information about the currently loaded document (not implemented)."""
#     raise HTTPException(status_code=501, detail="Not implemented yet. Document info is not persisted across sessions.")