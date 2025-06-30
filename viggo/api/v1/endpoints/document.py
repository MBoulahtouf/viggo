import io
import os
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pypdf import PdfReader
from viggo.core.config import settings
from viggo.models.schemas import DocumentInfoResponse
from viggo.core.services.rag_service import RAGService
from viggo.core.services.graph_service import GraphService
from viggo.dependencies import get_rag_service, get_graph_service

router = APIRouter()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
    graph_service: GraphService = Depends(get_graph_service)
):
    # Save the uploaded file to the data directory
    file_location = os.path.join(settings.data_dir, file.filename)
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())

    # Use the injected RAGService instance to process the PDF
    # This method will be re-implemented in rag_service.py to handle PDF reading and chunking
    num_chunks, vector_index, all_chunks_with_metadata = rag_service.process_pdf(file_location)

    # Extract and load graph data
    graph_service.extract_and_load_graph(file.filename, all_chunks_with_metadata)

    return {
        "filename": file.filename,
        "num_chunks_indexed": num_chunks,
        "message": "Document processed and indexed for RAG. Graph processing acknowledged."
    }

@router.get("/info", response_model=DocumentInfoResponse)
async def get_document_info():
    raise HTTPException(status_code=501, detail="Not implemented yet. Document info is not persisted across sessions.")