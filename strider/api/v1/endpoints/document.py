# strider/api/v1/endpoints/document.py
import io
from fastapi import APIRouter, File, UploadFile, HTTPException
from pypdf import PdfReader
from strider.core.services import rag_service, graph_service
from strider.core import state
from strider.models.schemas import DocumentInfoResponse

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Reset state on new upload
    state.document_metadata, state.document_store, state.all_chunks_with_metadata = {}, [], []
    state.vector_index = None

    pdf_bytes = await file.read()
    all_pages_data = [{"page": i + 1, "content": page.extract_text()} for i, page in enumerate(PdfReader(io.BytesIO(pdf_bytes)).pages) if page.extract_text()]
    
    if not all_pages_data:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    state.document_store = rag_service.find_content_pages(all_pages_data)
    if not state.document_store:
        raise HTTPException(status_code=400, detail="Could not identify main content pages.")

    state.document_metadata = {
        "filename": file.filename, "total_pages": len(all_pages_data),
        "content_start_page": state.document_store[0]['page'],
        "content_end_page": state.document_store[-1]['page']
    }

    # Build RAG Index
    state.vector_index, state.all_chunks_with_metadata = rag_service.build_rag_index(state.document_store)

    # Knowledge Graph & Feature Population
    print("Starting fast graph and feature extraction...")
    nodes_added, rels_added, journeys = graph_service.extract_graph_and_features(state.document_store)

    # Store the character journeys
    state.document_metadata['character_journeys'] = journeys
    
    print(f"Graph populated: {nodes_added} nodes, {rels_added} relationships.")
    print(f"Character journeys tracked for {len(journeys)} characters.")
    
     

    return {
        **state.document_metadata,
        "rag_chunks_indexed": state.vector_index.ntotal,
        "graph_nodes_added": nodes_added,
        "graph_relationships_added": rels_added,
        "message": "Document processed.RAG, Graph, and Character Journeys are readu."
    }
    
    
"""
    # Build Knowledge Graph
    nodes_added, rels_added = graph_service.extract_and_load_graph(state.document_store)
"""    
    
@router.get("/info", response_model=DocumentInfoResponse)
async def get_document_info():
    if not state.document_metadata:
        raise HTTPException(status_code=404, detail="No document has been uploaded yet.")
    return state.document_metadata
