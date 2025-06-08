# strider/main.py

import io
import os
import numpy as np
import faiss
import spacy
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase

# --- Application Setup ---
load_dotenv() # Load environment variables from .env file

class Settings(BaseSettings):
    groq_api_key: str
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    class Config:
        env_file = ".env"

settings = Settings()

app = FastAPI(
    title="Strider API | Spoiler-Free Lore Companion & Knowledge Explorer",
    description="An API to ask questions about documents without getting spoilers and explore knowledge graphs from your favourite fictions.",
    version="0.2.5",
)

# --- Service Clients & Models ---
try:
    groq_client = Groq(api_key=settings.groq_api_key)
    nlp_model = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError(f"Failed to load a required service: {e}")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- In-Memory Storage & Model Loading ---
document_metadata = {} 
document_store, chunk_store, all_chunks_with_metadata = [], [], []
vector_index = None

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    page_number: int

class QueryResponse(BaseModel):
    answer: str
    source_pages: list[int]

class DocumentInfoResponse(BaseModel):
    filename: str
    total_pages: int
    content_start_page: int
    content_end_page: int

# --- Helper Function ---
def chunk_text(text, chunk_size=256, overlap=32):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# Heuristic to find the main content of the book
def find_content_pages(all_pages: list, sample_ratio=0.5, density_threshold_ratio=0.3) -> list:
    """Analyzes page text density to identify and return only the main content pages."""
    if not all_pages:
        return []

    word_counts = [len(page['content'].split()) for page in all_pages]
    
     # Avoid errors on empty or very short documents
    if not any(word_counts):
        return []
        
     # 1. Get a stable average word count from a central sample of the book
    sample_start = int(len(word_counts) * (0.5 - sample_ratio / 2))
    sample_end = int(len(word_counts) * (0.5 + sample_ratio / 2))
    sample_word_counts = word_counts[sample_start:sample_end]
    
    if not sample_word_counts:
        # Fallback for very short documents
        sample_word_counts = word_counts
        
    average_density = np.mean(sample_word_counts)
    density_threshold = average_density * density_threshold_ratio

    # 2. Find the first page from the beginning that meets the threshold
    start_page_index = 0
    for i, count in enumerate(word_counts):
        if count >= density_threshold:
            start_page_index = i
            break
    
    # 3. Find the first page from the end that meets the threshold
    end_page_index = len(word_counts) - 1
    for i in range(len(word_counts) - 1, -1, -1):
        if word_counts[i] >= density_threshold:
            end_page_index = i
            break
            
    # Ensure start is not after end
    if start_page_index > end_page_index:
        return []

    print(f"Content analysis: Found main content from page {all_pages[start_page_index]['page']} to {all_pages[end_page_index]['page']}.")
    
    return all_pages[start_page_index:end_page_index + 1]


def get_db_driver():
    """Establishes connection to the Neo4j database."""
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))

def add_node(tx, label, name):
    """Adds a node to the Neo4j graph."""
    tx.run(f"MERGE (n:{label} {{name: $name}})", name=name)

def add_relationship(tx, name1, name2, relationship="RELATED_TO"):
    """Adds a relationship between two nodes."""
    query = (
        f"MATCH (a {{name: $name1}}), (b {{name: $name2}}) "
        f"MERGE (a)-[r:{relationship}]->(b) "
        "RETURN type(r)"
    )
    tx.run(query, name1=name1, name2=name2)

def extract_and_load_graph(text_data: list):
    """Extracts entities and relationships and loads them into Neo4j."""
    driver = get_db_driver()
    nodes_added, rels_added = 0, 0

    with driver.session() as session:
        # Clear existing graph data for this document
        session.run("MATCH (n) DETACH DELETE n")
        for page in text_data:
            doc = nlp_model(page['content'])
            entities = [(ent.text.strip(), ent.label_) for ent in doc.ents if ent.label_ in ["PERSON", "GPE", "ORG", "LOC"]]  
            # Add all found entities as nodes
            for name, label in set(entities):
                if len(name) > 2: # Simple filter for quality
                    session.write_transaction(add_node, label, name)
                    nodes_added += 1
            # Infer relationships (co-occurrence in a sentence)
            for sentence in doc.sents:
                sent_entities = [ent.text.strip() for ent in sentence.ents if ent.label_ in ["PERSON", "GPE", "ORG", "LOC"]]
                if len(sent_entities) > 1:
                    for i in range(len(sent_entities)):
                        for j in range(i + 1, len(sent_entities)):
                            session.execute_write(add_relationship, sent_entities[i], sent_entities[j])
                            rels_added += 1
    driver.close()
    return nodes_added, rels_added

# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to Strider, your spoiler-free lore companion!"}


@app.post("/upload", tags=["Document Processing"])
async def upload_document(file: UploadFile = File(...)):
    """Uploads a document, builds the RAG index, and populates the knowledge graph."""
    global document_metadata, document_store, chunk_store, all_chunks_with_metadata, vector_index
    # Reset all stores
    document_metadata, document_store, chunk_store, all_chunks_with_metadata = {}, [], [], []
    vector_index = None

    
    pdf_bytes = await file.read()
    pdf_file = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    
    # Step 1: Extract text from ALL pages first
    all_pages_data = [{"page": i + 1, "content": page.extract_text()} for i, page in enumerate(reader.pages) if page.extract_text()]
    if not all_pages_data: raise HTTPException(status_code=400, detail="Could not extract text from PDF.")


    # Step 2: Use our new heuristic to find the actual content
    document_store = find_content_pages(all_pages_data)
    if not document_store: raise HTTPException(status_code=400, detail="Could not identify main content pages.")

    # Step 3: Store document metadata after processing
    document_metadata = {
        "filename": file.filename,
        "total_pages": len(all_pages_data),
        "content_start_page": document_store[0]['page'],
        "content_end_page": document_store[-1]['page']
    }

    # --- RAG Indexing ---
    for doc in document_store:
        page_chunks = chunk_text(doc["content"])
        for chunk in page_chunks: all_chunks_with_metadata.append({"chunk_text": chunk, "source_page": doc["page"]})
    chunk_store = [item["chunk_text"] for item in all_chunks_with_metadata]
    chunk_embeddings = embedding_model.encode(chunk_store, convert_to_tensor=False)
    embedding_dimension = chunk_embeddings.shape[1]
    vector_index = faiss.IndexFlatL2(embedding_dimension)
    vector_index.add(chunk_embeddings.astype(np.float32))
    nodes_added, rels_added = extract_and_load_graph(document_store)
    
    return {
        **document_metadata,
        "rag_chunks_indexed": vector_index.ntotal,
        "graph_nodes_added": nodes_added,
        "graph_relationships_added": rels_added,
        "message": "Document processed for RAG and Knowledge Graph."
    }

@app.get("/document/info", response_model=DocumentInfoResponse, tags=["Document Processing"])
async def get_document_info():
    """Returns metadata about the currently processed document, including the content page range."""
    if not document_metadata:
        raise HTTPException(status_code=404, detail="No document has been uploaded yet.")
    return document_metadata
    
    
@app.post("/query", response_model=QueryResponse, tags=["Q&A"])
async def query_document(request: QueryRequest):
    global vector_index, all_chunks_with_metadata

    if vector_index is None:
        raise HTTPException(status_code=400, detail="No document has been uploaded and indexed yet.")

    # 1. SPOILER GUARDRAIL: Filter chunks based on the user's page number
    spoiler_free_indices = [
        i for i, meta in enumerate(all_chunks_with_metadata) 
        if meta["source_page"] <= request.page_number
    ]
    if not spoiler_free_indices:
        raise HTTPException(status_code=404, detail="No content available at or before the specified page.")

    # 2. Embed the user's question
    question_embedding = embedding_model.encode([request.question], convert_to_tensor=False)

    # 3. Search the index for the most relevant chunks (within the spoiler-free set)
    # We create a new, temporary index with only the spoiler-free vectors
    spoiler_free_vectors = np.array([vector_index.reconstruct(i) for i in spoiler_free_indices]).astype('float32')
    temp_index = faiss.IndexFlatL2(spoiler_free_vectors.shape[1])
    temp_index.add(spoiler_free_vectors)
    
    k = 3 # Number of relevant chunks to retrieve
    distances, retrieved_indices_in_temp = temp_index.search(question_embedding.astype(np.float32), k)

    # Map indices from the temporary index back to the original `all_chunks_with_metadata` index
    original_indices = [spoiler_free_indices[i] for i in retrieved_indices_in_temp[0]]

    # 4. Retrieve the context and source pages
    relevant_context = "\n---\n".join([all_chunks_with_metadata[i]["chunk_text"] for i in original_indices])
    source_pages = sorted(list(set([all_chunks_with_metadata[i]["source_page"] for i in original_indices])))
    
    # 5. Build the prompt and call the LLM
    prompt = f"""
    You are an expert on the provided text. Answer the user's question based *only* on the following context.
    Do not use any outside knowledge. If the context is not sufficient, say that you cannot answer.

    CONTEXT:
    {relevant_context}

    QUESTION:
    {request.question}
    """
    
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )
    answer = chat_completion.choices[0].message.content

    return {"answer": answer, "source_pages": source_pages}


