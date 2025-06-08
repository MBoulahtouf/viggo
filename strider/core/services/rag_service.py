# strider/core/services/rag_service.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from strider.core.config import settings

# Initialize models and clients here
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=settings.groq_api_key)

def chunk_text(text, chunk_size=256, overlap=32):
    words = text.split(); chunks = [];
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def find_content_pages(all_pages: list, sample_ratio=0.5, density_threshold_ratio=0.3) -> list:
    if not all_pages: return []
    word_counts = [len(page['content'].split()) for page in all_pages]
    if not any(word_counts): return []
    sample_start = int(len(word_counts) * (0.5 - sample_ratio / 2))
    sample_end = int(len(word_counts) * (0.5 + sample_ratio / 2))
    sample_word_counts = word_counts[sample_start:sample_end] or word_counts
    average_density = np.mean(sample_word_counts)
    density_threshold = average_density * density_threshold_ratio
    start_page_index = 0
    for i, count in enumerate(word_counts):
        if count >= density_threshold:
            start_page_index = i
            break
    end_page_index = len(word_counts) - 1
    for i in range(len(word_counts) - 1, -1, -1):
        if word_counts[i] >= density_threshold:
            end_page_index = i
            break
    if start_page_index > end_page_index: return []
    print(f"Content analysis: Found main content from page {all_pages[start_page_index]['page']} to {all_pages[end_page_index]['page']}.")
    return all_pages[start_page_index:end_page_index + 1]

def build_rag_index(document_store: list) -> tuple:
    all_chunks_with_metadata = []
    for doc in document_store:
        page_chunks = chunk_text(doc["content"])
        for chunk in page_chunks:
            all_chunks_with_metadata.append({"chunk_text": chunk, "source_page": doc["page"]})
    
    chunk_texts = [item["chunk_text"] for item in all_chunks_with_metadata]
    chunk_embeddings = embedding_model.encode(chunk_texts, convert_to_tensor=False)
    
    embedding_dimension = chunk_embeddings.shape[1]
    vector_index = faiss.IndexFlatL2(embedding_dimension)
    vector_index.add(chunk_embeddings.astype(np.float32))
    
    return vector_index, all_chunks_with_metadata

def perform_rag_query(question: str, page_number: int, vector_index: faiss.Index, all_chunks_with_metadata: list) -> dict:
    spoiler_free_indices = [i for i, meta in enumerate(all_chunks_with_metadata) if meta["source_page"] <= page_number]
    if not spoiler_free_indices: return None

    question_embedding = embedding_model.encode([question], convert_to_tensor=False)
    
    spoiler_free_vectors = np.array([vector_index.reconstruct(i) for i in spoiler_free_indices]).astype('float32')
    temp_index = faiss.IndexFlatL2(spoiler_free_vectors.shape[1])
    temp_index.add(spoiler_free_vectors)
    
    k = min(3, len(spoiler_free_vectors))
    distances, retrieved_indices_in_temp = temp_index.search(question_embedding.astype(np.float32), k)
    
    original_indices = [spoiler_free_indices[i] for i in retrieved_indices_in_temp[0]]
    
    relevant_context = "\n---\n".join([all_chunks_with_metadata[i]["chunk_text"] for i in original_indices])
    source_pages = sorted(list(set([all_chunks_with_metadata[i]["source_page"] for i in original_indices])))
    
    prompt = f"CONTEXT:\n{relevant_context}\n\nQUESTION:\n{question}\n\nAnswer based only on the context."
    chat_completion = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama3-8b-8192")
    answer = chat_completion.choices[0].message.content
    
    return {"answer": answer, "source_pages": source_pages}
