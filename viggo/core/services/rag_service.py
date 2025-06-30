# viggo/core/services/rag_service.py
import os
import pickle
import spacy
from pypdf import PdfReader # Added back PdfReader import
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2, write_index, read_index
from typing import List, Dict, Tuple
from groq import Groq
from viggo.core.config import settings

from viggo.core.services.graph_service import GraphService

class RAGService:
    def __init__(self, graph_service: GraphService, model_name: str = "all-MiniLM-L6-v2", index_path: str = "faiss_index.bin", doc_data_path: str = "document_data.pkl"):
        self.graph_service = graph_service
        self.model = SentenceTransformer(model_name)
        self.nlp = spacy.load("en_core_web_sm") # Load spaCy model for sentence segmentation
        self.groq_client = Groq(api_key=settings.groq_api_key)
        self.index = None
        self.documents = [] # Stores the actual text chunks
        self.all_chunks_with_metadata = [] # Stores chunks with metadata (page, etc.)
        self.index_path = index_path
        self.doc_data_path = doc_data_path

        if os.path.exists(self.index_path) and os.path.exists(self.doc_data_path):
            self.index = read_index(self.index_path)
            with open(self.doc_data_path, 'rb') as f:
                self.documents, self.all_chunks_with_metadata = pickle.load(f)

    def find_content_pages(self, all_pages_data: List[Dict]) -> List[Dict]:
        # This is a placeholder for actual content page identification logic
        # For now, it returns all pages as content pages
        return all_pages_data

    def build_rag_index(self, document_store: List[Dict]) -> Tuple[int, IndexFlatL2, List[Dict]]:
        self.documents = []
        self.all_chunks_with_metadata = []
        
        for doc_page in document_store:
            text = doc_page['content']
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
            
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 500: # Keep chunks under 500 characters
                    current_chunk += " " + sentence
                else:
                    if current_chunk:
                        # Extract entities for the current chunk
                        chunk_doc = self.nlp(current_chunk.strip())
                        entities = [{"text": ent.text, "label": ent.label_} for ent in chunk_doc.ents]
                        print(f"[DEBUG] Extracted entities for chunk (page {doc_page.get('page')}): {entities}")
                        self.documents.append(current_chunk.strip())
                        self.all_chunks_with_metadata.append({"content": current_chunk.strip(), "page": doc_page.get("page"), "entities": entities})
                    current_chunk = sentence
            if current_chunk: # Add the last chunk
                chunk_doc = self.nlp(current_chunk.strip())
                entities = [{"text": ent.text, "label": ent.label_} for ent in chunk_doc.ents]
                print(f"[DEBUG] Extracted entities for last chunk (page {doc_page.get('page')}): {entities}")
                self.documents.append(current_chunk.strip())
                self.all_chunks_with_metadata.append({"content": current_chunk.strip(), "page": doc_page.get("page"), "entities": entities})

        if not self.documents:
            return 0, None, []

        embeddings = self.model.encode(self.documents)
        self.index = IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        write_index(self.index, self.index_path)
        with open(self.doc_data_path, 'wb') as f:
            pickle.dump((self.documents, self.all_chunks_with_metadata), f)

        return len(self.documents), self.index, self.all_chunks_with_metadata

    def process_pdf(self, file_path: str) -> Tuple[int, IndexFlatL2, List[Dict]]:
        reader = PdfReader(file_path)
        all_pages_data = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                all_pages_data.append({"page": i + 1, "content": text})
                print(f"[DEBUG] Extracted text from page {i+1}: {text[:200]}...") # Print first 200 chars of page text
        
        document_store = self.find_content_pages(all_pages_data)
        num_chunks, vector_index, all_chunks_with_metadata = self.build_rag_index(document_store)
        
        return num_chunks, vector_index, all_chunks_with_metadata

    def query(self, query_text: str, k: int = 5) -> List[Dict]:
        if self.index is None:
            return []

        query_embedding = self.model.encode([query_text])
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({"content": self.documents[idx], "distance": distances[0][i], "metadata": self.all_chunks_with_metadata[idx]}) # Added metadata
        return results

    def _generate_answer_with_llm(self, question: str, context: str) -> str:
        if not context:
            return "No relevant information found to answer the question."

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides concise, spoiler-free answers based on the given context. If the answer is not in the context, state that you don't have enough information."
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\nContext: {context}\nAnswer:"
                    }
                ],
                model="llama3-8b-8192", # Using a fast model for quick responses
                temperature=0.1,
                max_tokens=150,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer with LLM: {e}")
            return "Error generating answer."

    def _query_graph_for_context(self, question: str) -> str:
        doc = self.nlp(question)
        graph_context_parts = []

        for ent in doc.ents:
            entity_name = ent.text
            entity_label = ent.label_

            # Query for related information based on entity type
            if entity_label == "PERSON":
                related_info = self.graph_service.get_related_info_for_entity(entity_name, "Character")
                if related_info:
                    graph_context_parts.append(f"Knowledge Graph about {entity_name} (Character):\n{related_info}")
            elif entity_label == "LOC":
                related_info = self.graph_service.get_related_info_for_entity(entity_name, "Location")
                if related_info:
                    graph_context_parts.append(f"Knowledge Graph about {entity_name} (Location):\n{related_info}")
            elif entity_label == "ORG":
                related_info = self.graph_service.get_related_info_for_entity(entity_name, "Organization")
                if related_info:
                    graph_context_parts.append(f"Knowledge Graph about {entity_name} (Organization):\n{related_info}")
            # Add more entity types as needed

        if graph_context_parts:
            return "\n\n".join(graph_context_parts)
        return ""

    def perform_rag_query(self, question: str, page_number: int = None, vector_index=None, all_chunks_with_metadata: List[Dict] = None) -> Dict:
        print(f"[DEBUG] Query received: question='{question}', page_number={page_number}")
        current_index = vector_index if vector_index is not None else self.index
        current_chunks_with_metadata = all_chunks_with_metadata if all_chunks_with_metadata is not None else self.all_chunks_with_metadata

        if current_index is None:
            print("[DEBUG] current_index is None. Returning None.")
            return None

        print(f"[DEBUG] Number of chunks in metadata: {len(current_chunks_with_metadata)}")
        query_embedding = self.model.encode([question])
        print(f"[DEBUG] Query embedding shape: {query_embedding.shape}")
        distances, indices = current_index.search(query_embedding, 5) # Get top 5 relevant chunks
        print(f"[DEBUG] FAISS search results - distances: {distances}, indices: {indices}")

        relevant_chunks_content = []
        source_pages = set()
        for i, idx in enumerate(indices[0]):
            if idx < len(current_chunks_with_metadata):
                chunk_info = current_chunks_with_metadata[idx]
                # Check if page_number is provided and if the chunk's page matches or is within the allowed range
                if page_number is None or (chunk_info.get("page") is not None and chunk_info.get("page") <= page_number):
                    relevant_chunks_content.append(chunk_info["content"])
                    if chunk_info.get("page"):
                        source_pages.add(chunk_info["page"])
        
        graph_context = self._query_graph_for_context(question)
        full_context = " ".join(relevant_chunks_content)
        if graph_context:
            full_context = f"{graph_context}\n\n{full_context}"

        print(f"[DEBUG] Relevant chunks content (before LLM): {relevant_chunks_content}")
        print(f"[DEBUG] Source pages: {source_pages}")

        context = full_context
        print(f"[DEBUG] Context passed to LLM: {context[:500]}...") # Print first 500 chars of context
        answer = self._generate_answer_with_llm(question, context)
        print(f"[DEBUG] Answer from LLM: {answer}")

        return {
            "question": question,
            "answer": answer,
            "source_pages": sorted(list(source_pages)) if source_pages else []
        }