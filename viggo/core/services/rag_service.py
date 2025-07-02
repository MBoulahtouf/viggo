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

    def extract_relationships(self, doc: spacy.tokens.doc.Doc, filtered_entities=None) -> List[Dict]:
        relationships = []
        # Build a set of allowed entity texts for quick lookup
        allowed_entity_texts = set()
        allowed_entity_labels = dict()
        if filtered_entities is not None:
            for ent in filtered_entities:
                allowed_entity_texts.add(ent["text"])
                allowed_entity_labels[ent["text"]] = ent["label"]
        for sent in doc.sents:
            ents = [ent for ent in sent.ents if filtered_entities is None or (" ".join(ent.text.split()) in allowed_entity_texts)]
            if len(ents) > 1:
                root = sent.root
                if root.pos_ == 'VERB':
                    relationship_type = root.lemma_.upper()
                    if any(child.dep_ == "neg" for child in root.children):
                        relationship_type = "NOT_" + relationship_type
                    # Only create relationships between allowed types
                    for i in range(len(ents)):
                        for j in range(i + 1, len(ents)):
                            ent1 = " ".join(ents[i].text.split())
                            ent2 = " ".join(ents[j].text.split())
                            label1 = allowed_entity_labels.get(ent1)
                            label2 = allowed_entity_labels.get(ent2)
                            if label1 and label2 and label1 != "CARDINAL" and label2 != "CARDINAL":
                                relationships.append({
                                    "source": ent1,
                                    "target": ent2,
                                    "type": relationship_type
                                })
                else:
                    for i in range(len(ents)):
                        for j in range(i + 1, len(ents)):
                            ent1 = " ".join(ents[i].text.split())
                            ent2 = " ".join(ents[j].text.split())
                            label1 = allowed_entity_labels.get(ent1)
                            label2 = allowed_entity_labels.get(ent2)
                            if label1 and label2 and label1 != "CARDINAL" and label2 != "CARDINAL":
                                relationships.append({
                                    "source": ent1,
                                    "target": ent2,
                                    "type": "RELATED_TO"
                                })
        return relationships

    def build_rag_index(self, document_store: List[Dict]) -> Tuple[int, IndexFlatL2, List[Dict]]:
        self.documents = []
        self.all_chunks_with_metadata = []
        
        # Only keep these entity types and map to domain
        ENTITY_LABEL_MAP = {
            "PERSON": "Character",
            "ORG": "Organization",
            "GPE": "Location",
            "LOC": "Location"
        }
        ALLOWED_LABELS = set(ENTITY_LABEL_MAP.keys())

        for doc_page in document_store:
            text = doc_page['content']
            doc = self.nlp(text)
            sentences = [sent for sent in doc.sents]
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence.text) < 500:
                    current_chunk += " " + sentence.text
                else:
                    if current_chunk:
                        chunk_doc = self.nlp(current_chunk.strip())
                        # Filter and normalize entities
                        entities = []
                        for ent in chunk_doc.ents:
                            if ent.label_ in ALLOWED_LABELS:
                                ent_text = " ".join(ent.text.split()) # Normalize whitespace
                                entities.append({
                                    "text": ent_text,
                                    "label": ENTITY_LABEL_MAP[ent.label_]
                                })
                        relationships = self.extract_relationships(chunk_doc, entities)
                        print(f"[DEBUG] Filtered entities for chunk (page {doc_page.get('page')}): {entities}")
                        print(f"[DEBUG] Filtered relationships for chunk (page {doc_page.get('page')}): {relationships}")
                        self.documents.append(current_chunk.strip())
                        self.all_chunks_with_metadata.append({"content": current_chunk.strip(), "page": doc_page.get("page"), "entities": entities, "relationships": relationships})
                    current_chunk = sentence.text
            if current_chunk:
                chunk_doc = self.nlp(current_chunk.strip())
                entities = []
                for ent in chunk_doc.ents:
                    if ent.label_ in ALLOWED_LABELS:
                        ent_text = " ".join(ent.text.split())
                        entities.append({
                            "text": ent_text,
                            "label": ENTITY_LABEL_MAP[ent.label_]
                        })
                relationships = self.extract_relationships(chunk_doc, entities)
                print(f"[DEBUG] Filtered entities for last chunk (page {doc_page.get('page')}): {entities}")
                print(f"[DEBUG] Filtered relationships for last chunk (page {doc_page.get('page')}): {relationships}")
                self.documents.append(current_chunk.strip())
                self.all_chunks_with_metadata.append({"content": current_chunk.strip(), "page": doc_page.get("page"), "entities": entities, "relationships": relationships})

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
                related_info_list = self.graph_service.get_related_info_for_entity(entity_name, "Character")
                if related_info_list:
                    for info in related_info_list:
                        entity_info = f"{info["entity"]["name"]} ({', '.join(info["entity"]["labels"])})"
                        if "relationship" in info:
                            relationship_info = f"{info["relationship"]["type"]} {info["related_node"]["name"]} ({', '.join(info["related_node"]["labels"])})"
                            graph_context_parts.append(f"Knowledge Graph: {entity_info} {relationship_info}")
                        else:
                            graph_context_parts.append(f"Knowledge Graph: {entity_info}")
            elif entity_label == "LOC":
                related_info_list = self.graph_service.get_related_info_for_entity(entity_name, "Location")
                if related_info_list:
                    for info in related_info_list:
                        entity_info = f"{info["entity"]["name"]} ({', '.join(info["entity"]["labels"])})"
                        if "relationship" in info:
                            relationship_info = f"{info["relationship"]["type"]} {info["related_node"]["name"]} ({', '.join(info["related_node"]["labels"])})"
                            graph_context_parts.append(f"Knowledge Graph: {entity_info} {relationship_info}")
                        else:
                            graph_context_parts.append(f"Knowledge Graph: {entity_info}")
            elif entity_label == "ORG":
                related_info_list = self.graph_service.get_related_info_for_entity(entity_name, "Organization")
                if related_info_list:
                    for info in related_info_list:
                        entity_info = f"{info["entity"]["name"]} ({', '.join(info["entity"]["labels"])})"
                        if "relationship" in info:
                            relationship_info = f"{info["relationship"]["type"]} {info["related_node"]["name"]} ({', '.join(info["related_node"]["labels"])})"
                            graph_context_parts.append(f"Knowledge Graph: {entity_info} {relationship_info}")
                        else:
                            graph_context_parts.append(f"Knowledge Graph: {entity_info}")
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