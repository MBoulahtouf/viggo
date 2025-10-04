# viggo/core/services/rag_service.py
import os
import pickle
import spacy
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2, write_index, read_index
from typing import List, Dict, Tuple, Optional, Set
from groq import Groq
from viggo.core.config import settings

from viggo.core.services.graph_service import GraphService
from viggo.core.services.hybrid_search_service import HybridSearchService
from viggo.core.services.hybrid_retriever import HybridRetriever
from viggo.core.services.enhanced_entity_extractor import EnhancedEntityExtractor
from viggo.core.services.hybrid_chunking_service import HybridChunkingService, ChunkLevel, ChunkingConfig
from viggo.core.utils.entity_utils import filter_and_map_entities, get_entity_label_map, get_allowed_labels
from viggo.core.processors import DocumentProcessorFactory

class RAGService:
    def __init__(self, graph_service: GraphService, model_name: str = "all-MiniLM-L6-v2", index_path: str = "faiss_index.bin", doc_data_path: str = "document_data.pkl"):
        self.graph_service = graph_service
        self.hybrid_search_service = HybridSearchService(model_name)
        
        # Initialize hybrid retriever for parallel retrieval
        self.hybrid_retriever = None  # Will be initialized after document processing
        
        self.model = SentenceTransformer(model_name)
        self.nlp = spacy.load("en_core_web_sm") # Load spaCy model for sentence segmentation
        self.enhanced_extractor = EnhancedEntityExtractor(self.nlp)
        self.groq_client = Groq(api_key=settings.groq_api_key)
        self.document_processor_factory = DocumentProcessorFactory()
        
        # Initialize hybrid chunking service
        self.chunking_config = ChunkingConfig()
        self.hybrid_chunking = HybridChunkingService(config=self.chunking_config)
        
        # Legacy storage for backward compatibility
        self.index = None
        self.documents = [] # Stores the actual text chunks
        self.all_chunks_with_metadata = [] # Stores chunks with metadata (page, etc.)
        self.index_path = index_path
        self.doc_data_path = doc_data_path

        if os.path.exists(self.index_path) and os.path.exists(self.doc_data_path):
            self.index = read_index(self.index_path)
            with open(self.doc_data_path, 'rb') as f:
                self.documents, self.all_chunks_with_metadata = pickle.load(f)
            
            # Initialize hybrid retriever with loaded data
            self._initialize_hybrid_retriever()

    def _initialize_hybrid_retriever(self):
        """Initialize the hybrid retriever with current data."""
        if self.index and self.all_chunks_with_metadata:
            self.hybrid_retriever = HybridRetriever(
                vector_index=self.index,
                all_chunks_with_metadata=self.all_chunks_with_metadata,
                model_name="all-MiniLM-L6-v2"
            )
            # Set graph service if available
            if self.graph_service:
                self.hybrid_retriever.graph_service = self.graph_service

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

    def _chunk_document(self, document_store: List[Dict]) -> List[Dict]:
        """
        Chunk documents into smaller pieces for processing with format-aware strategies.
        
        Args:
            document_store: List of document pages with content
            
        Returns:
            List of chunks with metadata
        """
        chunks_with_metadata = []
        
        for doc_page in document_store:
            # Determine document type and chunking strategy
            chunking_strategy = self._determine_chunking_strategy(doc_page)
            
            if chunking_strategy == "epub_optimized":
                # Use EPUB-optimized chunking
                page_chunks = self._chunk_epub_content(doc_page)
            else:
                # Use standard PDF-style chunking
                page_chunks = self._chunk_standard_content(doc_page)
            
            chunks_with_metadata.extend(page_chunks)
        
        return chunks_with_metadata
    
    def _determine_chunking_strategy(self, doc_page: Dict) -> str:
        """
        Determine the best chunking strategy based on document metadata.
        
        Args:
            doc_page: Document page with metadata
            
        Returns:
            Chunking strategy name
        """
        # Check if this is EPUB content based on metadata
        if any(key in doc_page for key in ['chapter_title', 'chapter_metadata', 'word_count']):
            return "epub_optimized"
        
        # Default to standard chunking for PDFs and other formats
        return "standard"
    
    def _chunk_epub_content(self, doc_page: Dict) -> List[Dict]:
        """
        EPUB-optimized chunking strategy that respects chapter boundaries and content structure.
        
        Args:
            doc_page: EPUB document page with metadata
            
        Returns:
            List of chunks with metadata
        """
        text = doc_page['content']
        chapter_title = doc_page.get('chapter_title', '')
        word_count = doc_page.get('word_count', 0)
        
        # For short chapters, keep as single chunk
        if word_count < 300:
            return [self._process_chunk(text, doc_page.get("page"), {
                'chapter_title': chapter_title,
                'chunk_type': 'full_chapter',
                'word_count': word_count
            })]
        
        # For longer chapters, use paragraph-aware chunking
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_word_count = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_words = len(paragraph.split())
            
            # If adding this paragraph would exceed optimal chunk size, finalize current chunk
            if chunk_word_count + paragraph_words > 400 and current_chunk:
                chunks.append(self._process_chunk(current_chunk.strip(), doc_page.get("page"), {
                    'chapter_title': chapter_title,
                    'chunk_type': 'paragraph_group',
                    'word_count': chunk_word_count
                }))
                current_chunk = paragraph
                chunk_word_count = paragraph_words
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                chunk_word_count += paragraph_words
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._process_chunk(current_chunk.strip(), doc_page.get("page"), {
                'chapter_title': chapter_title,
                'chunk_type': 'paragraph_group',
                'word_count': chunk_word_count
            }))
        
        return chunks
    
    def _chunk_standard_content(self, doc_page: Dict) -> List[Dict]:
        """
        Standard chunking strategy for PDFs and other formats.
        
        Args:
            doc_page: Document page with content
            
        Returns:
            List of chunks with metadata
        """
        text = doc_page['content']
        doc = self.nlp(text)
        sentences = [sent for sent in doc.sents]
        current_chunk = ""
        chunks = []
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence.text) < 500:
                current_chunk += " " + sentence.text
            else:
                if current_chunk:
                    chunks.append(self._process_chunk(current_chunk.strip(), doc_page.get("page")))
                current_chunk = sentence.text
                
        if current_chunk:
            chunks.append(self._process_chunk(current_chunk.strip(), doc_page.get("page")))
        
        return chunks
    
    def _process_chunk(self, chunk_text: str, page_number: int, additional_metadata: Optional[Dict] = None) -> Dict:
        """
        Process a single chunk to extract entities and relationships using enhanced extraction.
        
        Args:
            chunk_text: The text content of the chunk
            page_number: The page number this chunk came from
            additional_metadata: Additional metadata for EPUB chunks
            
        Returns:
            Dictionary with chunk metadata including entities and relationships
        """
        # Use enhanced entity extractor for better filtering and deduplication
        entities = self.enhanced_extractor.extract_entities_enhanced(chunk_text, page_number)
        
        # Extract relationships using the enhanced entities
        chunk_doc = self.nlp(chunk_text)
        relationships = self.extract_relationships(chunk_doc, entities)
        
        # Build base chunk metadata
        chunk_metadata = {
            "content": chunk_text,
            "page": page_number,
            "entities": entities,
            "relationships": relationships,
            "word_count": len(chunk_text.split()),
            "char_count": len(chunk_text)
        }
        
        # Add EPUB-specific metadata if available
        if additional_metadata:
            chunk_metadata.update(additional_metadata)
        
        print(f"[DEBUG] Enhanced entities for chunk (page {page_number}): {entities}")
        print(f"[DEBUG] Enhanced relationships for chunk (page {page_number}): {relationships}")
        
        return chunk_metadata
    
    def _build_vector_index(self, chunks_with_metadata: List[Dict]) -> Tuple[IndexFlatL2, List[str]]:
        """
        Build FAISS vector index from chunks.
        
        Args:
            chunks_with_metadata: List of chunks with metadata
            
        Returns:
            Tuple of (FAISS index, list of document texts)
        """
        documents = [chunk["content"] for chunk in chunks_with_metadata]
        
        if not documents:
            raise ValueError("No documents to index")
        
        embeddings = self.model.encode(documents)
        index = IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        return index, documents
    
    def _save_index(self, index: IndexFlatL2, chunks_with_metadata: List[Dict]) -> None:
        """
        Save the FAISS index and document data to disk.
        
        Args:
            index: The FAISS index to save
            chunks_with_metadata: The chunks metadata to save
        """
        write_index(index, self.index_path)
        with open(self.doc_data_path, 'wb') as f:
            pickle.dump((self.documents, chunks_with_metadata), f)
    
    def build_rag_index(self, document_store: List[Dict]) -> Tuple[int, IndexFlatL2, List[Dict]]:
        """
        Build RAG index from document store using modular approach.
        
        Args:
            document_store: List of document pages with content
            
        Returns:
            Tuple of (number of chunks, FAISS index, chunks with metadata)
        """
        # Step 1: Chunk the documents
        chunks_with_metadata = self._chunk_document(document_store)
        
        if not chunks_with_metadata:
            return 0, None, []
        
        # Step 2: Build vector index
        index, documents = self._build_vector_index(chunks_with_metadata)
        
        # Step 3: Update instance variables
        self.documents = documents
        self.all_chunks_with_metadata = chunks_with_metadata
        self.index = index
        
        # Step 4: Save to disk
        self._save_index(index, chunks_with_metadata)
        
        # Step 5: Initialize hybrid retriever
        self._initialize_hybrid_retriever()
        
        return len(documents), index, chunks_with_metadata

    def process_document_enhanced(self, file_path: str) -> Tuple[int, IndexFlatL2, List[Dict]]:
        """
        Process document with enhanced entity extraction and content filtering.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (num_chunks, vector_index, filtered_chunks_with_metadata)
        """
        print(f"🔍 Processing document with enhanced entity extraction: {file_path}")
        
        # Process document normally first
        num_chunks, vector_index, chunks_with_metadata = self.process_document(file_path)
        
        # Apply enhanced processing to filter out noise and improve entities
        print("🧹 Applying enhanced content filtering and entity processing...")
        enhanced_chunks = self.enhanced_extractor.process_chunks_enhanced(chunks_with_metadata)
        
        print(f"✅ Enhanced processing complete:")
        print(f"   Original chunks: {len(chunks_with_metadata)}")
        print(f"   Filtered chunks: {len(enhanced_chunks)}")
        print(f"   Filtered out: {len(chunks_with_metadata) - len(enhanced_chunks)} noisy chunks")
        
        return len(enhanced_chunks), vector_index, enhanced_chunks
    
    def process_document_hybrid_chunking(self, file_path: str) -> Dict:
        """
        Process document using the new hybrid chunking strategy for reduced noise.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with hybrid chunking results and statistics
        """
        print(f"🏗️ Processing document with hybrid chunking strategy: {file_path}")
        
        # Step 1: Extract text using document processor
        all_pages_data = self.document_processor_factory.process_document(file_path)
        
        # Step 2: Apply hybrid chunking strategy
        chunking_result = self.hybrid_chunking.chunk_document_hierarchical(all_pages_data)
        
        # Step 3: Convert to legacy format for backward compatibility
        legacy_chunks = self._convert_hybrid_to_legacy_format(chunking_result["chunks"])
        
        # Step 4: Build vector index from passage-level chunks
        if legacy_chunks:
            index, documents = self._build_vector_index(legacy_chunks)
            
            # Update instance variables
            self.documents = documents
            self.all_chunks_with_metadata = legacy_chunks
            self.index = index
            
            # Save to disk
            self._save_index(index, legacy_chunks)
            
            # Initialize hybrid retriever
            self._initialize_hybrid_retriever()
            
            print(f"✅ Hybrid chunking processing complete:")
            print(f"   Total chunks: {chunking_result['statistics']['total_chunks']}")
            print(f"   Chapters: {len(chunking_result['chunks'].get(ChunkLevel.CHAPTER.value, []))}")
            print(f"   Passages: {len(chunking_result['chunks'].get(ChunkLevel.PASSAGE.value, []))}")
            print(f"   Overlapping: {len(chunking_result['chunks'].get(ChunkLevel.SENTENCE.value, []))}")
            
            return {
                "file_path": file_path,
                "num_chunks": len(legacy_chunks),
                "vector_index": index,
                "chunks_with_metadata": legacy_chunks,
                "hybrid_chunking_result": chunking_result,
                "processing_method": "hybrid_chunking"
            }
        else:
            print("❌ No chunks generated from hybrid chunking")
            return {
                "file_path": file_path,
                "num_chunks": 0,
                "vector_index": None,
                "chunks_with_metadata": [],
                "hybrid_chunking_result": chunking_result,
                "processing_method": "hybrid_chunking",
                "error": "No chunks generated"
            }
    
    def _convert_hybrid_to_legacy_format(self, hierarchical_chunks: Dict[str, List[Dict]]) -> List[Dict]:
        """Convert hybrid chunking results to legacy format for backward compatibility."""
        legacy_chunks = []
        
        # Use passage-level chunks as the primary chunks for legacy compatibility
        passage_chunks = hierarchical_chunks.get(ChunkLevel.PASSAGE.value, [])
        
        for chunk in passage_chunks:
            metadata = chunk["metadata"]
            legacy_chunk = {
                "content": chunk["content"],
                "page": metadata.page_number,
                "entities": metadata.entities,
                "relationships": metadata.relationships,
                "word_count": metadata.word_count,
                "char_count": metadata.char_count,
                "chapter_title": metadata.chapter_title,
                "chunk_type": chunk["chunk_type"],
                "lore_significance": metadata.lore_significance,
                "chunk_id": chunk["id"],
                "parent_id": metadata.parent_id,
                "level": chunk["level"]
            }
            legacy_chunks.append(legacy_chunk)
        
        return legacy_chunks

    def process_document(self, file_path: str) -> Tuple[int, IndexFlatL2, List[Dict]]:
        """
        Process a document file (PDF, EPUB, etc.) and build RAG index.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (number of chunks, FAISS index, chunks with metadata)
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        # Use the document processor factory to extract text
        all_pages_data = self.document_processor_factory.process_document(file_path)
        
        # Filter content pages (placeholder for now)
        document_store = self.find_content_pages(all_pages_data)
        
        # Build RAG index
        num_chunks, vector_index, all_chunks_with_metadata = self.build_rag_index(document_store)
        
        return num_chunks, vector_index, all_chunks_with_metadata
    
    def process_pdf(self, file_path: str) -> Tuple[int, IndexFlatL2, List[Dict]]:
        """
        Legacy method for backward compatibility.
        Now delegates to the generic process_document method.
        """
        return self.process_document(file_path)
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported document formats.
        
        Returns:
            List of supported file extensions
        """
        return self.document_processor_factory.get_supported_extensions()
    
    def is_format_supported(self, file_path: str) -> bool:
        """
        Check if a file format is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if format is supported, False otherwise
        """
        return self.document_processor_factory.is_supported(file_path)
    
    def get_document_metadata(self, file_path: str) -> Dict:
        """
        Get comprehensive metadata for a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with document metadata
        """
        processor = self.document_processor_factory.get_processor(file_path)
        if not processor:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Get format-specific metadata
        if hasattr(processor, 'get_epub_info'):
            return processor.get_epub_info(file_path)
        elif hasattr(processor, 'get_pdf_info'):
            return processor.get_pdf_info(file_path)
        else:
            return processor.get_file_info(file_path)
    
    def extract_document_for_hybrid_search(self, file_path: str) -> Dict:
        """
        Extract document content optimized for hybrid search (Azure Cognitive Search + FAISS).
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with content optimized for hybrid search
        """
        # Process document to get chunks
        num_chunks, vector_index, chunks_with_metadata = self.process_document(file_path)
        
        # Get document metadata
        doc_metadata = self.get_document_metadata(file_path)
        
        # Prepare content for Azure Cognitive Search indexing
        search_docs = []
        for chunk in chunks_with_metadata:
            search_doc = {
                "content": chunk["content"],
                "page": chunk["page"],
                "word_count": chunk.get("word_count", len(chunk["content"].split())),
                "char_count": chunk.get("char_count", len(chunk["content"])),
                "entities": [entity["text"] for entity in chunk.get("entities", [])],
                "entity_labels": [entity["label"] for entity in chunk.get("entities", [])],
                "relationships": chunk.get("relationships", []),
                "document_metadata": doc_metadata
            }
            
            # Add EPUB-specific fields if available
            if "chapter_title" in chunk:
                search_doc["chapter_title"] = chunk["chapter_title"]
                search_doc["chunk_type"] = chunk.get("chunk_type", "standard")
            
            search_docs.append(search_doc)
        
        # Calculate chunk statistics
        chunk_stats = self._calculate_chunk_statistics(chunks_with_metadata)
        
        return {
            "document_metadata": doc_metadata,
            "chunks": search_docs,
            "num_chunks": num_chunks,
            "vector_index": vector_index,
            "chunks_with_metadata": chunks_with_metadata,
            "chunk_statistics": chunk_stats
        }
    
    def index_document_in_azure_search(self, file_path: str) -> bool:
        """
        Process and index a document in Azure Cognitive Search.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract document for hybrid search
            doc_data = self.extract_document_for_hybrid_search(file_path)
            
            # Create index if it doesn't exist
            if not self.hybrid_search_service.create_index():
                print("Failed to create Azure Cognitive Search index")
                return False
            
            # Index documents
            success = self.hybrid_search_service.index_documents(doc_data["chunks"])
            
            if success:
                print(f"Successfully indexed {doc_data['num_chunks']} chunks in Azure Cognitive Search")
                return True
            else:
                print("Failed to index documents in Azure Cognitive Search")
                return False
                
        except Exception as e:
            print(f"Error indexing document in Azure Search: {e}")
            return False
    
    def _calculate_chunk_statistics(self, chunks_with_metadata: List[Dict]) -> Dict:
        """
        Calculate statistics about the chunks for analysis and optimization.
        
        Args:
            chunks_with_metadata: List of chunks with metadata
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks_with_metadata:
            return {}
        
        word_counts = [chunk.get("word_count", len(chunk["content"].split())) for chunk in chunks_with_metadata]
        char_counts = [chunk.get("char_count", len(chunk["content"])) for chunk in chunks_with_metadata]
        
        # Count chunk types (for EPUB)
        chunk_types = {}
        chapter_titles = set()
        
        for chunk in chunks_with_metadata:
            chunk_type = chunk.get("chunk_type", "standard")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            if "chapter_title" in chunk:
                chapter_titles.add(chunk["chapter_title"])
        
        return {
            "total_chunks": len(chunks_with_metadata),
            "avg_words_per_chunk": sum(word_counts) / len(word_counts),
            "min_words_per_chunk": min(word_counts),
            "max_words_per_chunk": max(word_counts),
            "avg_chars_per_chunk": sum(char_counts) / len(char_counts),
            "min_chars_per_chunk": min(char_counts),
            "max_chars_per_chunk": max(char_counts),
            "chunk_types": chunk_types,
            "unique_chapters": len(chapter_titles),
            "chapter_titles": list(chapter_titles)
        }

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
                model=settings.llm_model,
                temperature=settings.llm_temperature,
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

    def _search_relevant_chunks(self, question: str, page_number: Optional[int] = None, vector_index=None, all_chunks_with_metadata: List[Dict] = None) -> Tuple[List[str], Set[int]]:
        """
        Search for relevant chunks using vector similarity.
        
        Args:
            question: The query question
            page_number: Optional page number filter
            vector_index: Optional custom vector index
            all_chunks_with_metadata: Optional custom chunks metadata
            
        Returns:
            Tuple of (relevant_chunks_content, source_pages)
        """
        current_index = vector_index if vector_index is not None else self.index
        current_chunks_with_metadata = all_chunks_with_metadata if all_chunks_with_metadata is not None else self.all_chunks_with_metadata

        if current_index is None:
            print("[DEBUG] current_index is None. Returning empty results.")
            return [], set()

        print(f"[DEBUG] Number of chunks in metadata: {len(current_chunks_with_metadata)}")
        query_embedding = self.model.encode([question])
        print(f"[DEBUG] Query embedding shape: {query_embedding.shape}")
        distances, indices = current_index.search(query_embedding, 5)  # Get top 5 relevant chunks
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
        
        return relevant_chunks_content, source_pages

    def _generate_answer_with_llm(self, question: str, context: str) -> str:
        """
        Generate answer using LLM with the provided context.
        
        Args:
            question: The question to answer
            context: The context to use for answering
            
        Returns:
            The generated answer
        """
        prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say so.

Context: {context}

Question: {question}

Answer:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            return "I apologize, but I encountered an error while generating the answer."

    def perform_rag_query(self, question: str, page_number: int = None, vector_index=None, all_chunks_with_metadata: List[Dict] = None) -> Dict:
        """
        Perform RAG query using true hybrid retrieval with parallel search.
        
        Args:
            question: The question to answer
            page_number: Optional page number filter
            vector_index: Optional custom vector index (legacy support)
            all_chunks_with_metadata: Optional custom chunks metadata (legacy support)
            
        Returns:
            Dictionary with question, answer, source pages, and retrieval metadata
        """
        print(f"[DEBUG] Query received: question='{question}', page_number={page_number}")
        
        # Use hybrid retriever if available
        if self.hybrid_retriever:
            try:
                print("[DEBUG] Using hybrid retriever for parallel search")
                
                # Perform parallel hybrid retrieval
                import asyncio
                retrieval_result = asyncio.run(self.hybrid_retriever.retrieve(question, top_k=5, page_filter=page_number))
                
                results = retrieval_result["results"]
                metadata = retrieval_result["metadata"]
                
                if results:
                    print(f"[DEBUG] Found {len(results)} results from hybrid retrieval")
                    print(f"[DEBUG] Sources used: {metadata['sources_used']}")
                    print(f"[DEBUG] Retrieval times: {metadata['retrieval_times']}")
                    
                    # Create hybrid prompt with structured context
                    hybrid_prompt = self.hybrid_retriever.create_hybrid_prompt(question, results)
                    
                    # Generate answer with hybrid context
                    answer = self._generate_answer_with_llm(question, hybrid_prompt)
                    print(f"[DEBUG] Answer from LLM: {answer}")
                    
                    # Extract source pages from results
                    source_pages = set()
                    for result in results:
                        if result.get("page", 0) > 0:
                            source_pages.add(result["page"])
                    
                    return {
                        "question": question,
                        "answer": answer,
                        "source_pages": sorted(list(source_pages)) if source_pages else [],
                        "search_method": "hybrid_parallel",
                        "retrieval_metadata": metadata,
                        "sources_used": metadata["sources_used"]
                    }
                else:
                    print("[DEBUG] No results from hybrid retrieval, falling back to FAISS")
            except Exception as e:
                print(f"[DEBUG] Hybrid retrieval failed: {e}, falling back to FAISS")
        
        # Fallback to FAISS search if hybrid retriever fails or is not available
        print("[DEBUG] Falling back to FAISS search")
        relevant_chunks_content, source_pages = self._search_relevant_chunks(
            question, page_number, vector_index, all_chunks_with_metadata
        )
        
        if not relevant_chunks_content:
            return {
                "question": question,
                "answer": "No relevant information found in the document.",
                "source_pages": [],
                "search_method": "faiss_fallback"
            }
        
        # Get graph context
        graph_context = self._query_graph_for_context(question)
        
        # Combine contexts
        full_context = " ".join(relevant_chunks_content)
        if graph_context:
            full_context = f"{graph_context}\n\n{full_context}"

        print(f"[DEBUG] Relevant chunks content (before LLM): {relevant_chunks_content}")
        print(f"[DEBUG] Source pages: {source_pages}")
        print(f"[DEBUG] Context passed to LLM: {full_context[:500]}...")
        
        # Generate answer with LLM
        answer = self._generate_answer_with_llm(question, full_context)
        print(f"[DEBUG] Answer from LLM: {answer}")

        return {
            "question": question,
            "answer": answer,
            "source_pages": sorted(list(source_pages)) if source_pages else [],
            "search_method": "faiss_fallback"
        }