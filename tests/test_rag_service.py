"""
Unit tests for RAGService.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from viggo.core.services.rag_service import RAGService
from viggo.core.services.graph_service import GraphService


class TestRAGService:
    """Test cases for RAGService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_graph_service = Mock(spec=GraphService)
        self.mock_model = Mock()
        self.mock_nlp = Mock()
        self.mock_groq_client = Mock()
        
        # Mock spaCy model loading
        with patch('viggo.core.services.rag_service.spacy.load', return_value=self.mock_nlp):
            with patch('viggo.core.services.rag_service.SentenceTransformer', return_value=self.mock_model):
                with patch('viggo.core.services.rag_service.Groq', return_value=self.mock_groq_client):
                    self.rag_service = RAGService(
                        graph_service=self.mock_graph_service,
                        model_name="test-model",
                        index_path="test_index.bin",
                        doc_data_path="test_docs.pkl"
                    )
    
    def test_initialization(self):
        """Test RAGService initialization."""
        assert self.rag_service.graph_service == self.mock_graph_service
        assert self.rag_service.model == self.mock_model
        assert self.rag_service.nlp == self.mock_nlp
        assert self.rag_service.groq_client == self.mock_groq_client
        assert self.rag_service.index is None
        assert self.rag_service.documents == []
        assert self.rag_service.all_chunks_with_metadata == []
    
    def test_chunk_document(self):
        """Test document chunking functionality."""
        # Mock spaCy document and sentences
        mock_doc = Mock()
        mock_sent1 = Mock()
        mock_sent1.text = "This is a short sentence."
        mock_sent2 = Mock()
        mock_sent2.text = "This is another short sentence."
        mock_doc.sents = [mock_sent1, mock_sent2]
        
        self.mock_nlp.return_value = mock_doc
        
        # Mock entity filtering
        with patch('viggo.core.services.rag_service.filter_and_map_entities', return_value=[]):
            with patch.object(self.rag_service, 'extract_relationships', return_value=[]):
                document_store = [
                    {"content": "This is a short sentence. This is another short sentence.", "page": 1}
                ]
                
                chunks = self.rag_service._chunk_document(document_store)
                
                assert len(chunks) == 1
                assert chunks[0]["content"] == "This is a short sentence. This is another short sentence."
                assert chunks[0]["page"] == 1
                assert chunks[0]["entities"] == []
                assert chunks[0]["relationships"] == []
    
    def test_process_chunk(self):
        """Test processing individual chunks."""
        # Mock spaCy document
        mock_doc = Mock()
        self.mock_nlp.return_value = mock_doc
        
        # Mock entity filtering and relationship extraction
        with patch('viggo.core.services.rag_service.filter_and_map_entities', return_value=[{"text": "Alice", "label": "Character"}]):
            with patch.object(self.rag_service, 'extract_relationships', return_value=[{"type": "KNOWS", "source": "Alice", "target": "Bob"}]):
                chunk_metadata = self.rag_service._process_chunk("Alice knows Bob.", 1)
                
                assert chunk_metadata["content"] == "Alice knows Bob."
                assert chunk_metadata["page"] == 1
                assert chunk_metadata["entities"] == [{"text": "Alice", "label": "Character"}]
                assert chunk_metadata["relationships"] == [{"type": "KNOWS", "source": "Alice", "target": "Bob"}]
    
    def test_build_vector_index(self):
        """Test building vector index from chunks."""
        # Mock model encoding
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.mock_model.encode.return_value = mock_embeddings
        
        chunks_with_metadata = [
            {"content": "First chunk", "page": 1, "entities": [], "relationships": []},
            {"content": "Second chunk", "page": 2, "entities": [], "relationships": []}
        ]
        
        index, documents = self.rag_service._build_vector_index(chunks_with_metadata)
        
        assert documents == ["First chunk", "Second chunk"]
        assert index is not None
        self.mock_model.encode.assert_called_once_with(["First chunk", "Second chunk"])
    
    def test_build_vector_index_empty_chunks(self):
        """Test building vector index with empty chunks."""
        with pytest.raises(ValueError, match="No documents to index"):
            self.rag_service._build_vector_index([])
    
    def test_save_index(self):
        """Test saving index to disk."""
        # Mock FAISS index
        mock_index = Mock()
        
        chunks_with_metadata = [
            {"content": "Test chunk", "page": 1, "entities": [], "relationships": []}
        ]
        
        with patch('viggo.core.services.rag_service.write_index') as mock_write_index:
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('viggo.core.services.rag_service.pickle.dump') as mock_pickle_dump:
                    self.rag_service._save_index(mock_index, chunks_with_metadata)
                    
                    mock_write_index.assert_called_once_with(mock_index, "test_index.bin")
                    mock_pickle_dump.assert_called_once()
    
    def test_build_rag_index(self):
        """Test building complete RAG index."""
        # Mock all sub-methods
        mock_chunks = [{"content": "Test chunk", "page": 1, "entities": [], "relationships": []}]
        mock_index = Mock()
        mock_documents = ["Test chunk"]
        
        with patch.object(self.rag_service, '_chunk_document', return_value=mock_chunks):
            with patch.object(self.rag_service, '_build_vector_index', return_value=(mock_index, mock_documents)):
                with patch.object(self.rag_service, '_save_index'):
                    document_store = [{"content": "Test content", "page": 1}]
                    
                    num_chunks, index, chunks = self.rag_service.build_rag_index(document_store)
                    
                    assert num_chunks == 1
                    assert index == mock_index
                    assert chunks == mock_chunks
                    assert self.rag_service.documents == mock_documents
                    assert self.rag_service.all_chunks_with_metadata == mock_chunks
                    assert self.rag_service.index == mock_index
    
    def test_build_rag_index_empty_documents(self):
        """Test building RAG index with empty documents."""
        with patch.object(self.rag_service, '_chunk_document', return_value=[]):
            num_chunks, index, chunks = self.rag_service.build_rag_index([])
            
            assert num_chunks == 0
            assert index is None
            assert chunks == []
    
    def test_search_relevant_chunks(self):
        """Test searching for relevant chunks."""
        # Mock FAISS index and search results
        mock_index = Mock()
        mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
        
        # Mock model encoding
        self.mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Mock chunks metadata
        chunks_with_metadata = [
            {"content": "First chunk", "page": 1},
            {"content": "Second chunk", "page": 2}
        ]
        
        relevant_chunks, source_pages = self.rag_service._search_relevant_chunks(
            "test question", page_number=None, vector_index=mock_index, all_chunks_with_metadata=chunks_with_metadata
        )
        
        assert relevant_chunks == ["First chunk", "Second chunk"]
        assert source_pages == {1, 2}
        mock_index.search.assert_called_once()
    
    def test_search_relevant_chunks_with_page_filter(self):
        """Test searching for relevant chunks with page filter."""
        # Mock FAISS index and search results
        mock_index = Mock()
        mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
        
        # Mock model encoding
        self.mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Mock chunks metadata
        chunks_with_metadata = [
            {"content": "First chunk", "page": 1},
            {"content": "Second chunk", "page": 3}  # This should be filtered out
        ]
        
        relevant_chunks, source_pages = self.rag_service._search_relevant_chunks(
            "test question", page_number=2, vector_index=mock_index, all_chunks_with_metadata=chunks_with_metadata
        )
        
        assert relevant_chunks == ["First chunk"]  # Only page 1 is <= 2
        assert source_pages == {1}
    
    def test_search_relevant_chunks_no_index(self):
        """Test searching with no index available."""
        relevant_chunks, source_pages = self.rag_service._search_relevant_chunks("test question")
        
        assert relevant_chunks == []
        assert source_pages == set()
    
    def test_generate_answer_with_llm(self):
        """Test generating answer with LLM."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is the answer."
        self.mock_groq_client.chat.completions.create.return_value = mock_response
        
        answer = self.rag_service._generate_answer_with_llm("What is this?", "Context here")
        
        assert answer == "This is the answer."
        self.mock_groq_client.chat.completions.create.assert_called_once()
    
    def test_generate_answer_with_llm_failure(self):
        """Test LLM failure handling."""
        # Mock LLM failure
        self.mock_groq_client.chat.completions.create.side_effect = Exception("API error")
        
        answer = self.rag_service._generate_answer_with_llm("What is this?", "Context here")
        
        assert answer == "I apologize, but I encountered an error while generating the answer."
    
    def test_perform_rag_query(self):
        """Test complete RAG query process."""
        # Mock all sub-methods
        with patch.object(self.rag_service, '_search_relevant_chunks', return_value=(["chunk1", "chunk2"], {1, 2})):
            with patch.object(self.rag_service, '_query_graph_for_context', return_value="graph context"):
                with patch.object(self.rag_service, '_generate_answer_with_llm', return_value="Generated answer"):
                    result = self.rag_service.perform_rag_query("test question", page_number=5)
                    
                    assert result["question"] == "test question"
                    assert result["answer"] == "Generated answer"
                    assert result["source_pages"] == [1, 2]
    
    def test_perform_rag_query_no_relevant_chunks(self):
        """Test RAG query with no relevant chunks found."""
        with patch.object(self.rag_service, '_search_relevant_chunks', return_value=([], set())):
            result = self.rag_service.perform_rag_query("test question")
            
            assert result["question"] == "test question"
            assert result["answer"] == "No relevant information found in the document."
            assert result["source_pages"] == []
    
    def test_extract_relationships(self):
        """Test relationship extraction."""
        # Mock spaCy document and entities
        mock_doc = Mock()
        mock_sent = Mock()
        mock_ent1 = Mock()
        mock_ent1.text = "Alice"
        mock_ent2 = Mock()
        mock_ent2.text = "Bob"
        mock_sent.ents = [mock_ent1, mock_ent2]
        mock_sent.root.pos_ = "VERB"
        mock_sent.root.lemma_ = "know"
        mock_doc.sents = [mock_sent]
        
        filtered_entities = [{"text": "Alice", "label": "Character"}, {"text": "Bob", "label": "Character"}]
        
        relationships = self.rag_service.extract_relationships(mock_doc, filtered_entities)
        
        assert len(relationships) == 1
        assert relationships[0]["type"] == "KNOW"
        assert relationships[0]["source"] == "Alice"
        assert relationships[0]["target"] == "Bob"
    
    def test_extract_relationships_no_verb(self):
        """Test relationship extraction with no verb."""
        # Mock spaCy document and entities
        mock_doc = Mock()
        mock_sent = Mock()
        mock_ent1 = Mock()
        mock_ent1.text = "Alice"
        mock_ent2 = Mock()
        mock_ent2.text = "Bob"
        mock_sent.ents = [mock_ent1, mock_ent2]
        mock_sent.root.pos_ = "NOUN"  # No verb
        mock_doc.sents = [mock_sent]
        
        filtered_entities = [{"text": "Alice", "label": "Character"}, {"text": "Bob", "label": "Character"}]
        
        relationships = self.rag_service.extract_relationships(mock_doc, filtered_entities)
        
        assert len(relationships) == 1
        assert relationships[0]["type"] == "RELATED_TO"
        assert relationships[0]["source"] == "Alice"
        assert relationships[0]["target"] == "Bob"


def mock_open():
    """Mock open function for file operations."""
    return MagicMock()
