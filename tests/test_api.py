"""
API integration tests for Viggo endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import io
from viggo.main import app


class TestDocumentEndpoints:
    """Test cases for document endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    @patch('viggo.dependencies.get_rag_service')
    @patch('viggo.dependencies.get_graph_service')
    @patch('viggo.core.utils.file_ops.clear_indexes_and_graph')
    def test_upload_document_success(self, mock_clear, mock_get_graph, mock_get_rag):
        """Test successful document upload."""
        # Mock services
        mock_rag_service = Mock()
        mock_rag_service.process_pdf.return_value = (10, Mock(), [])
        mock_graph_service = Mock()
        mock_get_rag.return_value = mock_rag_service
        mock_get_graph.return_value = mock_graph_service
        
        # Create test PDF content
        test_pdf_content = b"Test PDF content"
        
        # Mock file operations
        with patch('builtins.open', mock_open()):
            with patch('viggo.core.config.settings.data_dir', '/test/data'):
                response = self.client.post(
                    "/api/v1/documents/upload",
                    files={"file": ("test.pdf", io.BytesIO(test_pdf_content), "application/pdf")}
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert data["num_chunks_indexed"] == 10
        assert "Document processed and indexed" in data["message"]
        
        # Verify services were called
        mock_clear.assert_called_once_with(mock_rag_service, mock_graph_service)
        mock_rag_service.process_pdf.assert_called_once()
        mock_graph_service.extract_and_load_graph.assert_called_once()
    
    @patch('viggo.dependencies.get_rag_service')
    @patch('viggo.dependencies.get_graph_service')
    def test_upload_document_no_file(self, mock_get_graph, mock_get_rag):
        """Test document upload without file."""
        response = self.client.post("/api/v1/documents/upload")
        
        assert response.status_code == 422  # Validation error


class TestQueryEndpoints:
    """Test cases for query endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    @patch('viggo.dependencies.get_rag_service')
    def test_query_document_success(self, mock_get_rag):
        """Test successful document query."""
        # Mock RAG service
        mock_rag_service = Mock()
        mock_rag_service.index = Mock()  # Has index
        mock_rag_service.all_chunks_with_metadata = []
        mock_rag_service.perform_rag_query.return_value = {
            "question": "What is this?",
            "answer": "This is a test answer.",
            "source_pages": [1, 2]
        }
        mock_get_rag.return_value = mock_rag_service
        
        query_data = {
            "question": "What is this?",
            "page_number": 5
        }
        
        response = self.client.post("/api/v1/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is this?"
        assert data["answer"] == "This is a test answer."
        assert data["source_pages"] == [1, 2]
        
        mock_rag_service.perform_rag_query.assert_called_once_with(
            question="What is this?",
            page_number=5,
            vector_index=mock_rag_service.index,
            all_chunks_with_metadata=[]
        )
    
    @patch('viggo.dependencies.get_rag_service')
    def test_query_document_no_index(self, mock_get_rag):
        """Test query without indexed document."""
        # Mock RAG service without index
        mock_rag_service = Mock()
        mock_rag_service.index = None
        mock_get_rag.return_value = mock_rag_service
        
        query_data = {
            "question": "What is this?",
            "page_number": 5
        }
        
        response = self.client.post("/api/v1/query", json=query_data)
        
        assert response.status_code == 400
        assert "No document has been indexed yet" in response.json()["detail"]
    
    def test_query_document_malformed_request(self):
        """Test query with malformed request."""
        query_data = {
            "question": "What is this?"
            # Missing page_number
        }
        
        response = self.client.post("/api/v1/query", json=query_data)
        
        assert response.status_code == 422  # Validation error


class TestGraphEndpoints:
    """Test cases for graph endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    @patch('viggo.dependencies.get_graph_service')
    def test_get_entity_graph_data_success(self, mock_get_graph):
        """Test successful entity graph data retrieval."""
        # Mock graph service
        mock_graph_service = Mock()
        mock_graph_service.get_related_info_for_entity.return_value = {
            "name": "Alice",
            "labels": ["Character"],
            "properties": {"age": 30},
            "relationships": []
        }
        mock_get_graph.return_value = mock_graph_service
        
        response = self.client.get("/api/v1/graph/entity/Alice?entity_label=Character")
        
        assert response.status_code == 200
        data = response.json()
        assert data["entity_name"] == "Alice"
        assert data["graph_data"]["name"] == "Alice"
        assert data["graph_data"]["labels"] == ["Character"]
        
        mock_graph_service.get_related_info_for_entity.assert_called_once_with(
            "Alice", "Character", excluded_rel_types=None, excluded_node_labels=None
        )
    
    @patch('viggo.dependencies.get_graph_service')
    def test_get_entity_graph_data_not_found(self, mock_get_graph):
        """Test entity graph data retrieval when entity not found."""
        # Mock graph service
        mock_graph_service = Mock()
        mock_graph_service.get_related_info_for_entity.return_value = {}
        mock_get_graph.return_value = mock_graph_service
        
        response = self.client.get("/api/v1/graph/entity/Unknown")
        
        assert response.status_code == 404
        assert "Entity not found" in response.json()["detail"]
    
    @patch('viggo.dependencies.get_graph_service')
    def test_list_all_nodes_success(self, mock_get_graph):
        """Test successful node listing."""
        # Mock graph service
        mock_graph_service = Mock()
        from viggo.core.services.graph_service import NodeResult
        mock_nodes = [
            NodeResult(name="Alice", labels=["Character"], properties={"age": 30}),
            NodeResult(name="Bob", labels=["Character"], properties={"age": 25})
        ]
        mock_graph_service.list_all_nodes.return_value = mock_nodes
        mock_get_graph.return_value = mock_graph_service
        
        response = self.client.get("/api/v1/graph/nodes")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["nodes"]) == 2
        assert data["nodes"][0]["name"] == "Alice"
        assert data["nodes"][0]["labels"] == ["Character"]
        assert data["nodes"][0]["properties"] == {"age": 30}
    
    @patch('viggo.dependencies.get_graph_service')
    def test_list_all_nodes_with_pagination(self, mock_get_graph):
        """Test node listing with pagination."""
        # Mock graph service
        mock_graph_service = Mock()
        from viggo.core.services.graph_service import NodeResult
        mock_nodes = [NodeResult(name="Alice", labels=["Character"], properties={})]
        mock_graph_service.list_all_nodes.return_value = mock_nodes
        mock_get_graph.return_value = mock_graph_service
        
        response = self.client.get("/api/v1/graph/nodes?limit=10&offset=5")
        
        assert response.status_code == 200
        # Verify pagination parameters were passed
        mock_graph_service.list_all_nodes.assert_called_once()
        call_args = mock_graph_service.list_all_nodes.call_args
        assert call_args[1]["pagination"].limit == 10
        assert call_args[1]["pagination"].offset == 5
    
    @patch('viggo.dependencies.get_graph_service')
    def test_list_all_nodes_with_label_filter(self, mock_get_graph):
        """Test node listing with label filter."""
        # Mock graph service
        mock_graph_service = Mock()
        from viggo.core.services.graph_service import NodeResult
        mock_nodes = [NodeResult(name="Alice", labels=["Character"], properties={})]
        mock_graph_service.list_all_nodes.return_value = mock_nodes
        mock_get_graph.return_value = mock_graph_service
        
        response = self.client.get("/api/v1/graph/nodes?label=Character")
        
        assert response.status_code == 200
        mock_graph_service.list_all_nodes.assert_called_once_with(label="Character", pagination=None)
    
    @patch('viggo.dependencies.get_graph_service')
    def test_grouped_nodes_success(self, mock_get_graph):
        """Test successful grouped nodes retrieval."""
        # Mock graph service
        mock_graph_service = Mock()
        mock_grouped_data = [
            {
                "canonical": "alice",
                "aliases": ["Alice", "alice"],
                "labels": ["Character"]
            }
        ]
        mock_graph_service.grouped_nodes.return_value = mock_grouped_data
        mock_get_graph.return_value = mock_graph_service
        
        response = self.client.get("/api/v1/graph/grouped_nodes")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["grouped_nodes"]) == 1
        assert data["grouped_nodes"][0]["canonical"] == "alice"
        assert "Alice" in data["grouped_nodes"][0]["aliases"]
    
    @patch('viggo.dependencies.get_graph_service')
    def test_get_entity_with_aliases_success(self, mock_get_graph):
        """Test successful entity with aliases retrieval."""
        # Mock graph service
        mock_graph_service = Mock()
        mock_entity_data = {
            "canonical_name": "Thomas",
            "entity_data": {"name": "Thomas", "labels": ["Character"], "properties": {}, "relationships": []},
            "aliases": ["Thomas", "Tom", "Tommy"],
            "alias_count": 3
        }
        mock_graph_service.get_entity_with_aliases.return_value = mock_entity_data
        mock_get_graph.return_value = mock_graph_service
        
        response = self.client.get("/api/v1/graph/entity/Thomas/aliases")
        
        assert response.status_code == 200
        data = response.json()
        assert data["canonical_name"] == "Thomas"
        assert data["alias_count"] == 3
        assert "Tom" in data["aliases"]
    
    @patch('viggo.dependencies.get_graph_service')
    def test_add_alias_mapping_success(self, mock_get_graph):
        """Test successful alias mapping addition."""
        # Mock graph service
        mock_graph_service = Mock()
        mock_get_graph.return_value = mock_graph_service
        
        response = self.client.post(
            "/api/v1/graph/aliases?alias=Tom&canonical=Thomas&confidence=0.9&source=manual"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "Added alias mapping" in data["message"]
        assert data["confidence"] == 0.9
        assert data["source"] == "manual"
        
        mock_graph_service.add_alias_mapping.assert_called_once_with("Tom", "Thomas", 0.9, "manual")
    
    @patch('viggo.dependencies.get_graph_service')
    def test_suggest_aliases_success(self, mock_get_graph):
        """Test successful alias suggestion."""
        # Mock graph service
        mock_graph_service = Mock()
        mock_suggestions = ["Tom", "Tommy"]
        mock_graph_service.suggest_aliases_for_entity.return_value = mock_suggestions
        mock_get_graph.return_value = mock_graph_service
        
        response = self.client.get("/api/v1/graph/entity/Thomas/suggest-aliases")
        
        assert response.status_code == 200
        data = response.json()
        assert data["entity_name"] == "Thomas"
        assert data["suggested_aliases"] == ["Tom", "Tommy"]
        
        mock_graph_service.suggest_aliases_for_entity.assert_called_once_with("Thomas")


class TestAPIErrorHandling:
    """Test cases for API error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    @patch('viggo.dependencies.get_graph_service')
    def test_graph_service_error_handling(self, mock_get_graph):
        """Test error handling in graph endpoints."""
        # Mock graph service that raises an exception
        mock_graph_service = Mock()
        mock_graph_service.list_all_nodes.side_effect = Exception("Database error")
        mock_get_graph.return_value = mock_graph_service
        
        response = self.client.get("/api/v1/graph/nodes")
        
        assert response.status_code == 500
        assert "Failed to list nodes" in response.json()["detail"]
    
    @patch('viggo.dependencies.get_rag_service')
    def test_rag_service_error_handling(self, mock_get_rag):
        """Test error handling in RAG endpoints."""
        # Mock RAG service that raises an exception
        mock_rag_service = Mock()
        mock_rag_service.index = Mock()
        mock_rag_service.all_chunks_with_metadata = []
        mock_rag_service.perform_rag_query.side_effect = Exception("LLM error")
        mock_get_rag.return_value = mock_rag_service
        
        query_data = {
            "question": "What is this?",
            "page_number": 5
        }
        
        response = self.client.post("/api/v1/query", json=query_data)
        
        assert response.status_code == 500


def mock_open():
    """Mock open function for file operations."""
    return MagicMock()
