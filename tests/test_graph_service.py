"""
Unit tests for GraphService.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from viggo.core.services.graph_service import (
    GraphService,
    GraphServiceError,
    PaginationParams,
    NodeResult,
    RelationshipResult,
    EntityGraphResult
)


class TestPaginationParams:
    """Test cases for PaginationParams."""
    
    def test_valid_pagination_params(self):
        """Test creating valid pagination parameters."""
        params = PaginationParams(limit=50, offset=10)
        assert params.limit == 50
        assert params.offset == 10
    
    def test_default_pagination_params(self):
        """Test default pagination parameters."""
        params = PaginationParams()
        assert params.limit == 100
        assert params.offset == 0
    
    def test_invalid_limit(self):
        """Test invalid limit raises ValueError."""
        with pytest.raises(ValueError, match="Limit must be positive"):
            PaginationParams(limit=0)
        
        with pytest.raises(ValueError, match="Limit must be positive"):
            PaginationParams(limit=-1)
    
    def test_invalid_offset(self):
        """Test invalid offset raises ValueError."""
        with pytest.raises(ValueError, match="Offset must be non-negative"):
            PaginationParams(offset=-1)


class TestGraphService:
    """Test cases for GraphService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the Neo4j driver
        self.mock_driver = Mock()
        self.mock_session = Mock()
        self.mock_driver.session.return_value.__enter__.return_value = self.mock_session
        self.mock_driver.session.return_value.__exit__.return_value = None
    
    @patch('viggo.core.services.graph_service.GraphDatabase')
    def test_successful_initialization(self, mock_graph_db):
        """Test successful GraphService initialization."""
        mock_graph_db.driver.return_value = self.mock_driver
        self.mock_session.run.return_value.single.return_value = {"deleted_count": 0}
        
        service = GraphService("bolt://localhost:7687", "neo4j", "password", clear_on_startup=False)
        
        assert service.driver == self.mock_driver
        assert service.aliasing_service is not None
    
    @patch('viggo.core.services.graph_service.GraphDatabase')
    def test_initialization_with_connection_failure(self, mock_graph_db):
        """Test GraphService initialization with connection failure."""
        mock_graph_db.driver.side_effect = Exception("Connection failed")
        
        with pytest.raises(GraphServiceError, match="Failed to connect to Neo4j"):
            GraphService("bolt://localhost:7687", "neo4j", "password")
    
    @patch('viggo.core.services.graph_service.GraphDatabase')
    def test_initialization_with_custom_aliases(self, mock_graph_db):
        """Test GraphService initialization with custom aliases."""
        mock_graph_db.driver.return_value = self.mock_driver
        self.mock_session.run.return_value.single.return_value = {"deleted_count": 0}
        
        custom_aliases = {"Tom": "Thomas", "Bob": "Robert"}
        service = GraphService("bolt://localhost:7687", "neo4j", "password", custom_aliases=custom_aliases)
        
        assert service.aliasing_service.resolve_to_canonical("Tom") == "Thomas"
        assert service.aliasing_service.resolve_to_canonical("Bob") == "Robert"
    
    @patch('viggo.core.services.graph_service.GraphDatabase')
    def test_clear_database(self, mock_graph_db):
        """Test clearing the database."""
        mock_graph_db.driver.return_value = self.mock_driver
        self.mock_session.run.return_value.single.return_value = {"deleted_count": 5}
        
        service = GraphService("bolt://localhost:7687", "neo4j", "password", clear_on_startup=False)
        service.clear_database()
        
        self.mock_session.run.assert_called_with("MATCH (n) DETACH DELETE n RETURN count(n) as deleted_count")
    
    @patch('viggo.core.services.graph_service.GraphDatabase')
    def test_clear_database_failure(self, mock_graph_db):
        """Test clearing the database with failure."""
        mock_graph_db.driver.return_value = self.mock_driver
        self.mock_session.run.side_effect = Exception("Database error")
        
        service = GraphService("bolt://localhost:7687", "neo4j", "password", clear_on_startup=False)
        
        with pytest.raises(GraphServiceError, match="Failed to clear database"):
            service.clear_database()
    
    @patch('viggo.core.services.graph_service.GraphDatabase')
    def test_close(self, mock_graph_db):
        """Test closing the driver connection."""
        mock_graph_db.driver.return_value = self.mock_driver
        self.mock_session.run.return_value.single.return_value = {"deleted_count": 0}
        
        service = GraphService("bolt://localhost:7687", "neo4j", "password", clear_on_startup=False)
        service.close()
        
        self.mock_driver.close.assert_called_once()
    
    @patch('viggo.core.services.graph_service.GraphDatabase')
    def test_list_all_nodes_with_label(self, mock_graph_db):
        """Test listing nodes with label filter."""
        mock_graph_db.driver.return_value = self.mock_driver
        
        # Mock query result
        mock_result = [
            {"name": "Alice", "labels": ["Character"], "properties": {"age": 30}},
            {"name": "Bob", "labels": ["Character"], "properties": {"age": 25}}
        ]
        self.mock_session.run.return_value = mock_result
        
        service = GraphService("bolt://localhost:7687", "neo4j", "password", clear_on_startup=False)
        nodes = service.list_all_nodes(label="Character")
        
        assert len(nodes) == 2
        assert isinstance(nodes[0], NodeResult)
        assert nodes[0].name == "Alice"
        assert nodes[0].labels == ["Character"]
        assert nodes[0].properties == {"age": 30}
    
    @patch('viggo.core.services.graph_service.GraphDatabase')
    def test_list_all_nodes_with_pagination(self, mock_graph_db):
        """Test listing nodes with pagination."""
        mock_graph_db.driver.return_value = self.mock_driver
        
        # Mock query result
        mock_result = [{"name": "Alice", "labels": ["Character"], "properties": {}}]
        self.mock_session.run.return_value = mock_result
        
        service = GraphService("bolt://localhost:7687", "neo4j", "password", clear_on_startup=False)
        pagination = PaginationParams(limit=10, offset=5)
        nodes = service.list_all_nodes(pagination=pagination)
        
        # Verify pagination parameters were passed
        call_args = self.mock_session.run.call_args
        assert call_args[1]["limit"] == 10
        assert call_args[1]["offset"] == 5
    
    @patch('viggo.core.services.graph_service.GraphDatabase')
    def test_list_all_nodes_failure(self, mock_graph_db):
        """Test listing nodes with failure."""
        mock_graph_db.driver.return_value = self.mock_driver
        self.mock_session.run.side_effect = Exception("Query failed")
        
        service = GraphService("bolt://localhost:7687", "neo4j", "password", clear_on_startup=False)
        
        with pytest.raises(GraphServiceError, match="Failed to list nodes"):
            service.list_all_nodes()
    
    @patch('viggo.core.services.graph_service.GraphDatabase')
    def test_grouped_nodes(self, mock_graph_db):
        """Test grouping nodes by canonical name."""
        mock_graph_db.driver.return_value = self.mock_driver
        
        # Mock query result with duplicate names
        mock_result = [
            {"name": "Alice", "labels": ["Character"], "properties": {}},
            {"name": "alice", "labels": ["Character"], "properties": {}},
            {"name": "Bob", "labels": ["Character"], "properties": {}}
        ]
        self.mock_session.run.return_value = mock_result
        
        service = GraphService("bolt://localhost:7687", "neo4j", "password", clear_on_startup=False)
        grouped = service.grouped_nodes()
        
        assert len(grouped) == 2  # Alice and Bob groups
        
        # Find Alice group
        alice_group = next((g for g in grouped if g["canonical"] == "alice"), None)
        assert alice_group is not None
        assert "Alice" in alice_group["aliases"]
        assert "alice" in alice_group["aliases"]
        assert "Character" in alice_group["labels"]
    
    @patch('viggo.core.services.graph_service.GraphDatabase')
    def test_add_alias_mapping(self, mock_graph_db):
        """Test adding alias mappings."""
        mock_graph_db.driver.return_value = self.mock_driver
        self.mock_session.run.return_value.single.return_value = {"deleted_count": 0}
        
        service = GraphService("bolt://localhost:7687", "neo4j", "password", clear_on_startup=False)
        service.add_alias_mapping("Tom", "Thomas", confidence=0.9, source="manual")
        
        assert service.aliasing_service.resolve_to_canonical("Tom") == "Thomas"
    
    @patch('viggo.core.services.graph_service.GraphDatabase')
    def test_resolve_entity_name(self, mock_graph_db):
        """Test resolving entity names to canonical form."""
        mock_graph_db.driver.return_value = self.mock_driver
        self.mock_session.run.return_value.single.return_value = {"deleted_count": 0}
        
        service = GraphService("bolt://localhost:7687", "neo4j", "password", clear_on_startup=False)
        service.add_alias_mapping("Tom", "Thomas")
        
        assert service.resolve_entity_name("Tom") == "Thomas"
        assert service.resolve_entity_name("Unknown") == "Unknown"
    
    @patch('viggo.core.services.graph_service.GraphDatabase')
    def test_get_entity_with_aliases(self, mock_graph_db):
        """Test getting entity data with aliases."""
        mock_graph_db.driver.return_value = self.mock_driver
        self.mock_session.run.return_value.single.return_value = {"deleted_count": 0}
        
        # Mock entity data
        mock_entity_data = {
            "name": "Thomas",
            "labels": ["Character"],
            "properties": {"age": 30},
            "relationships": []
        }
        
        service = GraphService("bolt://localhost:7687", "neo4j", "password", clear_on_startup=False)
        service.add_alias_mapping("Tom", "Thomas")
        
        # Mock the get_related_info_for_entity method
        service.get_related_info_for_entity = Mock(return_value=mock_entity_data)
        
        result = service.get_entity_with_aliases("Tom")
        
        assert result["canonical_name"] == "Thomas"
        assert result["entity_data"] == mock_entity_data
        assert "Tom" in result["aliases"]
        assert "Thomas" in result["aliases"]
        assert result["alias_count"] == 2
    
    @patch('viggo.core.services.graph_service.GraphDatabase')
    def test_suggest_aliases_for_entity(self, mock_graph_db):
        """Test suggesting aliases for an entity."""
        mock_graph_db.driver.return_value = self.mock_driver
        self.mock_session.run.return_value.single.return_value = {"deleted_count": 0}
        
        # Mock list_all_nodes result
        mock_nodes = [
            NodeResult(name="Thomas", labels=["Character"], properties={}),
            NodeResult(name="Tom", labels=["Character"], properties={}),
            NodeResult(name="Alice", labels=["Character"], properties={})
        ]
        
        service = GraphService("bolt://localhost:7687", "neo4j", "password", clear_on_startup=False)
        service.list_all_nodes = Mock(return_value=mock_nodes)
        
        suggestions = service.suggest_aliases_for_entity("Thomas")
        
        # Should suggest "Tom" as it's similar to "Thomas"
        assert "Tom" in suggestions
        # Should not suggest "Alice" as it's not similar
        assert "Alice" not in suggestions


class TestNodeResult:
    """Test cases for NodeResult dataclass."""
    
    def test_node_result_creation(self):
        """Test creating NodeResult instances."""
        node = NodeResult(
            name="Alice",
            labels=["Character"],
            properties={"age": 30}
        )
        
        assert node.name == "Alice"
        assert node.labels == ["Character"]
        assert node.properties == {"age": 30}


class TestRelationshipResult:
    """Test cases for RelationshipResult dataclass."""
    
    def test_relationship_result_creation(self):
        """Test creating RelationshipResult instances."""
        target_node = NodeResult(name="Bob", labels=["Character"], properties={})
        relationship = RelationshipResult(
            type="KNOWS",
            properties={"since": 2020},
            target_node=target_node
        )
        
        assert relationship.type == "KNOWS"
        assert relationship.properties == {"since": 2020}
        assert relationship.target_node == target_node


class TestEntityGraphResult:
    """Test cases for EntityGraphResult dataclass."""
    
    def test_entity_graph_result_creation(self):
        """Test creating EntityGraphResult instances."""
        target_node = NodeResult(name="Bob", labels=["Character"], properties={})
        relationship = RelationshipResult(
            type="KNOWS",
            properties={},
            target_node=target_node
        )
        
        entity_graph = EntityGraphResult(
            name="Alice",
            labels=["Character"],
            properties={"age": 30},
            relationships=[relationship]
        )
        
        assert entity_graph.name == "Alice"
        assert entity_graph.labels == ["Character"]
        assert entity_graph.properties == {"age": 30}
        assert len(entity_graph.relationships) == 1
        assert entity_graph.relationships[0].type == "KNOWS"
