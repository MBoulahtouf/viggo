"""
Unit tests for aliasing service.
"""

import pytest
from viggo.core.services.aliasing_service import (
    AliasingService,
    AliasMapping,
    CanonicalGroup
)


class TestAliasingService:
    """Test cases for AliasingService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aliasing_service = AliasingService()
    
    def test_add_alias_mapping(self):
        """Test adding alias mappings."""
        # Add a mapping
        self.aliasing_service.add_alias_mapping("Tom", "Thomas", confidence=0.9, source="manual")
        
        # Verify it was added
        assert "tom" in self.aliasing_service.alias_to_canonical
        assert self.aliasing_service.alias_to_canonical["tom"] == "thomas"
        assert "thomas" in self.aliasing_service.canonical_to_aliases
        assert "tom" in self.aliasing_service.canonical_to_aliases["thomas"]
    
    def test_add_alias_mapping_self_reference(self):
        """Test that self-referencing aliases are ignored."""
        # Try to map to self
        self.aliasing_service.add_alias_mapping("John", "John")
        
        # Should not be added
        assert "john" not in self.aliasing_service.alias_to_canonical
    
    def test_resolve_to_canonical(self):
        """Test resolving aliases to canonical names."""
        # Add some mappings
        self.aliasing_service.add_alias_mapping("Tom", "Thomas")
        self.aliasing_service.add_alias_mapping("Tommy", "Thomas")
        
        # Test resolution
        assert self.aliasing_service.resolve_to_canonical("Tom") == "Thomas"
        assert self.aliasing_service.resolve_to_canonical("Tommy") == "Thomas"
        assert self.aliasing_service.resolve_to_canonical("Thomas") == "Thomas"
        assert self.aliasing_service.resolve_to_canonical("Unknown") == "Unknown"
    
    def test_get_all_aliases(self):
        """Test getting all aliases for a canonical name."""
        # Add some mappings
        self.aliasing_service.add_alias_mapping("Tom", "Thomas")
        self.aliasing_service.add_alias_mapping("Tommy", "Thomas")
        
        # Get all aliases
        aliases = self.aliasing_service.get_all_aliases("Thomas")
        
        assert "thomas" in aliases  # Canonical name itself
        assert "tom" in aliases
        assert "tommy" in aliases
        assert len(aliases) == 3
    
    def test_group_entities_with_aliases(self):
        """Test grouping entities with alias resolution."""
        # Add some mappings
        self.aliasing_service.add_alias_mapping("Tom", "Thomas")
        self.aliasing_service.add_alias_mapping("Tommy", "Thomas")
        
        # Create test entities
        entities = [
            {"name": "Tom", "labels": ["Character"]},
            {"name": "Tommy", "labels": ["Character"]},
            {"name": "Thomas", "labels": ["Character"]},
            {"name": "Alice", "labels": ["Character"]}
        ]
        
        # Group entities
        groups = self.aliasing_service.group_entities_with_aliases(entities)
        
        # Should have 2 groups: Thomas (with aliases) and Alice
        assert len(groups) == 2
        
        # Find the Thomas group
        thomas_group = next((g for g in groups if g.canonical == "thomas"), None)
        assert thomas_group is not None
        assert "Tom" in thomas_group.aliases
        assert "Tommy" in thomas_group.aliases
        assert "Thomas" in thomas_group.aliases
        assert "Character" in thomas_group.labels
    
    def test_suggest_aliases(self):
        """Test alias suggestion based on similarity."""
        # Create test entities
        all_entities = [
            {"name": "Thomas", "labels": ["Character"]},
            {"name": "Tom", "labels": ["Character"]},
            {"name": "Alice", "labels": ["Character"]},
            {"name": "Bob", "labels": ["Character"]}
        ]
        
        # Suggest aliases for "Thomas"
        suggestions = self.aliasing_service.suggest_aliases("Thomas", all_entities)
        
        # Should suggest "Tom" as it's similar
        assert "Tom" in suggestions
        # Should not suggest "Alice" or "Bob" as they're not similar
        assert "Alice" not in suggestions
        assert "Bob" not in suggestions
    
    def test_export_import_mappings(self):
        """Test exporting and importing mappings."""
        # Add some mappings
        self.aliasing_service.add_alias_mapping("Tom", "Thomas", confidence=0.9, source="manual")
        self.aliasing_service.add_alias_mapping("Alice", "Alice", confidence=1.0, source="direct")
        
        # Export mappings
        exported = self.aliasing_service.export_mappings()
        
        # Create new service and import
        new_service = AliasingService()
        new_service.import_mappings(exported)
        
        # Verify mappings were imported correctly
        assert new_service.resolve_to_canonical("Tom") == "Thomas"
        assert new_service.confidence_scores["tom"] == 0.9
        assert new_service.sources["tom"] == "manual"
    
    def test_custom_mappings_initialization(self):
        """Test initialization with custom mappings."""
        custom_mappings = {
            "Tom": "Thomas",
            "Bob": "Robert"
        }
        
        service = AliasingService(custom_mappings)
        
        # Verify custom mappings were added
        assert service.resolve_to_canonical("Tom") == "Thomas"
        assert service.resolve_to_canonical("Bob") == "Robert"
        assert service.sources["tom"] == "manual"
        assert service.sources["bob"] == "manual"


class TestAliasMapping:
    """Test cases for AliasMapping dataclass."""
    
    def test_alias_mapping_creation(self):
        """Test creating AliasMapping instances."""
        mapping = AliasMapping(
            alias="Tom",
            canonical="Thomas",
            confidence=0.9,
            source="manual"
        )
        
        assert mapping.alias == "Tom"
        assert mapping.canonical == "Thomas"
        assert mapping.confidence == 0.9
        assert mapping.source == "manual"


class TestCanonicalGroup:
    """Test cases for CanonicalGroup dataclass."""
    
    def test_canonical_group_creation(self):
        """Test creating CanonicalGroup instances."""
        group = CanonicalGroup(
            canonical="thomas",
            aliases={"thomas", "tom", "tommy"},
            labels={"Character"},
            confidence_scores={"tom": 0.9, "tommy": 0.8},
            sources={"tom": "manual", "tommy": "automatic"}
        )
        
        assert group.canonical == "thomas"
        assert "thomas" in group.aliases
        assert "tom" in group.aliases
        assert "tommy" in group.aliases
        assert "Character" in group.labels
        assert group.confidence_scores["tom"] == 0.9
        assert group.sources["tom"] == "manual"
