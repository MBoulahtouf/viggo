"""
Unit tests for entity utilities.
"""

import pytest
import spacy
from viggo.core.utils.entity_utils import (
    normalize_entity_name,
    filter_and_map_entities,
    get_entity_label_map,
    get_allowed_labels,
    add_custom_label_mapping,
    remove_label_mapping
)


class TestEntityUtils:
    """Test cases for entity utility functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Load spaCy model for testing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")
    
    def test_normalize_entity_name(self):
        """Test entity name normalization."""
        # Test basic normalization
        assert normalize_entity_name("  John   Doe  ") == "John Doe"
        assert normalize_entity_name("Alice\nBob") == "Alice Bob"
        assert normalize_entity_name("Charlie\tDavid") == "Charlie David"
        
        # Test edge cases
        assert normalize_entity_name("") == ""
        assert normalize_entity_name("   ") == ""
        assert normalize_entity_name("Single") == "Single"
    
    def test_filter_and_map_entities(self):
        """Test entity filtering and mapping."""
        # Create a test document
        text = "John Smith works at Microsoft in Seattle."
        doc = self.nlp(text)
        
        # Test with default allowed labels
        entities = filter_and_map_entities(doc)
        
        # Should find PERSON, ORG, and GPE entities
        entity_texts = [ent["text"] for ent in entities]
        entity_labels = [ent["label"] for ent in entities]
        
        assert "John Smith" in entity_texts
        assert "Microsoft" in entity_texts
        assert "Seattle" in entity_texts
        
        assert "Character" in entity_labels
        assert "Organization" in entity_labels
        assert "Location" in entity_labels
    
    def test_filter_and_map_entities_with_custom_labels(self):
        """Test entity filtering with custom allowed labels."""
        text = "John Smith works at Microsoft in Seattle."
        doc = self.nlp(text)
        
        # Test with only PERSON entities
        entities = filter_and_map_entities(doc, allowed_labels={"PERSON"})
        
        entity_texts = [ent["text"] for ent in entities]
        assert "John Smith" in entity_texts
        assert "Microsoft" not in entity_texts
        assert "Seattle" not in entity_texts
    
    def test_get_entity_label_map(self):
        """Test getting entity label mapping."""
        label_map = get_entity_label_map()
        
        assert isinstance(label_map, dict)
        assert label_map["PERSON"] == "Character"
        assert label_map["ORG"] == "Organization"
        assert label_map["GPE"] == "Location"
        assert label_map["LOC"] == "Location"
    
    def test_get_allowed_labels(self):
        """Test getting allowed labels."""
        allowed_labels = get_allowed_labels()
        
        assert isinstance(allowed_labels, set)
        assert "PERSON" in allowed_labels
        assert "ORG" in allowed_labels
        assert "GPE" in allowed_labels
        assert "LOC" in allowed_labels
    
    def test_add_custom_label_mapping(self):
        """Test adding custom label mappings."""
        # Add a custom mapping
        add_custom_label_mapping("WORK_OF_ART", "Book")
        
        # Verify it was added
        label_map = get_entity_label_map()
        allowed_labels = get_allowed_labels()
        
        assert label_map["WORK_OF_ART"] == "Book"
        assert "WORK_OF_ART" in allowed_labels
    
    def test_remove_label_mapping(self):
        """Test removing label mappings."""
        # First add a custom mapping
        add_custom_label_mapping("WORK_OF_ART", "Book")
        
        # Then remove it
        remove_label_mapping("WORK_OF_ART")
        
        # Verify it was removed
        label_map = get_entity_label_map()
        allowed_labels = get_allowed_labels()
        
        assert "WORK_OF_ART" not in label_map
        assert "WORK_OF_ART" not in allowed_labels
    
    def test_remove_nonexistent_label_mapping(self):
        """Test removing a label mapping that doesn't exist."""
        # Should not raise an error
        remove_label_mapping("NONEXISTENT_LABEL")
        
        # Verify original mappings are still intact
        label_map = get_entity_label_map()
        assert "PERSON" in label_map
        assert "ORG" in label_map
