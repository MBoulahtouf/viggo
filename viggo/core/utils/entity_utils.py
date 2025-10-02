"""
Entity normalization and label mapping utilities for Viggo.

This module provides functions for:
- Mapping spaCy entity labels to domain-specific types
- Normalizing entity names (whitespace, case handling)
- Filtering entities by allowed types
"""

from typing import Dict, List, Set, Optional
import spacy
from spacy.tokens import Doc, Span


# Domain-specific entity label mapping
ENTITY_LABEL_MAP = {
    "PERSON": "Character",
    "ORG": "Organization", 
    "GPE": "Location",
    "LOC": "Location"
}

# Allowed spaCy entity labels for extraction
ALLOWED_LABELS = set(ENTITY_LABEL_MAP.keys())


def normalize_entity_name(entity_text: str) -> str:
    """
    Normalize entity name by cleaning whitespace and standardizing format.
    
    Args:
        entity_text: Raw entity text from spaCy
        
    Returns:
        Normalized entity name
    """
    return " ".join(entity_text.split())


def filter_and_map_entities(doc: Doc, allowed_labels: Optional[Set[str]] = None) -> List[Dict[str, str]]:
    """
    Extract entities from spaCy doc, filter by allowed labels, and map to domain types.
    
    Args:
        doc: spaCy document
        allowed_labels: Set of spaCy labels to keep (defaults to ALLOWED_LABELS)
        
    Returns:
        List of filtered entities with normalized names and domain labels
    """
    if allowed_labels is None:
        allowed_labels = ALLOWED_LABELS
        
    entities = []
    for ent in doc.ents:
        if ent.label_ in allowed_labels:
            normalized_text = normalize_entity_name(ent.text)
            entities.append({
                "text": normalized_text,
                "label": ENTITY_LABEL_MAP[ent.label_]
            })
    return entities


def get_entity_label_map() -> Dict[str, str]:
    """Get the current entity label mapping."""
    return ENTITY_LABEL_MAP.copy()


def get_allowed_labels() -> Set[str]:
    """Get the set of allowed spaCy entity labels."""
    return ALLOWED_LABELS.copy()


def add_custom_label_mapping(spacy_label: str, domain_label: str) -> None:
    """
    Add a custom mapping from spaCy label to domain label.
    
    Args:
        spacy_label: spaCy entity label (e.g., "WORK_OF_ART")
        domain_label: Domain-specific label (e.g., "Book")
    """
    ENTITY_LABEL_MAP[spacy_label] = domain_label
    ALLOWED_LABELS.add(spacy_label)


def remove_label_mapping(spacy_label: str) -> None:
    """
    Remove a label mapping and from allowed labels.
    
    Args:
        spacy_label: spaCy entity label to remove
    """
    if spacy_label in ENTITY_LABEL_MAP:
        del ENTITY_LABEL_MAP[spacy_label]
    ALLOWED_LABELS.discard(spacy_label)
