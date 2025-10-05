"""
Entity normalization and label mapping utilities for Viggo.

This module provides functions for:
- Mapping spaCy entity labels to domain-specific types
- Normalizing entity names (whitespace, case handling)
- Filtering entities by allowed types
- Entity relationship extraction for hybrid RAG
- Entity confidence scoring and validation
"""

from typing import Dict, List, Set, Optional, Tuple
import spacy
from spacy.tokens import Doc, Span
import re
from dataclasses import dataclass


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


@dataclass
class EntityWithContext:
    """Entity with surrounding context for hybrid RAG."""
    name: str
    entity_type: str
    confidence: float
    start_char: int
    end_char: int
    context_before: str
    context_after: str
    page_number: int
    relationships: List[str]


def extract_entities_with_context(
    text: str, 
    page_number: int = 1,
    context_window: int = 50
) -> List[EntityWithContext]:
    """
    Extract entities with surrounding context for better relationship detection.
    
    Args:
        text: Text to extract entities from
        page_number: Page number for reference
        context_window: Number of characters before/after entity for context
        
    Returns:
        List of entities with context information
    """
    entities = []
    
    # Simple entity extraction - in production, use spaCy
    # Look for capitalized words (potential entities)
    for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
        entity_name = match.group()
        start_char = match.start()
        end_char = match.end()
        
        # Get context
        context_before = text[max(0, start_char - context_window):start_char]
        context_after = text[end_char:min(len(text), end_char + context_window)]
        
        # Simple confidence based on capitalization pattern
        confidence = 0.8 if len(entity_name.split()) > 1 else 0.6
        
        entities.append(EntityWithContext(
            name=entity_name,
            entity_type="Unknown",  # Would be determined by spaCy
            confidence=confidence,
            start_char=start_char,
            end_char=end_char,
            context_before=context_before,
            context_after=context_after,
            page_number=page_number,
            relationships=[]
        ))
    
    return entities


def find_entity_relationships(
    entities: List[EntityWithContext],
    max_distance: int = 100
) -> List[Tuple[str, str, str]]:
    """
    Find potential relationships between entities based on proximity and context.
    
    Args:
        entities: List of entities with context
        max_distance: Maximum character distance for relationship detection
        
    Returns:
        List of (entity1, entity2, relationship_type) tuples
    """
    relationships = []
    
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities[i+1:], i+1):
            # Check if entities are close enough
            distance = abs(entity1.start_char - entity2.start_char)
            if distance <= max_distance:
                # Determine relationship type based on context
                relationship_type = _determine_relationship_type(
                    entity1, entity2, entity1.context_after + entity2.context_before
                )
                
                if relationship_type:
                    relationships.append((entity1.name, entity2.name, relationship_type))
    
    return relationships


def _determine_relationship_type(
    entity1: EntityWithContext,
    entity2: EntityWithContext,
    context: str
) -> Optional[str]:
    """Determine relationship type based on context words."""
    context_lower = context.lower()
    
    # Relationship patterns
    relationship_patterns = {
        "is": ["is", "was", "are", "were"],
        "has": ["has", "had", "have", "having"],
        "belongs_to": ["belongs to", "part of", "member of"],
        "located_in": ["in", "at", "located in", "found in"],
        "related_to": ["related to", "connected to", "associated with"]
    }
    
    for rel_type, patterns in relationship_patterns.items():
        if any(pattern in context_lower for pattern in patterns):
            return rel_type
    
    return None


def calculate_entity_confidence(
    entity_name: str,
    context: str,
    frequency: int = 1
) -> float:
    """
    Calculate confidence score for an entity based on various factors.
    
    Args:
        entity_name: Name of the entity
        context: Surrounding context
        frequency: How many times entity appears in document
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    confidence = 0.5  # Base confidence
    
    # Boost confidence for longer names (more specific)
    if len(entity_name.split()) > 1:
        confidence += 0.2
    
    # Boost confidence for proper capitalization
    if entity_name.istitle():
        confidence += 0.1
    
    # Boost confidence for frequency
    confidence += min(0.2, frequency * 0.05)
    
    # Boost confidence for context indicators
    context_lower = context.lower()
    if any(word in context_lower for word in ["the", "a", "an"]):
        confidence += 0.1
    
    return min(1.0, confidence)


def deduplicate_entities(entities: List[EntityWithContext]) -> List[EntityWithContext]:
    """
    Remove duplicate entities and merge their information.
    
    Args:
        entities: List of entities with potential duplicates
        
    Returns:
        Deduplicated list of entities
    """
    entity_map = {}
    
    for entity in entities:
        key = entity.name.lower()
        if key in entity_map:
            # Merge information
            existing = entity_map[key]
            existing.confidence = max(existing.confidence, entity.confidence)
            existing.relationships.extend(entity.relationships)
            # Keep the entity with more context
            if len(entity.context_before + entity.context_after) > len(existing.context_before + existing.context_after):
                existing.context_before = entity.context_before
                existing.context_after = entity.context_after
        else:
            entity_map[key] = entity
    
    return list(entity_map.values())
