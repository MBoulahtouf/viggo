"""
Concrete implementation of aliasing service following SOLID principles.
"""

import logging
from collections import defaultdict
from typing import Any

from viggo.core.services.interfaces.aliasing_service import (
    CanonicalGroup,
    IAliasingService,
)


class AliasingService(IAliasingService):
    """
    Service for managing entity aliases and canonical mappings.
    """

    def __init__(self, custom_mappings: dict[str, str] | None = None):
        """
        Initialize the aliasing service.
        
        Args:
            custom_mappings: Optional dictionary of alias -> canonical mappings
        """
        self.alias_to_canonical: dict[str, str] = {}
        self.canonical_to_aliases: dict[str, set[str]] = defaultdict(set)
        self.confidence_scores: dict[str, float] = {}
        self.sources: dict[str, str] = {}

        # Add custom mappings if provided
        if custom_mappings:
            for alias, canonical in custom_mappings.items():
                self.add_alias_mapping(alias, canonical, confidence=1.0, source="manual")

    def add_alias_mapping(self, alias: str, canonical: str, confidence: float = 1.0, source: str = "manual") -> None:
        """
        Add a mapping from alias to canonical name.
        
        Args:
            alias: The alias/nickname
            canonical: The canonical name
            confidence: Confidence score (0.0 to 1.0)
            source: Source of the mapping ("manual", "automatic", "inferred")
        """
        alias_lower = alias.lower().strip()
        canonical_lower = canonical.lower().strip()

        if alias_lower == canonical_lower:
            return  # Don't map to self

        self.alias_to_canonical[alias_lower] = canonical_lower
        self.canonical_to_aliases[canonical_lower].add(alias_lower)
        self.confidence_scores[alias_lower] = confidence
        self.sources[alias_lower] = source

        logging.info(f"Added alias mapping: '{alias}' -> '{canonical}' (confidence: {confidence}, source: {source})")

    def resolve_to_canonical(self, entity_name: str) -> str:
        """
        Resolve an entity name to its canonical form.
        
        Args:
            entity_name: The entity name to resolve
            
        Returns:
            The canonical name, or the original name if no mapping exists
        """
        entity_lower = entity_name.lower().strip()
        return self.alias_to_canonical.get(entity_lower, entity_name)

    def get_all_aliases(self, canonical_name: str) -> set[str]:
        """
        Get all aliases for a canonical name.
        
        Args:
            canonical_name: The canonical name
            
        Returns:
            Set of all aliases (including the canonical name itself)
        """
        canonical_lower = canonical_name.lower().strip()
        aliases = self.canonical_to_aliases.get(canonical_lower, set())
        aliases.add(canonical_lower)  # Include the canonical name itself
        return aliases

    def group_entities_with_aliases(self, entities: list[dict[str, Any]]) -> list[CanonicalGroup]:
        """
        Group entities by canonical name, incorporating alias mappings.
        
        Args:
            entities: List of entity dictionaries with 'name' and 'labels' keys
            
        Returns:
            List of CanonicalGroup objects
        """
        grouped = defaultdict(lambda: {
            "aliases": set(),
            "labels": set(),
            "confidence_scores": {},
            "sources": {}
        })

        for entity in entities:
            if not entity.get("name"):
                continue

            original_name = entity["name"]
            canonical_name = self.resolve_to_canonical(original_name)

            # Add to the canonical group
            group = grouped[canonical_name]
            group["aliases"].add(original_name)
            group["labels"].update(entity.get("labels", []))

            # Track confidence and source for this alias
            alias_lower = original_name.lower().strip()
            if alias_lower in self.confidence_scores:
                group["confidence_scores"][original_name] = self.confidence_scores[alias_lower]
                group["sources"][original_name] = self.sources[alias_lower]
            else:
                group["confidence_scores"][original_name] = 1.0  # Default confidence
                group["sources"][original_name] = "direct"

        # Convert to CanonicalGroup objects
        result = []
        for canonical, group_data in grouped.items():
            result.append(CanonicalGroup(
                canonical=canonical,
                aliases=group_data["aliases"],
                labels=group_data["labels"],
                confidence_scores=group_data["confidence_scores"],
                sources=group_data["sources"]
            ))

        return result

    def suggest_aliases(self, entity_name: str, all_entities: list[dict[str, Any]]) -> list[str]:
        """
        Suggest potential aliases for an entity based on similarity.
        
        Args:
            entity_name: The entity to find aliases for
            all_entities: All available entities
            
        Returns:
            List of potential aliases
        """
        suggestions = []
        entity_lower = entity_name.lower().strip()

        for entity in all_entities:
            if not entity.get("name"):
                continue

            other_name = entity["name"]
            other_lower = other_name.lower().strip()

            # Skip if it's the same entity
            if other_lower == entity_lower:
                continue

            # Simple similarity checks
            if self._names_are_similar(entity_lower, other_lower):
                suggestions.append(other_name)

        return suggestions

    def _names_are_similar(self, name1: str, name2: str) -> bool:
        """
        Check if two names are similar enough to be potential aliases.
        
        Args:
            name1: First name (lowercase)
            name2: Second name (lowercase)
            
        Returns:
            True if names are similar
        """
        # Check for common patterns
        if name1 in name2 or name2 in name1:
            return True

        # Check for shared words (for multi-word names)
        words1 = set(name1.split())
        words2 = set(name2.split())

        if len(words1) > 1 and len(words2) > 1:
            shared_words = words1.intersection(words2)
            if len(shared_words) >= min(len(words1), len(words2)) * 0.5:
                return True

        return False

    def export_mappings(self) -> dict[str, Any]:
        """
        Export all alias mappings for persistence.
        
        Returns:
            Dictionary containing all mappings
        """
        return {
            "alias_to_canonical": self.alias_to_canonical,
            "canonical_to_aliases": {k: list(v) for k, v in self.canonical_to_aliases.items()},
            "confidence_scores": self.confidence_scores,
            "sources": self.sources
        }

    def import_mappings(self, mappings: dict[str, Any]) -> None:
        """
        Import alias mappings from a dictionary.
        
        Args:
            mappings: Dictionary containing mappings to import
        """
        self.alias_to_canonical = mappings.get("alias_to_canonical", {})
        self.canonical_to_aliases = defaultdict(set)
        for canonical, aliases in mappings.get("canonical_to_aliases", {}).items():
            self.canonical_to_aliases[canonical] = set(aliases)
        self.confidence_scores = mappings.get("confidence_scores", {})
        self.sources = mappings.get("sources", {})
