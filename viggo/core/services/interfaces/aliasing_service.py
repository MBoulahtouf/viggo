"""
Aliasing service interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass


@dataclass
class AliasMapping:
    """Represents a mapping from alias to canonical name."""
    alias: str
    canonical: str
    confidence: float = 1.0  # 0.0 to 1.0, where 1.0 is certain
    source: str = "manual"  # "manual", "automatic", "inferred"


@dataclass
class CanonicalGroup:
    """Represents a group of aliases for a canonical entity."""
    canonical: str
    aliases: Set[str]
    labels: Set[str]
    confidence_scores: Dict[str, float]
    sources: Dict[str, str]


class IAliasingService(ABC):
    """Interface for entity aliasing operations."""
    
    @abstractmethod
    def add_alias_mapping(self, alias: str, canonical: str, confidence: float = 1.0, source: str = "manual") -> None:
        """Add a mapping from alias to canonical name."""
        pass
    
    @abstractmethod
    def resolve_to_canonical(self, entity_name: str) -> str:
        """Resolve an entity name to its canonical form."""
        pass
    
    @abstractmethod
    def get_all_aliases(self, canonical_name: str) -> Set[str]:
        """Get all aliases for a canonical name."""
        pass
    
    @abstractmethod
    def group_entities_with_aliases(self, entities: List[Dict[str, Any]]) -> List[CanonicalGroup]:
        """Group entities by canonical name, incorporating alias mappings."""
        pass
    
    @abstractmethod
    def suggest_aliases(self, entity_name: str, all_entities: List[Dict[str, Any]]) -> List[str]:
        """Suggest potential aliases for an entity based on similarity."""
        pass
    
    @abstractmethod
    def export_mappings(self) -> Dict[str, Any]:
        """Export all alias mappings for persistence."""
        pass
    
    @abstractmethod
    def import_mappings(self, mappings: Dict[str, Any]) -> None:
        """Import alias mappings from a dictionary."""
        pass
