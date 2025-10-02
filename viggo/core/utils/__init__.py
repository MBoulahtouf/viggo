"""
Core utilities for Viggo.

This module provides utility functions for:
- Entity normalization and label mapping
- File operations
- Other common operations
"""

from .entity_utils import (
    normalize_entity_name,
    filter_and_map_entities,
    get_entity_label_map,
    get_allowed_labels,
    add_custom_label_mapping,
    remove_label_mapping
)

__all__ = [
    "normalize_entity_name",
    "filter_and_map_entities", 
    "get_entity_label_map",
    "get_allowed_labels",
    "add_custom_label_mapping",
    "remove_label_mapping"
]
