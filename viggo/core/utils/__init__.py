"""
Core utilities for Viggo hybrid RAG system.

This module provides utility functions for:
- Entity normalization and label mapping
- Hybrid search operations and result combination
- File operations and backup management
- Performance monitoring and optimization
- Query preprocessing and analysis
"""

from .entity_utils import (
    EntityWithContext,
    add_custom_label_mapping,
    calculate_entity_confidence,
    deduplicate_entities,
    extract_entities_with_context,
    filter_and_map_entities,
    find_entity_relationships,
    get_allowed_labels,
    get_entity_label_map,
    normalize_entity_name,
    remove_label_mapping,
)
from .file_ops import (
    backup_rag_data,
    cleanup_old_backups,
    clear_indexes_and_graph,
    get_system_disk_usage,
    monitor_file_changes,
    restore_rag_data,
)
from .hybrid_search_utils import (
    HybridSearchResult,
    HybridSearchUtils,
    PerformanceMonitor,
    QueryPreprocessor,
    RetrievalResult,
)

__all__ = [
    # Entity utilities
    "normalize_entity_name",
    "filter_and_map_entities",
    "get_entity_label_map",
    "get_allowed_labels",
    "add_custom_label_mapping",
    "remove_label_mapping",
    "extract_entities_with_context",
    "find_entity_relationships",
    "calculate_entity_confidence",
    "deduplicate_entities",
    "EntityWithContext",

    # Hybrid search utilities
    "HybridSearchUtils",
    "QueryPreprocessor",
    "PerformanceMonitor",
    "RetrievalResult",
    "HybridSearchResult",

    # File operations
    "clear_indexes_and_graph",
    "backup_rag_data",
    "restore_rag_data",
    "get_system_disk_usage",
    "cleanup_old_backups",
    "monitor_file_changes"
]
