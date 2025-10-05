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
    normalize_entity_name,
    filter_and_map_entities,
    get_entity_label_map,
    get_allowed_labels,
    add_custom_label_mapping,
    remove_label_mapping,
    extract_entities_with_context,
    find_entity_relationships,
    calculate_entity_confidence,
    deduplicate_entities,
    EntityWithContext
)

from .hybrid_search_utils import (
    HybridSearchUtils,
    QueryPreprocessor,
    PerformanceMonitor,
    RetrievalResult,
    HybridSearchResult
)

from .file_ops import (
    clear_indexes_and_graph,
    backup_rag_data,
    restore_rag_data,
    get_system_disk_usage,
    cleanup_old_backups,
    monitor_file_changes
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
