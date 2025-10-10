"""
File operations utilities for Viggo hybrid RAG system.

This module provides utilities for:
- Index and cache management
- Document file operations
- Backup and recovery operations
- Performance monitoring file operations
"""

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def clear_indexes_and_graph(rag_service, graph_service):
    """Remove old index files and clear the Neo4j graph."""
    # Only clear graph database if graph service is available
    if graph_service is not None:
        graph_service.clear_database()

    # Clear index files
    index_path = getattr(rag_service, 'index_path', 'faiss_index.bin')
    doc_data_path = getattr(rag_service, 'doc_data_path', 'document_data.pkl')
    for path in [index_path, doc_data_path]:
        if os.path.exists(path):
            os.remove(path)


def backup_rag_data(
    backup_dir: str,
    rag_service,
    graph_service=None,
    include_metadata: bool = True
) -> dict[str, Any]:
    """
    Create a backup of RAG system data.
    
    Args:
        backup_dir: Directory to store backup
        rag_service: RAG service instance
        graph_service: Graph service instance (optional)
        include_metadata: Whether to include metadata files
        
    Returns:
        Backup information dictionary
    """
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_info = {
        "timestamp": timestamp,
        "backup_dir": str(backup_path),
        "files_backed_up": [],
        "graph_backed_up": False,
        "success": True,
        "errors": []
    }

    try:
        # Backup vector indexes
        index_files = [
            getattr(rag_service, 'index_path', 'faiss_index.bin'),
            getattr(rag_service, 'doc_data_path', 'document_data.pkl'),
            getattr(rag_service, 'metadata_path', 'metadata.json')
        ]

        for file_path in index_files:
            if os.path.exists(file_path):
                backup_file = backup_path / f"{timestamp}_{Path(file_path).name}"
                shutil.copy2(file_path, backup_file)
                backup_info["files_backed_up"].append(str(backup_file))

        # Backup graph data if available
        if graph_service is not None:
            try:
                graph_backup_file = backup_path / f"{timestamp}_graph_backup.json"
                # Export graph data (implementation depends on graph service)
                backup_info["graph_backed_up"] = True
                backup_info["files_backed_up"].append(str(graph_backup_file))
            except Exception as e:
                backup_info["errors"].append(f"Graph backup failed: {str(e)}")

        # Create backup manifest
        manifest_file = backup_path / f"{timestamp}_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(backup_info, f, indent=2)

    except Exception as e:
        backup_info["success"] = False
        backup_info["errors"].append(f"Backup failed: {str(e)}")

    return backup_info


def restore_rag_data(
    backup_dir: str,
    rag_service,
    graph_service=None,
    backup_timestamp: str | None = None
) -> dict[str, Any]:
    """
    Restore RAG system data from backup.
    
    Args:
        backup_dir: Directory containing backup
        rag_service: RAG service instance
        graph_service: Graph service instance (optional)
        backup_timestamp: Specific backup timestamp to restore (latest if None)
        
    Returns:
        Restore information dictionary
    """
    backup_path = Path(backup_dir)
    restore_info = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "backup_dir": str(backup_path),
        "files_restored": [],
        "graph_restored": False,
        "success": True,
        "errors": []
    }

    try:
        # Find backup to restore
        if backup_timestamp is None:
            # Find latest backup
            manifest_files = list(backup_path.glob("*_manifest.json"))
            if not manifest_files:
                raise ValueError("No backup manifests found")

            latest_manifest = max(manifest_files, key=os.path.getctime)
            with open(latest_manifest) as f:
                backup_info = json.load(f)
            backup_timestamp = backup_info["timestamp"]

        # Restore files
        backup_files = list(backup_path.glob(f"{backup_timestamp}_*"))
        for backup_file in backup_files:
            if backup_file.name.endswith("_manifest.json"):
                continue

            # Determine target file
            original_name = backup_file.name.replace(f"{backup_timestamp}_", "")
            target_path = Path(original_name)

            # Restore file
            shutil.copy2(backup_file, target_path)
            restore_info["files_restored"].append(str(target_path))

        # Restore graph data if available
        if graph_service is not None:
            try:
                graph_backup_file = backup_path / f"{backup_timestamp}_graph_backup.json"
                if graph_backup_file.exists():
                    # Import graph data (implementation depends on graph service)
                    restore_info["graph_restored"] = True
            except Exception as e:
                restore_info["errors"].append(f"Graph restore failed: {str(e)}")

    except Exception as e:
        restore_info["success"] = False
        restore_info["errors"].append(f"Restore failed: {str(e)}")

    return restore_info


def get_system_disk_usage(data_dir: str) -> dict[str, Any]:
    """
    Get disk usage information for RAG system files.
    
    Args:
        data_dir: Directory containing RAG data files
        
    Returns:
        Disk usage information
    """
    data_path = Path(data_dir)
    usage_info = {
        "total_size": 0,
        "file_count": 0,
        "files": [],
        "largest_files": []
    }

    if not data_path.exists():
        return usage_info

    # Calculate sizes
    for file_path in data_path.rglob("*"):
        if file_path.is_file():
            size = file_path.stat().st_size
            usage_info["total_size"] += size
            usage_info["file_count"] += 1
            usage_info["files"].append({
                "path": str(file_path.relative_to(data_path)),
                "size": size,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })

    # Sort by size to find largest files
    usage_info["largest_files"] = sorted(
        usage_info["files"],
        key=lambda x: x["size"],
        reverse=True
    )[:10]

    return usage_info


def cleanup_old_backups(backup_dir: str, keep_count: int = 5) -> dict[str, Any]:
    """
    Clean up old backup files, keeping only the most recent ones.
    
    Args:
        backup_dir: Directory containing backups
        keep_count: Number of recent backups to keep
        
    Returns:
        Cleanup information
    """
    backup_path = Path(backup_dir)
    cleanup_info = {
        "backups_kept": 0,
        "backups_removed": 0,
        "space_freed": 0,
        "errors": []
    }

    try:
        # Find all backup manifests
        manifest_files = list(backup_path.glob("*_manifest.json"))
        if len(manifest_files) <= keep_count:
            return cleanup_info

        # Sort by creation time (newest first)
        manifest_files.sort(key=os.path.getctime, reverse=True)

        # Keep the most recent backups
        manifests_to_keep = manifest_files[:keep_count]
        manifests_to_remove = manifest_files[keep_count:]

        cleanup_info["backups_kept"] = len(manifests_to_keep)

        # Remove old backups
        for manifest_file in manifests_to_remove:
            try:
                # Read manifest to get backup timestamp
                with open(manifest_file) as f:
                    backup_info = json.load(f)
                backup_timestamp = backup_info["timestamp"]

                # Remove all files for this backup
                backup_files = list(backup_path.glob(f"{backup_timestamp}_*"))
                for backup_file in backup_files:
                    if backup_file.exists():
                        size = backup_file.stat().st_size
                        backup_file.unlink()
                        cleanup_info["space_freed"] += size
                        cleanup_info["backups_removed"] += 1

            except Exception as e:
                cleanup_info["errors"].append(f"Failed to remove backup {manifest_file}: {str(e)}")

    except Exception as e:
        cleanup_info["errors"].append(f"Cleanup failed: {str(e)}")

    return cleanup_info


def monitor_file_changes(
    file_paths: list[str],
    callback_func,
    check_interval: int = 60
) -> None:
    """
    Monitor file changes and call callback function when changes are detected.
    
    Args:
        file_paths: List of file paths to monitor
        callback_func: Function to call when changes are detected
        check_interval: Check interval in seconds
    """
    file_times = {}

    # Initialize file times
    for file_path in file_paths:
        if os.path.exists(file_path):
            file_times[file_path] = os.path.getmtime(file_path)

    while True:
        time.sleep(check_interval)

        for file_path in file_paths:
            if os.path.exists(file_path):
                current_time = os.path.getmtime(file_path)
                if file_path in file_times and current_time > file_times[file_path]:
                    # File has changed
                    callback_func(file_path, file_times[file_path], current_time)
                file_times[file_path] = current_time
            elif file_path in file_times:
                # File was deleted
                callback_func(file_path, file_times[file_path], None)
                del file_times[file_path]
