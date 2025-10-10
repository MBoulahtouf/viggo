#!/usr/bin/env python3
"""
Migration script to move from FAISS to Elasticsearch for Viggo RAG system.
This script helps migrate existing FAISS data to Elasticsearch.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from viggo.core.services.implementations.storage_impl import ElasticsearchVectorStorage
from elasticsearch import Elasticsearch


def check_elasticsearch_connection() -> bool:
    """Check if Elasticsearch is running and accessible."""
    try:
        es = Elasticsearch(hosts=["localhost:9200"])
        if es.ping():
            print("✅ Elasticsearch is running and accessible")
            return True
        else:
            print("❌ Elasticsearch is not responding")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Elasticsearch: {e}")
        return False


def create_elasticsearch_storage() -> ElasticsearchVectorStorage:
    """Create and initialize Elasticsearch vector storage."""
    try:
        storage = ElasticsearchVectorStorage()
        print("✅ Elasticsearch vector storage initialized")
        return storage
    except Exception as e:
        print(f"❌ Failed to initialize Elasticsearch storage: {e}")
        return None


def migrate_sample_data(storage: ElasticsearchVectorStorage) -> bool:
    """Migrate sample data to test the setup."""
    try:
        # Sample vectors (384-dimensional for all-MiniLM-L6-v2)
        sample_vectors = [
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384
        ]
        
        sample_metadata = [
            {
                "content": "This is a sample document about artificial intelligence and machine learning.",
                "page": 1,
                "chunk_id": "sample_chunk_1",
                "entities": ["AI", "machine learning"],
                "entity_labels": ["Technology", "Technology"],
                "chapter_title": "Introduction to AI",
                "chunk_type": "standard",
                "lore_significance": 0.8,
                "word_count": 12,
                "char_count": 85
            },
            {
                "content": "Neural networks are computational models inspired by biological neural networks.",
                "page": 2,
                "chunk_id": "sample_chunk_2",
                "entities": ["neural networks", "biological"],
                "entity_labels": ["Technology", "Biology"],
                "chapter_title": "Neural Networks",
                "chunk_type": "technical",
                "lore_significance": 0.9,
                "word_count": 11,
                "char_count": 78
            },
            {
                "content": "Deep learning has revolutionized many fields including computer vision and NLP.",
                "page": 3,
                "chunk_id": "sample_chunk_3",
                "entities": ["deep learning", "computer vision", "NLP"],
                "entity_labels": ["Technology", "Technology", "Technology"],
                "chapter_title": "Deep Learning Applications",
                "chunk_type": "application",
                "lore_significance": 0.7,
                "word_count": 13,
                "char_count": 89
            }
        ]
        
        success = storage.add_vectors(sample_vectors, sample_metadata)
        if success:
            print("✅ Sample data migrated successfully")
            return True
        else:
            print("❌ Failed to migrate sample data")
            return False
            
    except Exception as e:
        print(f"❌ Error migrating sample data: {e}")
        return False


def test_search_functionality(storage: ElasticsearchVectorStorage) -> bool:
    """Test search functionality with sample queries."""
    try:
        # Test vector search
        query_vector = [0.15] * 384  # Similar to first sample
        results = storage.search_vectors(query_vector, top_k=3)
        
        if results:
            print(f"✅ Search returned {len(results)} results")
            for i, result in enumerate(results):
                print(f"   Result {i+1}: {result['content'][:50]}... (score: {result['score']:.3f})")
            return True
        else:
            print("❌ Search returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Error testing search: {e}")
        return False


def get_vector_count(storage: ElasticsearchVectorStorage) -> int:
    """Get the number of vectors in storage."""
    try:
        count = storage.get_vector_count()
        print(f"📊 Total vectors in storage: {count}")
        return count
    except Exception as e:
        print(f"❌ Error getting vector count: {e}")
        return 0


def main():
    """Main migration function."""
    print("🚀 Starting FAISS to Elasticsearch Migration")
    print("=" * 50)
    
    # Step 1: Check Elasticsearch connection
    if not check_elasticsearch_connection():
        print("\n💡 To start Elasticsearch, run:")
        print("   docker-compose -f docker-compose.elasticsearch.yml up -d")
        return False
    
    # Step 2: Create Elasticsearch storage
    storage = create_elasticsearch_storage()
    if not storage:
        return False
    
    # Step 3: Get initial vector count
    initial_count = get_vector_count(storage)
    
    # Step 4: Migrate sample data
    if not migrate_sample_data(storage):
        return False
    
    # Step 5: Verify migration
    final_count = get_vector_count(storage)
    if final_count > initial_count:
        print("✅ Data migration successful")
    else:
        print("⚠️ No new data was added")
    
    # Step 6: Test search functionality
    if not test_search_functionality(storage):
        return False
    
    print("\n🎉 Migration completed successfully!")
    print("\nNext steps:")
    print("1. Update your application configuration to use ElasticsearchVectorStorage")
    print("2. Remove FAISS dependencies from pyproject.toml")
    print("3. Test your application with the new Elasticsearch backend")
    print("4. Monitor performance and adjust Elasticsearch settings as needed")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
