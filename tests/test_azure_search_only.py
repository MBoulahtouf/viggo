#!/usr/bin/env python3
"""
Test script for Azure Cognitive Search functionality only.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from viggo.core.services.hybrid_search_service import HybridSearchService
from viggo.core.config import settings


def test_azure_search():
    """Test Azure Cognitive Search functionality."""
    print("🚀 Testing Azure Cognitive Search")
    print("=" * 50)
    
    # Initialize hybrid search service
    print("📡 Initializing Azure Cognitive Search service...")
    try:
        search_service = HybridSearchService()
        print("✅ Azure Cognitive Search service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Azure Cognitive Search service: {e}")
        return False
    
    # Test index creation
    print("\n📚 Testing index creation...")
    try:
        success = search_service.create_index()
        if success:
            print("✅ Index created successfully")
        else:
            print("⚠️  Index creation failed (might already exist)")
    except Exception as e:
        print(f"❌ Error creating index: {e}")
    
    # Test index stats
    print("\n📊 Testing index stats...")
    try:
        stats = search_service.get_index_stats()
        print(f"📈 Index stats: {stats}")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
    
    # Test with sample documents
    print("\n📖 Testing document indexing...")
    sample_docs = [
        {
            "content": "The Name of the Rose is a novel by Umberto Eco about a Franciscan friar investigating mysterious deaths in a medieval monastery.",
            "page": 1,
            "word_count": 20,
            "char_count": 120,
            "entities": ["Umberto Eco", "Franciscan friar", "monastery"],
            "entity_labels": ["PERSON", "PERSON", "LOCATION"],
            "chapter_title": "Prologue",
            "chunk_type": "standard",
            "document_metadata": {"title": "The Name of the Rose", "author": "Umberto Eco"}
        },
        {
            "content": "The Strange High House in the Mist is a short story by H.P. Lovecraft about a mysterious house on a cliff overlooking the sea.",
            "page": 1,
            "word_count": 25,
            "char_count": 150,
            "entities": ["H.P. Lovecraft", "house", "cliff", "sea"],
            "entity_labels": ["PERSON", "LOCATION", "LOCATION", "LOCATION"],
            "chapter_title": "The House",
            "chunk_type": "standard",
            "document_metadata": {"title": "The Strange High House in the Mist", "author": "H.P. Lovecraft"}
        }
    ]
    
    try:
        success = search_service.index_documents(sample_docs)
        if success:
            print("✅ Sample documents indexed successfully")
        else:
            print("❌ Failed to index sample documents")
    except Exception as e:
        print(f"❌ Error indexing documents: {e}")
    
    # Test search functionality
    print("\n🔍 Testing search functionality...")
    test_queries = [
        "What is the Name of the Rose about?",
        "Who wrote the story about the house?",
        "monastery",
        "Lovecraft"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        try:
            # Test hybrid search
            hybrid_results = search_service.hybrid_search(query, k=3)
            print(f"🔍 Hybrid search results: {len(hybrid_results)} found")
            for i, result in enumerate(hybrid_results):
                print(f"  {i+1}. Score: {result.get('score', 0):.3f} - {result['content'][:100]}...")
            
            # Test keyword search
            keyword_results = search_service.keyword_search(query, k=3)
            print(f"🔤 Keyword search results: {len(keyword_results)} found")
            
            # Test semantic search
            semantic_results = search_service.semantic_search(query, k=3)
            print(f"🧠 Semantic search results: {len(semantic_results)} found")
            
        except Exception as e:
            print(f"❌ Error with query '{query}': {e}")
    
    print("\n🎉 Azure Cognitive Search test completed!")
    return True


if __name__ == "__main__":
    test_azure_search()
