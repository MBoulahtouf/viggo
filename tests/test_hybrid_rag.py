#!/usr/bin/env python3
"""
Test script for hybrid RAG implementation with Azure Cognitive Search.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from viggo.core.services.graph_service import GraphService
from viggo.core.services.rag_service import RAGService
from viggo.core.config import settings


def test_hybrid_rag():
    """Test the hybrid RAG implementation."""
    print("🚀 Testing Hybrid RAG Implementation")
    print("=" * 50)
    
    # Initialize services
    print("📡 Initializing services...")
    try:
        graph_service = GraphService(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        rag_service = RAGService(graph_service)
        print("✅ Services initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        return False
    
    # Test document processing and indexing
    print("\n📚 Testing document processing...")
    test_documents = [
        "data/The_Name_of_the_Rose_Umberto_Eco.pdf",
        "data/The_Strange_High_House_in_the_Mist_HP_Lovecraft.pdf"
    ]
    
    for doc_path in test_documents:
        if os.path.exists(doc_path):
            print(f"📖 Processing: {doc_path}")
            try:
                # Test document indexing in Azure Cognitive Search
                success = rag_service.index_document_in_azure_search(doc_path)
                if success:
                    print(f"✅ Successfully indexed {doc_path}")
                else:
                    print(f"❌ Failed to index {doc_path}")
            except Exception as e:
                print(f"❌ Error processing {doc_path}: {e}")
        else:
            print(f"⚠️  Document not found: {doc_path}")
    
    # Test hybrid search
    print("\n🔍 Testing hybrid search...")
    test_queries = [
        "What is the main character's name?",
        "Where does the story take place?",
        "What is the central mystery?",
        "Who are the key characters?"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        try:
            result = rag_service.perform_rag_query(query)
            print(f"🔍 Search method: {result.get('search_method', 'unknown')}")
            print(f"📄 Source pages: {result.get('source_pages', [])}")
            print(f"💬 Answer: {result.get('answer', 'No answer')[:200]}...")
        except Exception as e:
            print(f"❌ Error with query '{query}': {e}")
    
    # Test Azure Cognitive Search stats
    print("\n📊 Testing Azure Cognitive Search stats...")
    try:
        stats = rag_service.hybrid_search_service.get_index_stats()
        print(f"📈 Index stats: {stats}")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
    
    print("\n🎉 Hybrid RAG test completed!")
    return True


if __name__ == "__main__":
    test_hybrid_rag()
