#!/usr/bin/env python3
"""
Simple test for Azure Cognitive Search without vector search.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchableField
from azure.core.credentials import AzureKeyCredential
from viggo.core.config import settings


def test_simple_azure_search():
    """Test basic Azure Cognitive Search functionality."""
    print("🚀 Testing Simple Azure Cognitive Search")
    print("=" * 50)
    
    # Initialize clients
    print("📡 Initializing Azure Cognitive Search clients...")
    try:
        search_client = SearchClient(
            endpoint=settings.elasticsearch_url,
            index_name="viggo-simple",
            credential=AzureKeyCredential(settings.elasticsearch_api_key)
        )
        index_client = SearchIndexClient(
            endpoint=settings.elasticsearch_url,
            credential=AzureKeyCredential(settings.elasticsearch_api_key)
        )
        print("✅ Azure Cognitive Search clients initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize clients: {e}")
        return False
    
    # Create a simple index
    print("\n📚 Creating simple index...")
    try:
        # Check if index exists
        try:
            existing_index = index_client.get_index("viggo-simple")
            print("✅ Index 'viggo-simple' already exists")
        except Exception:
            # Create simple index
            fields = [
                SimpleField(name="id", type="Edm.String", key=True),
                SearchableField(name="content", type="Edm.String"),
                SimpleField(name="page", type="Edm.Int32"),
                SimpleField(name="word_count", type="Edm.Int32")
            ]
            
            index = SearchIndex(name="viggo-simple", fields=fields)
            index_client.create_index(index)
            print("✅ Simple index created successfully")
    except Exception as e:
        print(f"❌ Error with index: {e}")
        return False
    
    # Index sample documents
    print("\n📖 Indexing sample documents...")
    try:
        sample_docs = [
            {
                "id": "1",
                "content": "The Name of the Rose is a novel by Umberto Eco about a Franciscan friar investigating mysterious deaths in a medieval monastery.",
                "page": 1,
                "word_count": 20
            },
            {
                "id": "2", 
                "content": "The Strange High House in the Mist is a short story by H.P. Lovecraft about a mysterious house on a cliff overlooking the sea.",
                "page": 1,
                "word_count": 25
            }
        ]
        
        result = search_client.upload_documents(sample_docs)
        failed_docs = [doc for doc in result if not doc.succeeded]
        if failed_docs:
            print(f"❌ Failed to index {len(failed_docs)} documents")
            for doc in failed_docs:
                print(f"Error: {doc.error_message}")
        else:
            print("✅ Sample documents indexed successfully")
    except Exception as e:
        print(f"❌ Error indexing documents: {e}")
        return False
    
    # Test search
    print("\n🔍 Testing search...")
    try:
        results = search_client.search(search_text="monastery")
        print(f"✅ Search for 'monastery' returned {len(list(results))} results")
        
        results = search_client.search(search_text="Lovecraft")
        print(f"✅ Search for 'Lovecraft' returned {len(list(results))} results")
        
    except Exception as e:
        print(f"❌ Error searching: {e}")
        return False
    
    # Get document count
    print("\n📊 Getting document count...")
    try:
        count = search_client.get_document_count()
        print(f"✅ Document count: {count}")
    except Exception as e:
        print(f"❌ Error getting count: {e}")
    
    print("\n🎉 Simple Azure Cognitive Search test completed!")
    return True


if __name__ == "__main__":
    test_simple_azure_search()
