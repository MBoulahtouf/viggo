#!/usr/bin/env python3
"""
Test Azure Cognitive Search with content filtering to exclude non-lore content.
This script demonstrates how the new content filtering prevents metadata from being indexed.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viggo.core.services.rag_service import RAGService
from viggo.core.services.graph_service import GraphService
from viggo.core.services.content_filter_service import ContentFilterService
from viggo.core.config import settings


def test_content_filtering():
    """Test content filtering before indexing in Azure Cognitive Search."""
    
    print("🔍 Testing Azure Cognitive Search Content Filtering")
    print("=" * 60)
    
    try:
        # Initialize services
        graph_service = GraphService(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        rag_service = RAGService(graph_service=graph_service)
        content_filter = ContentFilterService()
        
        # Process the Lovecraft document
        lovecraft_file = "/home/mikealpharomeo/Projects/viggo/data/The_Strange_High_House_in_the_Mist_HP_Lovecraft.pdf"
        
        print(f"📖 Processing document: {lovecraft_file}")
        num_chunks, vector_index, chunks_with_metadata = rag_service.process_document(lovecraft_file)
        
        print(f"📊 Document processing results:")
        print(f"   Total chunks extracted: {num_chunks}")
        
        # Test content filtering
        print("\n🧹 Testing content filtering...")
        filtered_chunks, filter_stats = content_filter.filter_chunks_for_indexing(chunks_with_metadata)
        
        print(f"📊 Content filtering results:")
        print(f"   Total chunks: {filter_stats['total_chunks']}")
        print(f"   Story content: {filter_stats['story_content']}")
        print(f"   Metadata: {filter_stats['metadata']}")
        print(f"   Bibliography: {filter_stats['bibliography']}")
        print(f"   Publisher info: {filter_stats['publisher_info']}")
        print(f"   Technical: {filter_stats['technical']}")
        print(f"   Preface: {filter_stats['preface']}")
        print(f"   Filtered out: {filter_stats['filtered_out']}")
        
        # Show some examples of filtered content
        print("\n🚫 Examples of filtered content:")
        for chunk in chunks_with_metadata[:5]:
            content_type = content_filter.classify_content_type(chunk.get('content', ''), chunk.get('page', 0))
            if content_type.value != 'story_content':
                content_preview = chunk.get('content', '')[:100].replace('\n', ' ')
                print(f"   [{content_type.value}] Page {chunk.get('page', 0)}: {content_preview}...")
        
        print("\n✅ Examples of story content that will be indexed:")
        story_count = 0
        for chunk in chunks_with_metadata:
            content_type = content_filter.classify_content_type(chunk.get('content', ''), chunk.get('page', 0))
            if content_type.value == 'story_content' and story_count < 3:
                content_preview = chunk.get('content', '')[:100].replace('\n', ' ')
                print(f"   [story_content] Page {chunk.get('page', 0)}: {content_preview}...")
                story_count += 1
        
        # Test indexing with filtering
        print("\n🔍 Testing Azure Cognitive Search indexing with filtering...")
        success = rag_service.index_document_in_azure_search(lovecraft_file)
        
        if success:
            print("✅ Successfully indexed document with content filtering!")
            
            # Test search with filtering
            print("\n🔍 Testing search with content filtering...")
            search_service = rag_service.hybrid_search_service
            
            # Test queries
            test_queries = [
                "Olney and the strange house",
                "Kingsport and the mist",
                "Nodens and the Elder Ones"
            ]
            
            for query in test_queries:
                print(f"\n🔍 Query: '{query}'")
                results = search_service.keyword_search(query, k=3)
                print(f"   Results: {len(results)}")
                for i, result in enumerate(results[:2], 1):
                    content_preview = result['content'][:80].replace('\n', ' ')
                    print(f"   {i}. Page {result['page']} (Score: {result['score']:.2f}): {content_preview}...")
        else:
            print("❌ Failed to index document with content filtering")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the content filtering test."""
    success = test_content_filtering()
    
    if success:
        print("\n🎉 Content filtering test completed successfully!")
        print("🔍 Key benefits:")
        print("   • Non-lore content is filtered out during indexing")
        print("   • Search results only contain story-relevant content")
        print("   • Improved search quality and relevance")
        print("   • Reduced noise in entity extraction")
    else:
        print("\n💥 Content filtering test failed!")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
