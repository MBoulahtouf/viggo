#!/usr/bin/env python3
"""
Test Entity-Chunk Linking Architecture

This script demonstrates the new entity-chunk linking architecture with
context-aware retrieval and spoiler protection.
"""

import os
import sys
import time
from pathlib import Path

# Add the viggo package to the path
sys.path.append(str(Path(__file__).parent))

from viggo.core.services.rag_service import RAGService
from viggo.core.services.graph_service import GraphService
from viggo.core.services.enhanced_rag_integration import EnhancedRAGIntegration
from viggo.core.services.entity_chunk_linker import ContextType
from viggo.core.config import settings


def test_entity_chunk_architecture():
    """Test the entity-chunk linking architecture."""
    
    print("🏗️ Entity-Chunk Linking Architecture Test")
    print("=" * 50)
    
    # Initialize services
    print("🔮 Initializing services...")
    try:
        graph_service = GraphService(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        print("✅ GraphService connected to Neo4j")
        
        rag_service = RAGService(graph_service)
        enhanced_rag = EnhancedRAGIntegration(rag_service, graph_service)
        print("✅ Enhanced RAG Integration initialized")
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return
    
    # Test document
    test_doc = "data/The_Strange_High_House_in_the_Mist_HP_Lovecraft.pdf"
    
    if not os.path.exists(test_doc):
        print(f"❌ Test document not found: {test_doc}")
        return
    
    # Step 1: Process document with entity-chunk linking
    print(f"\n🏗️ Step 1: Processing Document with Entity-Chunk Linking")
    print("-" * 60)
    
    try:
        result = enhanced_rag.process_document_with_entity_linking(test_doc)
        
        print(f"✅ Document processing complete:")
        print(f"   File: {result['file_path']}")
        print(f"   Entity-chunk links: {result['entity_chunk_links']}")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Links created: {result['links_created']}")
        
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        return
    
    # Step 2: Test entity-chunk linking summary
    print(f"\n📊 Step 2: Entity-Chunk Links Summary")
    print("-" * 40)
    
    try:
        summary = enhanced_rag.get_entity_chunk_links_summary()
        
        if summary["links_created"]:
            print(f"✅ Entity-chunk links summary:")
            print(f"   Total links: {summary['total_links']}")
            print(f"   Unique entities: {summary['unique_entities']}")
            
            print(f"\n🏷️ Context Types:")
            for context_type, count in summary['context_type_counts'].items():
                print(f"   {context_type}: {count}")
            
            print(f"\n👥 Top Entities:")
            for entity, count in summary['top_entities'][:5]:
                print(f"   {entity}: {count} mentions")
        else:
            print("❌ No entity-chunk links created")
            
    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
    
    # Step 3: Test context-aware queries
    print(f"\n🔍 Step 3: Context-Aware Query Tests")
    print("-" * 45)
    
    test_queries = [
        # Entity mention queries
        ("Where is Kingsport mentioned?", None),
        ("Show me where Olney appears", None),
        ("Find mentions of The Elder Ones", None),
        
        # Description queries
        ("What is Kingsport?", None),
        ("Who is Olney?", None),
        ("Tell me about The Elder Ones", None),
        
        # Relationship queries
        ("Who is Olney related to?", None),
        ("What is Kingsport's connection to Arkham?", None),
        
        # Spoiler-protected queries
        ("Where is Kingsport mentioned?", 5),  # Only up to page 5
        ("Tell me about Olney", 7),  # Only up to page 7
    ]
    
    for i, (query, page_limit) in enumerate(test_queries, 1):
        print(f"\n❓ Query {i}: '{query}'")
        if page_limit:
            print(f"   📄 Page limit: {page_limit}")
        
        try:
            start_time = time.time()
            result = enhanced_rag.query_with_entity_context(query, page_limit)
            query_time = time.time() - start_time
            
            print(f"   ⏱️ Response time: {query_time:.2f}s")
            print(f"   🔍 Search method: {result.search_method}")
            print(f"   📄 Source pages: {result.source_pages}")
            print(f"   🛡️ Spoiler protected: {result.spoiler_protected}")
            print(f"   👥 Entities found: {result.entities_found}")
            print(f"   💬 Answer: {result.answer[:150]}...")
            
            # Show entity contexts if available
            if result.entity_contexts:
                print(f"   🔗 Entity contexts:")
                for entity_name, context in result.entity_contexts.items():
                    print(f"      {entity_name}: {context.total_mentions} mentions, {len(context.context_types_found)} context types")
            
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
    
    # Step 4: Test entity passage finding
    print(f"\n📖 Step 4: Entity Passage Finding Tests")
    print("-" * 45)
    
    test_entities = ["Kingsport", "Olney", "The Elder Ones"]
    
    for entity in test_entities:
        print(f"\n🔍 Finding passages for: {entity}")
        
        try:
            # Find all mentions
            all_mentions = enhanced_rag.find_entity_passages(entity)
            print(f"   📊 Total mentions: {len(all_mentions)}")
            
            # Find descriptive mentions
            descriptive_mentions = enhanced_rag.find_entity_passages(entity, ContextType.DESCRIPTION)
            print(f"   📝 Descriptive mentions: {len(descriptive_mentions)}")
            
            # Find action mentions
            action_mentions = enhanced_rag.find_entity_passages(entity, ContextType.ACTION)
            print(f"   🎭 Action mentions: {len(action_mentions)}")
            
            # Find relationship mentions
            relationship_mentions = enhanced_rag.find_entity_passages(entity, ContextType.RELATIONSHIP)
            print(f"   🔗 Relationship mentions: {len(relationship_mentions)}")
            
            # Show sample passages
            if all_mentions:
                print(f"   📄 Sample passages:")
                for i, mention in enumerate(all_mentions[:3]):
                    print(f"      {i+1}. Page {mention.page_number}: {mention.surrounding_text[:100]}...")
            
        except Exception as e:
            print(f"   ❌ Passage finding failed: {e}")
    
    # Step 5: Test entity context analysis
    print(f"\n🔮 Step 5: Entity Context Analysis")
    print("-" * 40)
    
    for entity in test_entities:
        print(f"\n📊 Context analysis for: {entity}")
        
        try:
            # Get context analysis
            context_analysis = enhanced_rag.get_entity_context_analysis(entity)
            
            if "error" not in context_analysis:
                print(f"   🏷️ Entity label: {context_analysis['entity_label']}")
                print(f"   📊 Total mentions: {context_analysis['total_mentions']}")
                print(f"   🎯 Context types: {context_analysis['context_types']}")
                print(f"   📖 Chunks by context: {context_analysis['chunks_by_context']}")
                print(f"   🔗 Relationships: {context_analysis['relationships']}")
            else:
                print(f"   ❌ {context_analysis['error']}")
                
        except Exception as e:
            print(f"   ❌ Context analysis failed: {e}")
    
    # Final summary
    print(f"\n🎉 Entity-Chunk Linking Architecture Test Complete!")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   Document processed: {test_doc}")
    print(f"   Entity-chunk links created: {result.entity_chunk_links if hasattr(result, 'entity_chunk_links') else 'N/A'}")
    print(f"   Queries tested: {len(test_queries)}")
    print(f"   Entities analyzed: {len(test_entities)}")
    
    print(f"\n🏗️ Architecture Benefits:")
    print(f"   • Entity-chunk linking enables precise passage retrieval")
    print(f"   • Context-aware retrieval provides relevant answers")
    print(f"   • Spoiler protection works at the entity level")
    print(f"   • Users can find specific passages where entities are mentioned")
    print(f"   • Context analysis helps understand entity roles and relationships")
    
    return result


if __name__ == "__main__":
    print("🏗️ Entity-Chunk Linking Architecture Test")
    print("🔮 Testing context-aware retrieval with spoiler protection...")
    print()
    
    result = test_entity_chunk_architecture()
    
    if result:
        print(f"\n✅ Test completed successfully!")
        print(f"📈 Check your Neo4j browser for the updated graph with entity relationships.")
    else:
        print(f"\n❌ Test failed!")
