#!/usr/bin/env python3
"""
Test User-Aware RAG Service

This script demonstrates the user progress tracking and spoiler protection system
integrated with the entity-chunk linking architecture.
"""

import os
import sys
import time
from pathlib import Path

# Add the viggo package to the path
sys.path.append(str(Path(__file__).parent))

from viggo.core.services.rag_service import RAGService
from viggo.core.services.graph_service import GraphService
from viggo.core.services.user_progress_service import UserProgressService
from viggo.core.services.user_aware_rag_service import UserAwareRAGService
from viggo.core.config import settings


def test_user_aware_rag():
    """Test the user-aware RAG service with progress tracking."""
    
    print("👤 User-Aware RAG Service Test")
    print("=" * 50)
    print("🔮 Testing user progress tracking and spoiler protection...")
    print()
    
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
        user_progress_service = UserProgressService()
        user_aware_rag = UserAwareRAGService(rag_service, graph_service, user_progress_service)
        print("✅ User-Aware RAG Service initialized")
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return
    
    # Test document
    test_doc = "data/The_Strange_High_House_in_the_Mist_HP_Lovecraft.pdf"
    
    if not os.path.exists(test_doc):
        print(f"❌ Test document not found: {test_doc}")
        return
    
    # Test users with different progress levels
    test_users = [
        {"user_id": "alice", "name": "Alice", "current_page": 5, "finished_book": False},
        {"user_id": "bob", "name": "Bob", "current_page": 8, "finished_book": False},
        {"user_id": "charlie", "name": "Charlie", "current_page": 12, "finished_book": True},
    ]
    
    # Step 1: Process document for each user
    print(f"\n🏗️ Step 1: Processing Document for Multiple Users")
    print("-" * 60)
    
    document_results = {}
    
    for user in test_users:
        print(f"\n👤 Processing for {user['name']} (page {user['current_page']}, finished: {user['finished_book']})")
        
        try:
            result = user_aware_rag.process_document_for_user(
                file_path=test_doc,
                user_id=user['user_id'],
                document_name=f"Lovecraft - {user['name']}'s Copy",
                current_page=user['current_page'],
                finished_book=user['finished_book']
            )
            
            document_results[user['user_id']] = result
            
            print(f"   ✅ Document processed successfully")
            print(f"   📄 Document ID: {result['document_id']}")
            print(f"   📊 Total pages: {result['user_progress'].total_pages}")
            print(f"   🛡️ Spoiler protected: {result['spoiler_protected']}")
            print(f"   📍 Spoiler limit: {result['spoiler_limit']}")
            print(f"   ⏱️ Processing time: {result['total_processing_time']:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
    
    # Step 2: Test queries with different user contexts
    print(f"\n🔍 Step 2: Context-Aware Query Tests")
    print("-" * 50)
    
    test_queries = [
        "Where is Kingsport mentioned?",
        "Who is Olney?",
        "Tell me about The Elder Ones",
        "What happens in the story?",
        "Show me the passage where Olney climbs the cliff"
    ]
    
    for user in test_users:
        print(f"\n👤 Testing queries for {user['name']} (page {user['current_page']})")
        
        if user['user_id'] not in document_results:
            print(f"   ⚠️ Skipping - document not processed")
            continue
        
        document_id = document_results[user['user_id']]['document_id']
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   ❓ Query {i}: '{query}'")
            
            try:
                start_time = time.time()
                result = user_aware_rag.query_with_user_context(query, user['user_id'], document_id)
                query_time = time.time() - start_time
                
                if "error" in result:
                    print(f"      ❌ Error: {result['error']}")
                    continue
                
                rag_result = result['rag_result']
                user_progress = result['user_progress']
                
                print(f"      ⏱️ Response time: {query_time:.2f}s")
                print(f"      🔍 Search method: {rag_result['search_method']}")
                print(f"      📄 Source pages: {rag_result['source_pages']}")
                print(f"      🛡️ Spoiler protected: {rag_result['spoiler_protected']}")
                print(f"      👥 Entities found: {rag_result['entities_found']}")
                print(f"      💬 Answer: {rag_result['answer'][:100]}...")
                
                # Show spoiler protection in action
                if rag_result['spoiler_protected']:
                    print(f"      🛡️ Spoiler limit: {result['spoiler_limit']} (user on page {user_progress['current_page']})")
                
            except Exception as e:
                print(f"      ❌ Query failed: {e}")
    
    # Step 3: Test entity passage finding with user context
    print(f"\n📖 Step 3: Entity Passage Finding with User Context")
    print("-" * 60)
    
    test_entities = ["Kingsport", "Olney", "The Elder Ones"]
    
    for user in test_users:
        print(f"\n👤 Entity passages for {user['name']} (page {user['current_page']})")
        
        if user['user_id'] not in document_results:
            print(f"   ⚠️ Skipping - document not processed")
            continue
        
        document_id = document_results[user['user_id']]['document_id']
        
        for entity in test_entities:
            print(f"\n   🔍 Finding passages for: {entity}")
            
            try:
                result = user_aware_rag.find_entity_passages_for_user(entity, user['user_id'], document_id)
                
                if "error" in result:
                    print(f"      ❌ Error: {result['error']}")
                    continue
                
                passages = result['passages']
                user_progress = result['user_progress']
                
                print(f"      📊 Total passages: {result['total_passages']}")
                print(f"      🛡️ Spoiler protected: {user_progress['spoiler_protected']}")
                
                if user_progress['spoiler_protected']:
                    print(f"      🛡️ Spoiler limit: {result['spoiler_limit']} (user on page {user_progress['current_page']})")
                
                # Show sample passages
                if passages:
                    print(f"      📄 Sample passages:")
                    for i, passage in enumerate(passages[:3]):
                        print(f"         {i+1}. Page {passage['page_number']}: {passage['surrounding_text'][:80]}...")
                        print(f"            Context: {passage['context_type']}, Score: {passage['context_score']:.2f}")
                else:
                    print(f"      📄 No passages found (possibly due to spoiler protection)")
                
            except Exception as e:
                print(f"      ❌ Passage finding failed: {e}")
    
    # Step 4: Test user progress updates
    print(f"\n📈 Step 4: User Progress Updates")
    print("-" * 40)
    
    # Simulate Alice reading more pages
    alice_user = test_users[0]
    if alice_user['user_id'] in document_results:
        document_id = document_results[alice_user['user_id']]['document_id']
        
        print(f"\n👤 Simulating Alice's reading progress...")
        
        # Update Alice's progress
        new_progress = user_aware_rag.update_user_progress(
            alice_user['user_id'], document_id, page=7, finished_book=False
        )
        
        if new_progress:
            print(f"   ✅ Progress updated:")
            print(f"      📄 Current page: {new_progress.current_page}")
            print(f"      📊 Progress: {new_progress.get_progress_percentage():.1f}%")
            print(f"      🛡️ Spoiler protected: {new_progress.is_spoiler_protected()}")
            print(f"      📍 Spoiler limit: {new_progress.get_spoiler_limit()}")
        
        # Test a query with updated progress
        print(f"\n   🔍 Testing query with updated progress...")
        result = user_aware_rag.query_with_user_context(
            "Where is Kingsport mentioned?", alice_user['user_id'], document_id
        )
        
        if "error" not in result:
            rag_result = result['rag_result']
            user_progress = result['user_progress']
            print(f"      📄 Source pages: {rag_result['source_pages']}")
            print(f"      🛡️ Spoiler protected: {rag_result['spoiler_protected']}")
            print(f"      📍 Spoiler limit: {result['spoiler_limit']} (user now on page {user_progress['current_page']})")
    
    # Step 5: Test user reading summary
    print(f"\n📊 Step 5: User Reading Summary")
    print("-" * 40)
    
    for user in test_users:
        print(f"\n👤 Reading summary for {user['name']}:")
        
        try:
            summary = user_aware_rag.get_user_reading_summary(user['user_id'])
            
            print(f"   📚 Total documents: {summary['total_documents']}")
            print(f"   ✅ Finished documents: {summary['finished_documents']}")
            print(f"   📖 In progress documents: {summary['in_progress_documents']}")
            print(f"   📄 Not started documents: {summary['not_started_documents']}")
            
            if summary['documents']:
                print(f"   📋 Documents:")
                for doc in summary['documents']:
                    print(f"      • {doc['document_name']}: {doc['reading_status']} ({doc['progress_percentage']:.1f}%)")
            
        except Exception as e:
            print(f"   ❌ Summary failed: {e}")
    
    # Final summary
    print(f"\n🎉 User-Aware RAG Service Test Complete!")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   Document processed: {test_doc}")
    print(f"   Users tested: {len(test_users)}")
    print(f"   Queries tested: {len(test_queries)}")
    print(f"   Entities tested: {len(test_entities)}")
    
    print(f"\n👤 User Experience Benefits:")
    print(f"   • Users can upload documents and set their current page")
    print(f"   • Spoiler protection automatically applies based on user progress")
    print(f"   • Users can update their progress as they read")
    print(f"   • Entity-chunk linking works with user context")
    print(f"   • Reading progress is tracked across multiple documents")
    print(f"   • Users can mark books as finished to disable spoiler protection")
    
    return document_results


if __name__ == "__main__":
    print("👤 User-Aware RAG Service Test")
    print("🔮 Testing user progress tracking and spoiler protection...")
    print()
    
    results = test_user_aware_rag()
    
    if results:
        print(f"\n✅ Test completed successfully!")
        print(f"📈 Check your Neo4j browser for the updated graph with entity relationships.")
        print(f"💾 User progress data is stored in data/user_progress/")
    else:
        print(f"\n❌ Test failed!")
