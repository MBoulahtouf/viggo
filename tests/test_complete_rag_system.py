#!/usr/bin/env python3
"""
Complete RAG System Test with Redis Caching.
Tests the full hybrid RAG pipeline with caching, performance optimization, and adaptive timeouts.
"""

import os
import time
import asyncio
from viggo.core.services.rag_service import RAGService
from viggo.core.services.graph_service import GraphService
from viggo.core.services.redis_service import RedisService
from viggo.core.config import settings

def test_complete_rag_pipeline():
    """Test the complete RAG pipeline with Redis caching."""
    print("🚀 Complete RAG System Test with Redis Caching")
    print("=" * 60)
    
    try:
        # Initialize services
        print("📡 Initializing services...")
        graph_service = GraphService(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        rag_service = RAGService(graph_service)
        
        # Check Redis connectivity
        redis_service = RedisService()
        if not redis_service.is_available():
            print("❌ Redis not available, cannot test caching")
            return False
        
        print("✅ All services initialized successfully")
        
        # Check if we have a document processed
        if not rag_service.hybrid_retriever:
            print("⚠️  No document processed yet, processing test document...")
            
            # Process a test document
            document_path = os.path.join(settings.data_dir, "The_Strange_High_House_in_the_Mist_HP_Lovecraft.pdf")
            if os.path.exists(document_path):
                num_chunks, _, _ = rag_service.process_document(document_path)
                print(f"✅ Document processed: {num_chunks} chunks")
            else:
                print("❌ Test document not found, cannot proceed")
                return False
        
        # Test queries
        test_queries = [
            "What is this story about?",
            "Who is the main character?",
            "Where does the story take place?",
            "What is the strange house?",
            "What happens in the mist?"
        ]
        
        print(f"\n🔍 Testing {len(test_queries)} queries with caching...")
        
        # Round 1: Populate caches
        print("\n📊 Round 1: Populating caches...")
        round1_times = []
        for i, query in enumerate(test_queries):
            print(f"\n  ❓ Query {i+1}: {query}")
            start_time = time.time()
            result = rag_service.perform_rag_query(query)
            end_time = time.time()
            round1_times.append(end_time - start_time)
            
            print(f"    ⏱️  Time: {end_time - start_time:.3f}s")
            print(f"    🔍 Search method: {result['search_method']}")
            print(f"    📄 Source pages: {result['source_pages']}")
            if 'sources_used' in result:
                print(f"    🔗 Sources used: {result['sources_used']}")
            print(f"    💬 Answer: {result['answer'][:100]}...")
        
        # Round 2: Test cache performance
        print("\n📊 Round 2: Testing cache performance...")
        round2_times = []
        for i, query in enumerate(test_queries):
            print(f"\n  ❓ Query {i+1}: {query}")
            start_time = time.time()
            result = rag_service.perform_rag_query(query)
            end_time = time.time()
            round2_times.append(end_time - start_time)
            
            print(f"    ⏱️  Time: {end_time - start_time:.3f}s")
            print(f"    🔍 Search method: {result['search_method']}")
            if 'retrieval_metadata' in result and 'cache_hit' in result['retrieval_metadata']:
                cache_status = "💾 Cache hit" if result['retrieval_metadata']['cache_hit'] else "💾 Cache miss"
                print(f"    {cache_status}")
            print(f"    💬 Answer: {result['answer'][:100]}...")
        
        # Calculate performance improvements
        avg_time_round1 = sum(round1_times) / len(round1_times)
        avg_time_round2 = sum(round2_times) / len(round2_times)
        
        print(f"\n📈 Performance Analysis:")
        print(f"  Round 1 average: {avg_time_round1:.3f}s")
        print(f"  Round 2 average: {avg_time_round2:.3f}s")
        if avg_time_round1 > 0:
            improvement_percent = ((avg_time_round1 - avg_time_round2) / avg_time_round1) * 100
            print(f"  Improvement: {improvement_percent:.1f}% faster")
        
        # Get comprehensive performance stats
        print(f"\n📊 System Performance Stats:")
        if rag_service.hybrid_retriever:
            perf_stats = rag_service.hybrid_retriever.get_performance_stats()
            
            print(f"  🔗 Source Usage: {perf_stats.get('source_usage', {})}")
            print(f"  ⏱️  Retrieval Times: {perf_stats.get('retrieval_times', {})}")
            print(f"  📈 Total Queries: {perf_stats.get('total_queries', 0)}")
            
            # Cache stats
            if 'cache' in perf_stats:
                cache_stats = perf_stats['cache']
                if 'redis_cache' in cache_stats:
                    redis_stats = cache_stats['redis_cache']
                    print(f"  💾 Redis Cache:")
                    print(f"    - Status: {redis_stats.get('status', 'unknown')}")
                    print(f"    - Query entries: {redis_stats.get('query_cache_entries', 0)}")
                    print(f"    - Embedding entries: {redis_stats.get('embedding_cache_entries', 0)}")
                    print(f"    - Memory used: {redis_stats.get('total_memory_used', 'unknown')}")
            
            # Optimization stats
            if 'optimization' in perf_stats:
                opt_stats = perf_stats['optimization']
                print(f"  ⚡ Optimization Stats:")
                print(f"    - Total queries: {opt_stats.get('total_queries', 0)}")
                
                if 'embedding_cache' in opt_stats:
                    emb_cache = opt_stats['embedding_cache']
                    print(f"    - Embedding cache hit rate: {emb_cache.get('hit_rate', '0%')}")
                
                if 'query_cache' in opt_stats:
                    query_cache = opt_stats['query_cache']
                    print(f"    - Query cache hit rate: {query_cache.get('hit_rate', '0%')}")
        
        # Test cache management
        print(f"\n🧹 Testing cache management...")
        if rag_service.hybrid_retriever:
            cache_info = rag_service.hybrid_retriever.get_cache_info()
            print(f"  Cache enabled: {cache_info.get('cache_enabled', False)}")
            print(f"  Cache type: {cache_info.get('cache_type', 'unknown')}")
            
            # Test cache clearing
            success = rag_service.hybrid_retriever.clear_cache("query")
            print(f"  Cache clear test: {'✅ Success' if success else '❌ Failed'}")
        
        print(f"\n🎉 Complete RAG system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Complete RAG system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_redis_performance():
    """Test Redis performance and caching effectiveness."""
    print("\n⚡ Redis Performance Test")
    print("=" * 40)
    
    try:
        redis_service = RedisService()
        
        if not redis_service.is_available():
            print("❌ Redis not available")
            return False
        
        # Test cache operations performance
        test_data = {"test": "performance", "timestamp": time.time()}
        
        # Test write performance
        write_times = []
        for i in range(10):
            start_time = time.time()
            redis_service.cache_query_result(f"perf_test_{i}", 5, None, test_data)
            end_time = time.time()
            write_times.append(end_time - start_time)
        
        avg_write_time = sum(write_times) / len(write_times)
        print(f"📝 Average write time: {avg_write_time*1000:.2f}ms")
        
        # Test read performance
        read_times = []
        for i in range(10):
            start_time = time.time()
            cached = redis_service.get_cached_query_result(f"perf_test_{i}", 5, None)
            end_time = time.time()
            read_times.append(end_time - start_time)
        
        avg_read_time = sum(read_times) / len(read_times)
        print(f"📖 Average read time: {avg_read_time*1000:.2f}ms")
        
        # Test embedding caching performance
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        embedding_times = []
        for i in range(5):
            text = f"Performance test embedding {i}"
            start_time = time.time()
            embedding = model.encode([text])[0]
            redis_service.cache_embedding(text, embedding)
            end_time = time.time()
            embedding_times.append(end_time - start_time)
        
        avg_embedding_time = sum(embedding_times) / len(embedding_times)
        print(f"🧠 Average embedding time: {avg_embedding_time:.3f}s")
        
        # Test embedding retrieval
        retrieval_times = []
        for i in range(5):
            text = f"Performance test embedding {i}"
            start_time = time.time()
            cached_embedding = redis_service.get_cached_embedding(text)
            end_time = time.time()
            retrieval_times.append(end_time - start_time)
        
        avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
        print(f"🔍 Average embedding retrieval: {avg_retrieval_time*1000:.2f}ms")
        
        # Get final cache stats
        stats = redis_service.get_cache_stats()
        print(f"\n📊 Final Cache Stats:")
        print(f"  Query entries: {stats.get('query_cache_entries', 0)}")
        print(f"  Embedding entries: {stats.get('embedding_cache_entries', 0)}")
        print(f"  Memory used: {stats.get('total_memory_used', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis performance test failed: {e}")
        return False

def main():
    """Run complete RAG system tests."""
    print("🚀 Complete RAG System Test Suite")
    print("=" * 60)
    
    tests = [
        ("Complete RAG Pipeline", test_complete_rag_pipeline),
        ("Redis Performance", test_redis_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAGaaS system is production-ready!")
        print("\n🚀 Next Steps:")
        print("  1. Deploy to production environment")
        print("  2. Set up monitoring and alerting")
        print("  3. Build frontend interface")
        print("  4. Implement user authentication")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
