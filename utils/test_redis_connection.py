#!/usr/bin/env python3
"""
Simple test to verify connection to Azure Redis Cache.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_redis_connection():
    """Test basic connection to Azure Redis Cache."""
    try:
        from viggo.core.services.redis_service import RedisService
        
        print("🔗 Testing connection to Azure Redis Cache...")
        print(f"Host: viggo-redis.redis.cache.windows.net")
        print(f"Port: 6380 (SSL)")
        
        # Initialize Redis service
        redis_service = RedisService()
        
        # Check if connection is available
        if redis_service.is_available():
            print("✅ Successfully connected to Azure Redis Cache!")
            
            # Test health check
            health = redis_service.health_check()
            print(f"Health Status: {health['status']}")
            
            if health['status'] == 'healthy':
                print(f"Ping Time: {health.get('ping_time_ms', 'N/A')}ms")
                
                # Test basic operations
                test_key = "viggo:test:connection"
                test_data = {"message": "Hello from Azure Redis!", "timestamp": "2024-01-01"}
                
                # Test cache operation
                success = redis_service.cache_query_result(
                    "test connection query",
                    5,
                    None,
                    test_data
                )
                
                if success:
                    print("✅ Cache write operation successful")
                    
                    # Test retrieval
                    cached = redis_service.get_cached_query_result("test connection query", 5, None)
                    if cached:
                        print("✅ Cache read operation successful")
                        print(f"Cached data: {cached}")
                    else:
                        print("❌ Cache read operation failed")
                else:
                    print("❌ Cache write operation failed")
                
                # Get cache stats
                stats = redis_service.get_cache_stats()
                print(f"Cache Stats: {stats}")
                
                # Cleanup
                redis_service.clear_cache("viggo:test:*")
                print("✅ Cleanup completed")
                
            else:
                print(f"❌ Redis health check failed: {health.get('error', 'Unknown error')}")
        else:
            print("❌ Failed to connect to Azure Redis Cache")
            print("Check your Redis configuration and network connectivity")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure to install dependencies: pip install redis")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("Check your Azure Redis Cache configuration")

if __name__ == "__main__":
    test_redis_connection()
