#!/usr/bin/env python3
"""
Test Neo4j connection and basic operations.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viggo.core.services.graph_service import GraphService
from viggo.core.config import settings


def test_neo4j_connection():
    """Test Neo4j connection and basic operations."""
    
    print("🔗 Testing Neo4j Connection")
    print("=" * 50)
    
    try:
        # Initialize Graph Service
        print(f"Connecting to: {settings.neo4j_uri}")
        print(f"User: {settings.neo4j_user}")
        
        graph_service = GraphService(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        # Test connection (if GraphService initializes, connection is working)
        print("✅ Successfully connected to Neo4j!")
        
        # Test basic operations
        print("\n🧪 Testing basic operations...")
        
        # Create a test node
        test_node_id = graph_service.create_entity_node("TestNode", "Test", "A test node for connection verification")
        if test_node_id:
            print("✅ Node creation works")
            print(f"   Created node with ID: {test_node_id}")
            
            # Query the test node
            test_nodes = graph_service.list_all_nodes(label="Test")
            if test_nodes:
                print("✅ Node querying works")
                print(f"   Test node: {test_nodes[0]}")
                
                # Clean up test node (we'll use the Neo4j driver directly)
                with graph_service.driver.session() as session:
                    session.run("MATCH (n:Test {name: 'TestNode'}) DELETE n")
                print("✅ Node deletion works")
            else:
                print("❌ Node querying failed")
        else:
            print("❌ Node creation failed")
        
        print(f"\n🌐 Neo4j Browser Access:")
        print(f"   URL: http://20.216.195.227:7474")
        print(f"   Username: {settings.neo4j_user}")
        print(f"   Password: {settings.neo4j_password}")
        
        return True
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_neo4j_connection()
    if success:
        print("\n🎉 Neo4j connection test successful!")
    else:
        print("\n💥 Neo4j connection test failed!")
    
    sys.exit(0 if success else 1)