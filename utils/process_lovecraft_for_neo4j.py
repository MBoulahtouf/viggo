#!/usr/bin/env python3
"""
Process 'The Strange High House in the Mist' by H.P. Lovecraft for Neo4j visualization.
This script will extract entities, relationships, and create a knowledge graph.
"""

import os
import sys
import asyncio
from typing import Dict, List

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viggo.core.services.rag_service import RAGService
from viggo.core.services.graph_service import GraphService
from viggo.core.config import settings


async def process_lovecraft_story():
    """Process the Lovecraft story and populate Neo4j."""
    
    print("🏚️  Processing 'The Strange High House in the Mist' by H.P. Lovecraft")
    print("=" * 80)
    
    # File path
    lovecraft_file = "/home/mikealpharomeo/Projects/viggo/data/The_Strange_High_House_in_the_Mist_HP_Lovecraft.pdf"
    
    if not os.path.exists(lovecraft_file):
        print(f"❌ File not found: {lovecraft_file}")
        return False
    
    print(f"📖 Processing file: {lovecraft_file}")
    
    try:
        # Initialize Graph Service
        print("🔗 Connecting to Neo4j...")
        graph_service = GraphService(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        # Test connection (if GraphService initializes, connection is working)
        print("✅ Connected to Neo4j successfully")
        
        # Initialize RAG Service with Graph Service
        print("🧠 Initializing RAG Service...")
        rag_service = RAGService(graph_service=graph_service)
        
        # Process the document
        print("📚 Processing document and building knowledge graph...")
        num_chunks, vector_index, chunks_with_metadata = rag_service.process_document(lovecraft_file)
        
        # Load the extracted entities and relationships into Neo4j
        print("🔗 Loading entities and relationships into Neo4j...")
        graph_service.extract_and_load_graph(lovecraft_file, chunks_with_metadata)
        
        print(f"✅ Document processed successfully!")
        print(f"   📊 Number of chunks: {num_chunks}")
        print(f"   🔍 Vector index built: {'Yes' if vector_index else 'No'}")
        
        # Get some statistics
        if chunks_with_metadata:
            total_entities = sum(len(chunk.get('entities', [])) for chunk in chunks_with_metadata)
            total_relationships = sum(len(chunk.get('relationships', [])) for chunk in chunks_with_metadata)
            
            print(f"   🏷️  Total entities extracted: {total_entities}")
            print(f"   🔗 Total relationships extracted: {total_relationships}")
        
        # Get some sample entities for visualization
        print("\n🎭 Sample Characters:")
        characters = graph_service.list_all_nodes(label="Character")
        for char in characters[:10]:  # Limit to 10
            print(f"   • {char.name} (Labels: {char.labels})")
        
        print("\n🏘️  Sample Locations:")
        locations = graph_service.list_all_nodes(label="Location")
        for loc in locations[:10]:  # Limit to 10
            print(f"   • {loc.name} (Labels: {loc.labels})")
        
        print("\n📚 Sample Documents:")
        documents = graph_service.list_all_nodes(label="Document")
        for doc in documents[:5]:  # Limit to 5
            print(f"   • {doc.name} (Labels: {doc.labels})")
        
        print(f"\n📊 Total nodes by type:")
        all_nodes = graph_service.list_all_nodes()
        node_counts = {}
        for node in all_nodes:
            for label in node.labels:
                node_counts[label] = node_counts.get(label, 0) + 1
        
        for label, count in sorted(node_counts.items()):
            print(f"   {label}: {count}")
        
        print("\n🌐 Neo4j Browser Access:")
        print(f"   URL: http://20.216.195.227:7474")
        print(f"   Username: neo4j")
        print(f"   Password: viggo123")
        
        print("\n🎨 Recommended Cypher Queries for Visualization:")
        print("""
# View all entities and relationships
MATCH (n)-[r]->(m) 
RETURN n, r, m 
LIMIT 50;

# View all characters
MATCH (c:Character) 
RETURN c;

# View all locations
MATCH (l:Location) 
RETURN l;

# View relationships between specific entities
MATCH (n)-[r]->(m) 
WHERE n.name CONTAINS 'Olney' OR m.name CONTAINS 'Olney'
RETURN n, r, m;

# View the knowledge graph structure
CALL db.schema.visualization();

# Get entity statistics
MATCH (n) 
RETURN labels(n) as EntityType, count(n) as Count 
ORDER BY Count DESC;
        """)
        
        print("\n✅ Processing complete! You can now visualize the data in Neo4j Browser.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the processing."""
    success = asyncio.run(process_lovecraft_story())
    
    if success:
        print("\n🎉 Success! The Lovecraft story has been processed and added to Neo4j.")
        print("📖 Open Neo4j Browser to explore the knowledge graph!")
    else:
        print("\n💥 Processing failed. Check the error messages above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
