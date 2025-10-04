#!/usr/bin/env python3
"""
Process 'The Strange High House in the Mist' with enhanced entity extraction.
This script uses the new enhanced entity extractor to filter out noise and improve entity quality.
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


async def process_lovecraft_enhanced():
    """Process the Lovecraft story with enhanced entity extraction."""
    
    print("🏚️  Processing 'The Strange High House in the Mist' with Enhanced Entity Extraction")
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
        
        # Clear existing data
        print("🧹 Clearing existing data...")
        graph_service.clear_database()
        
        print("✅ Connected to Neo4j successfully")
        
        # Initialize RAG Service with Graph Service
        print("🧠 Initializing RAG Service with enhanced entity extraction...")
        rag_service = RAGService(graph_service=graph_service)
        
        # Process the document with enhanced extraction
        print("📚 Processing document with enhanced entity extraction...")
        num_chunks, vector_index, chunks_with_metadata = rag_service.process_document_enhanced(lovecraft_file)
        
        # Load the enhanced entities and relationships into Neo4j
        print("🔗 Loading enhanced entities and relationships into Neo4j...")
        graph_service.extract_and_load_graph(lovecraft_file, chunks_with_metadata)
        
        print(f"✅ Enhanced document processing completed!")
        print(f"   📊 Number of filtered chunks: {num_chunks}")
        print(f"   🔍 Vector index built: {'Yes' if vector_index else 'No'}")
        
        # Get enhanced statistics
        if chunks_with_metadata:
            total_entities = sum(len(chunk.get('entities', [])) for chunk in chunks_with_metadata)
            total_relationships = sum(len(chunk.get('relationships', [])) for chunk in chunks_with_metadata)
            
            print(f"   🏷️  Total enhanced entities: {total_entities}")
            print(f"   🔗 Total relationships: {total_relationships}")
        
        # Get enhanced sample entities for visualization
        print("\n🎭 Enhanced Characters (deduplicated):")
        characters = graph_service.list_all_nodes(label="Character")
        for char in characters[:10]:  # Limit to 10
            print(f"   • {char.name} (Labels: {char.labels})")
        
        print("\n🏘️  Enhanced Locations (deduplicated):")
        locations = graph_service.list_all_nodes(label="Location")
        for loc in locations[:10]:  # Limit to 10
            print(f"   • {loc.name} (Labels: {loc.labels})")
        
        print("\n🏢 Enhanced Organizations (deduplicated):")
        organizations = graph_service.list_all_nodes(label="Organization")
        for org in organizations[:10]:  # Limit to 10
            print(f"   • {org.name} (Labels: {org.labels})")
        
        print(f"\n📊 Enhanced nodes by type:")
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
        
        print("\n🎨 Recommended Cypher Queries for Enhanced Visualization:")
        print("""
# View all enhanced entities and relationships
MATCH (n)-[r]->(m) 
RETURN n, r, m 
LIMIT 50;

# View all characters (should be deduplicated now)
MATCH (c:Character) 
RETURN c;

# View all locations (should be deduplicated now)
MATCH (l:Location) 
RETURN l;

# View all organizations (should be deduplicated now)
MATCH (o:Organization) 
RETURN o;

# View relationships between specific entities (should be cleaner now)
MATCH (n)-[r]->(m) 
WHERE n.name CONTAINS 'Olney' OR m.name CONTAINS 'Olney'
RETURN n, r, m;

# Get entity statistics (should show better distribution)
MATCH (n) 
RETURN labels(n) as EntityType, count(n) as Count 
ORDER BY Count DESC;

# View the knowledge graph structure
CALL db.schema.visualization();
        """)
        
        print("\n✅ Enhanced processing complete! You can now visualize the clean data in Neo4j Browser.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the enhanced processing."""
    success = asyncio.run(process_lovecraft_enhanced())
    
    if success:
        print("\n🎉 Success! The Lovecraft story has been processed with enhanced entity extraction.")
        print("📖 Open Neo4j Browser to explore the clean knowledge graph!")
        print("🔍 Key improvements:")
        print("   • Filtered out metadata, prefaces, and publisher info")
        print("   • Deduplicated similar entities (e.g., 'Olney' vs 'Thomas Olney')")
        print("   • Disambiguated entity types (character vs organization conflicts)")
        print("   • Removed noisy entities and technical metadata")
    else:
        print("\n💥 Enhanced processing failed. Check the error messages above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
