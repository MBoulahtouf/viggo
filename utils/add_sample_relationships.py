#!/usr/bin/env python3
"""
Add some sample relationships to the Lovecraft knowledge graph for visualization.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viggo.core.services.graph_service import GraphService
from viggo.core.config import settings


def add_sample_relationships():
    """Add some sample relationships for visualization."""
    
    print("🔗 Adding sample relationships to Lovecraft knowledge graph...")
    
    try:
        # Initialize Graph Service
        graph_service = GraphService(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        print("✅ Connected to Neo4j")
        
        # Add some sample relationships based on the story
        relationships_to_add = [
            # Main character relationships
            ("Thomas Olney", "Character", "Kingsport", "Location", "VISITS"),
            ("Olney", "Character", "Kingsport", "Location", "LIVES_IN"),
            ("Olney", "Character", "the Elder Ones", "Organization", "ENCOUNTERS"),
            ("Olney", "Character", "Nodens", "Character", "MEETS"),
            
            # Location relationships
            ("Kingsport", "Location", "Arkham", "Location", "NEAR"),
            ("Central Hill", "Location", "Kingsport", "Location", "IN"),
            ("Congregational Hospital", "Organization", "Central Hill", "Location", "LOCATED_ON"),
            
            # Mythological relationships
            ("Nodens", "Character", "Neptune", "Organization", "SERVES"),
            ("the Elder Ones", "Organization", "Kingsport", "Location", "INFLUENCES"),
            ("Hatheg-Kia", "Organization", "Ulthar", "Location", "ORIGINATES_FROM"),
            
            # Character interactions
            ("Kingsporter", "Organization", "Olney", "Character", "WARNS"),
            ("Granny Orne", "Character", "Kingsport", "Location", "LIVES_IN"),
            ("Miskatonic", "Character", "Arkham", "Location", "WORKS_IN"),
        ]
        
        relationships_created = 0
        
        for source_name, source_label, target_name, target_label, rel_type in relationships_to_add:
            try:
                # Create the relationship
                graph_service.create_relationship(
                    source_name, source_label, 
                    target_name, target_label, 
                    rel_type
                )
                relationships_created += 1
                print(f"✅ Created: {source_name} --[{rel_type}]--> {target_name}")
                
            except Exception as e:
                print(f"⚠️  Failed to create relationship {source_name} --[{rel_type}]--> {target_name}: {e}")
        
        print(f"\n📊 Created {relationships_created} relationships")
        
        # Get final statistics
        print("\n📈 Final Database Statistics:")
        all_nodes = graph_service.list_all_nodes()
        node_counts = {}
        for node in all_nodes:
            for label in node.labels:
                node_counts[label] = node_counts.get(label, 0) + 1
        
        for label, count in sorted(node_counts.items()):
            print(f"   {label}: {count}")
        
        print(f"\n🌐 Neo4j Browser Access:")
        print(f"   URL: http://20.216.195.227:7474")
        print(f"   Username: neo4j")
        print(f"   Password: viggo123")
        
        print(f"\n🎨 Try these queries now:")
        print("""
# View all relationships
MATCH (n)-[r]->(m) 
RETURN n, r, m 
LIMIT 20;

# View relationships involving Olney
MATCH (n)-[r]->(m) 
WHERE n.name CONTAINS 'Olney' OR m.name CONTAINS 'Olney'
RETURN n, r, m;

# View all locations and their connections
MATCH (l:Location)-[r]-(n)
RETURN l, r, n
LIMIT 15;

# View the story structure
MATCH (c:Character)-[r]-(l:Location)
RETURN c, r, l
LIMIT 20;
        """)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = add_sample_relationships()
    if success:
        print("\n🎉 Relationships added successfully!")
    else:
        print("\n💥 Failed to add relationships!")
    
    sys.exit(0 if success else 1)
