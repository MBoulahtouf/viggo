#!/usr/bin/env python3
"""
Clean up noisy entities from the Lovecraft knowledge graph.
Remove metadata, file paths, and publisher information.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viggo.core.services.graph_service import GraphService
from viggo.core.config import settings


def cleanup_neo4j_data():
    """Remove noisy entities and chunks from Neo4j."""
    
    print("🧹 Cleaning up noisy data from Lovecraft knowledge graph...")
    
    try:
        # Initialize Graph Service
        graph_service = GraphService(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        print("✅ Connected to Neo4j")
        
        # Patterns to remove (noisy entities)
        noisy_patterns = [
            # File paths and technical metadata
            "/home/mikealpharomeo/Projects/viggo/data/",
            "_page",
            "_chunk",
            ".pdf",
            
            # Publisher metadata
            "OceanofPDF",
            "Feedbooks",
            "Wikipedia",
            "Source:",
            "Also available",
            "Copyright:",
            "Life+70",
            
            # Generic metadata
            "Published:",
            "Categorie(s):",
            "Fiction, Short Stories",
            "Strictly for personal use",
            "do not use this file for commercial purposes",
            
            # Lovecraft bibliography (not story content)
            "The Call of Cthulhu",
            "At the Mountains of Madness", 
            "The Dunwich Horror",
            "The Shadow out of Time",
            "The Shadow Over Innsmouth",
            "The Haunter of the Dark",
            "The Colour Out of Space",
            "The Whisperer in Darkness",
            "Supernatural Horror in Literature",
            "Dreams in the Witch-House",
            "The Statement of Randolph Carter",
            "The Silver Key",
            "The Tree",
            "What the Moon Brings",
            "The Temple",
            "Howard Phillips Lovecraft Poetry",
        ]
        
        with graph_service.driver.session() as session:
            # Remove chunks with noisy content
            for pattern in noisy_patterns:
                # Remove chunks containing these patterns
                result = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.content CONTAINS $pattern
                    DELETE c
                    RETURN count(c) as deleted
                """, pattern=pattern)
                
                count = result.single()["deleted"]
                if count > 0:
                    print(f"🗑️  Removed {count} chunks containing '{pattern}'")
            
            # Remove entities that are just metadata
            metadata_entities = [
                "Howard Phillips Lovecraft",  # Author name (appears too many times)
                "Lovecraft",
                "OceanofPDF.com",
                "Feedbooks Lovecraft",
                "The Dunwich Horror",
                "Time",
                "The Haunter of",
                "The Whisperer in Darkness",
                "Supernatural Horror in Literature",
                "Witch-House",
                "The Temple",
                "Howard Phillips Lovecraft Poetry",
                "Randolph Carter",
                "The Silver Key",
                "The Tree",
                "What the Moon Brings",
            ]
            
            for entity_name in metadata_entities:
                # First delete relationships, then the node
                result = session.run("""
                    MATCH (n)
                    WHERE n.name = $name
                    DETACH DELETE n
                    RETURN count(n) as deleted
                """, name=entity_name)
                
                count = result.single()["deleted"]
                if count > 0:
                    print(f"🗑️  Removed {count} entities named '{entity_name}'")
            
            # Remove orphaned nodes (nodes with no relationships)
            result = session.run("""
                MATCH (n)
                WHERE NOT (n)--()
                DETACH DELETE n
                RETURN count(n) as deleted
            """)
            
            orphaned_count = result.single()["deleted"]
            if orphaned_count > 0:
                print(f"🗑️  Removed {orphaned_count} orphaned nodes")
            
            # Get final statistics
            result = session.run("""
                MATCH (n)
                RETURN labels(n) as labels, count(n) as count
                ORDER BY count DESC
            """)
            
            print(f"\n📊 Final Clean Database Statistics:")
            for record in result:
                labels = record["labels"]
                count = record["count"]
                print(f"   {labels}: {count}")
        
        print(f"\n✅ Cleanup completed!")
        print(f"\n🌐 Neo4j Browser Access:")
        print(f"   URL: http://20.216.195.227:7474")
        print(f"   Username: neo4j")
        print(f"   Password: viggo123")
        
        print(f"\n🎨 Try these clean queries now:")
        print("""
# View all relationships (should be cleaner now)
MATCH (n)-[r]->(m) 
RETURN n, r, m 
LIMIT 20;

# View only story-relevant characters and locations
MATCH (c:Character)-[r]-(l:Location)
WHERE c.name IN ['Olney', 'Thomas Olney', 'Granny Orne', 'Nodens']
RETURN c, r, l;

# View the core story network
MATCH (n)-[r]->(m)
WHERE n.name IN ['Olney', 'Kingsport', 'the Elder Ones', 'Nodens', 'Arkham']
   OR m.name IN ['Olney', 'Kingsport', 'the Elder Ones', 'Nodens', 'Arkham']
RETURN n, r, m;
        """)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = cleanup_neo4j_data()
    if success:
        print("\n🎉 Data cleanup completed successfully!")
    else:
        print("\n💥 Cleanup failed!")
    
    sys.exit(0 if success else 1)
