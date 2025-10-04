#!/usr/bin/env python3
"""
Improved entity extraction for Lovecraft stories that focuses on narrative content.
Filters out metadata, publisher info, and file paths.
"""

import os
import sys
import re
from typing import List, Dict, Set

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viggo.core.services.rag_service import RAGService
from viggo.core.services.graph_service import GraphService
from viggo.core.config import settings


class ImprovedEntityExtractor:
    """Enhanced entity extractor that filters out noise."""
    
    def __init__(self):
        # Patterns to exclude from entity extraction
        self.exclude_patterns = {
            # File paths and technical metadata
            r'^/.*\.pdf$',
            r'.*_page\d+_chunk\d+.*',
            r'.*\.pdf.*',
            
            # Publisher and metadata
            r'^OceanofPDF.*',
            r'^Feedbooks.*',
            r'^Wikipedia.*',
            r'^Source:.*',
            r'^Also available.*',
            r'^Copyright:.*',
            r'^Published:.*',
            r'^Categorie\(s\):.*',
            r'^Fiction, Short Stories$',
            r'^Strictly for personal use.*',
            r'^do not use this file.*',
            r'^Life\+70.*',
            
            # Lovecraft bibliography (not story content)
            r'^The Call of Cthulhu.*',
            r'^At the Mountains of Madness.*',
            r'^The Dunwich Horror.*',
            r'^The Shadow out of Time.*',
            r'^The Shadow Over Innsmouth.*',
            r'^The Haunter of the Dark.*',
            r'^The Colour Out of Space.*',
            r'^The Whisperer in Darkness.*',
            r'^Supernatural Horror in Literature.*',
            r'^Dreams in the Witch-House.*',
            r'^The Statement of Randolph Carter.*',
            r'^The Silver Key.*',
            r'^The Tree.*',
            r'^What the Moon Brings.*',
            r'^The Temple.*',
            r'^Howard Phillips Lovecraft Poetry.*',
            
            # Generic metadata
            r'^Howard Phillips Lovecraft$',  # Author name
            r'^Lovecraft$',
            r'^Time$',
        }
        
        # Story-relevant entity patterns to prioritize
        self.story_entities = {
            'characters': [
                'Olney', 'Thomas Olney', 'Granny Orne', 'Nodens', 
                'Neptune', 'Kadath', 'Miskatonic', 'Kingsporter'
            ],
            'locations': [
                'Kingsport', 'Arkham', 'Central Hill', 'Tudor', 
                'Poseidon', 'Ulthar', 'New England', 'Bristol Highlands'
            ],
            'organizations': [
                'the Elder Ones', 'Congregational Hospital', 'Hatheg-Kia',
                'Mighty Ones', 'Yankees', 'Dragon', 'the Great Bear'
            ]
        }
    
    def is_noisy_entity(self, entity_text: str) -> bool:
        """Check if an entity is noise that should be filtered out."""
        entity_text = entity_text.strip()
        
        # Check against exclude patterns
        for pattern in self.exclude_patterns:
            if re.match(pattern, entity_text, re.IGNORECASE):
                return True
        
        # Filter out very short or very long entities
        if len(entity_text) < 2 or len(entity_text) > 50:
            return True
        
        # Filter out entities that are mostly numbers or special characters
        if re.match(r'^[\d\s\-_\.]+$', entity_text):
            return True
        
        return False
    
    def is_story_content(self, chunk_content: str) -> bool:
        """Check if a chunk contains actual story content."""
        # Skip chunks that are mostly metadata
        metadata_indicators = [
            'Published:', 'Categorie(s):', 'Source:', 'Copyright:',
            'Also available', 'OceanofPDF', 'Feedbooks', 'Wikipedia',
            'Strictly for personal use', 'Life+70', 'Available for countries'
        ]
        
        content_lower = chunk_content.lower()
        metadata_score = sum(1 for indicator in metadata_indicators 
                           if indicator.lower() in content_lower)
        
        # If more than 2 metadata indicators, likely not story content
        return metadata_score < 2
    
    def filter_entities(self, entities: List[Dict], chunk_content: str) -> List[Dict]:
        """Filter entities to keep only story-relevant ones."""
        if not self.is_story_content(chunk_content):
            return []  # Skip entire chunk if it's metadata
        
        filtered_entities = []
        for entity in entities:
            entity_text = entity.get('text', '').strip()
            
            # Skip noisy entities
            if self.is_noisy_entity(entity_text):
                continue
            
            # Keep story-relevant entities
            filtered_entities.append(entity)
        
        return filtered_entities
    
    def enhance_entity_labels(self, entities: List[Dict]) -> List[Dict]:
        """Improve entity labels based on Lovecraft story context."""
        enhanced_entities = []
        
        for entity in entities:
            entity_text = entity.get('text', '').strip()
            original_label = entity.get('label', '')
            
            # Override labels for known Lovecraft entities
            if entity_text in self.story_entities['characters']:
                entity['label'] = 'Character'
            elif entity_text in self.story_entities['locations']:
                entity['label'] = 'Location'
            elif entity_text in self.story_entities['organizations']:
                entity['label'] = 'Organization'
            
            enhanced_entities.append(entity)
        
        return enhanced_entities


def process_lovecraft_improved():
    """Process Lovecraft story with improved entity extraction."""
    
    print("🏚️  Processing Lovecraft story with improved entity extraction...")
    
    try:
        # Initialize services
        graph_service = GraphService(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        rag_service = RAGService(graph_service=graph_service)
        extractor = ImprovedEntityExtractor()
        
        # Clear existing data first
        print("🧹 Clearing existing data...")
        graph_service.clear_database()
        
        # Process the document
        lovecraft_file = "/home/mikealpharomeo/Projects/viggo/data/The_Strange_High_House_in_the_Mist_HP_Lovecraft.pdf"
        print(f"📖 Processing: {lovecraft_file}")
        
        num_chunks, vector_index, chunks_with_metadata = rag_service.process_document(lovecraft_file)
        
        print(f"📊 Processed {num_chunks} chunks")
        
        # Filter and enhance entities
        print("🔍 Filtering and enhancing entities...")
        filtered_chunks = []
        
        for chunk in chunks_with_metadata:
            # Filter entities
            filtered_entities = extractor.filter_entities(
                chunk.get('entities', []), 
                chunk.get('content', '')
            )
            
            # Enhance entity labels
            enhanced_entities = extractor.enhance_entity_labels(filtered_entities)
            
            # Update chunk with filtered entities
            chunk['entities'] = enhanced_entities
            chunk['relationships'] = []  # Clear relationships for now
            
            # Only keep chunks with story-relevant entities
            if enhanced_entities and extractor.is_story_content(chunk.get('content', '')):
                filtered_chunks.append(chunk)
        
        print(f"✅ Filtered to {len(filtered_chunks)} story-relevant chunks")
        
        # Load filtered data into Neo4j
        print("🔗 Loading filtered data into Neo4j...")
        graph_service.extract_and_load_graph(lovecraft_file, filtered_chunks)
        
        # Add clean relationships
        print("🔗 Adding story relationships...")
        story_relationships = [
            ("Thomas Olney", "Character", "Kingsport", "Location", "VISITS"),
            ("Olney", "Character", "Kingsport", "Location", "LIVES_IN"),
            ("Olney", "Character", "the Elder Ones", "Organization", "ENCOUNTERS"),
            ("Olney", "Character", "Nodens", "Character", "MEETS"),
            ("Kingsport", "Location", "Arkham", "Location", "NEAR"),
            ("Central Hill", "Location", "Kingsport", "Location", "IN"),
            ("Congregational Hospital", "Organization", "Central Hill", "Location", "LOCATED_ON"),
            ("Nodens", "Character", "Neptune", "Organization", "SERVES"),
            ("the Elder Ones", "Organization", "Kingsport", "Location", "INFLUENCES"),
            ("Granny Orne", "Character", "Kingsport", "Location", "LIVES_IN"),
            ("Miskatonic", "Character", "Arkham", "Location", "WORKS_IN"),
        ]
        
        for source_name, source_label, target_name, target_label, rel_type in story_relationships:
            try:
                graph_service.create_relationship(
                    source_name, source_label, 
                    target_name, target_label, 
                    rel_type
                )
                print(f"✅ Created: {source_name} --[{rel_type}]--> {target_name}")
            except Exception as e:
                print(f"⚠️  Failed to create relationship: {e}")
        
        # Get final statistics
        print("\n📈 Final Clean Database Statistics:")
        all_nodes = graph_service.list_all_nodes()
        node_counts = {}
        for node in all_nodes:
            for label in node.labels:
                node_counts[label] = node_counts.get(label, 0) + 1
        
        for label, count in sorted(node_counts.items()):
            print(f"   {label}: {count}")
        
        print(f"\n✅ Improved processing completed!")
        print(f"📊 Story-relevant entities: {sum(len(chunk.get('entities', [])) for chunk in filtered_chunks)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = process_lovecraft_improved()
    if success:
        print("\n🎉 Improved processing completed!")
    else:
        print("\n💥 Processing failed!")
    
    sys.exit(0 if success else 1)
