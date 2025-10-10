"""
Azure Search-based Graph RAG implementation following SOLID principles.
Adapted from Microsoft GraphRAG approach but using Azure Search instead of Weaviate.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from viggo.core.services.interfaces.graph_service import IGraphService
from viggo.core.services.interfaces.storage import VectorStorage


@dataclass
class EntityNode:
    """Represents an entity node in the knowledge graph."""
    name: str
    label: str
    description: str
    properties: dict[str, Any]
    confidence: float


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source: str
    target: str
    relationship_type: str
    properties: dict[str, Any]
    confidence: float


@dataclass
class EntityCommunity:
    """Represents a community of related entities."""
    community_id: str
    entities: list[str]
    summary: str
    relationships: list[Relationship]
    confidence: float


class AzureGraphRAGService:
    """
    Graph RAG service using Azure Search for vector operations and Neo4j for graph operations.
    Implements Microsoft GraphRAG approach adapted for Azure Search.
    """

    def __init__(self, graph_service: IGraphService, vector_storage: VectorStorage):
        self.graph_service = graph_service
        self.vector_storage = vector_storage

        # Entity types to extract (configurable)
        self.allowed_entities = ["Person", "Organization", "Location", "Event", "Work"]

        # Relationship patterns for extraction
        self.relationship_patterns = [
            # Character relationships
            r'(\w+)\s+(said|told|asked|replied|answered)\s+(to\s+)?(\w+)',
            r'(\w+)\s+(met|encountered|saw|visited)\s+(\w+)',
            r'(\w+)\s+(lived|resided|dwelt)\s+(in|at|near)\s+(\w+)',
            r'(\w+)\s+(worked|served)\s+(at|for|in)\s+(\w+)',
            r'(\w+)\s+(belonged to|was part of|member of)\s+(\w+)',

            # Location relationships
            r'(\w+)\s+(is located|lies|sits)\s+(in|at|near)\s+(\w+)',
            r'(\w+)\s+(traveled|went|journeyed)\s+(to|towards)\s+(\w+)',
            r'(\w+)\s+(came from|originated from)\s+(\w+)',

            # Organizational relationships
            r'(\w+)\s+(created|built|founded)\s+(\w+)',
            r'(\w+)\s+(owned|possessed|controlled)\s+(\w+)',
            r'(\w+)\s+(governed|ruled|led)\s+(\w+)',

            # Event relationships
            r'(\w+)\s+(happened|occurred|took place)\s+(in|at|during)\s+(\w+)',
            r'(\w+)\s+(caused|led to|resulted in)\s+(\w+)',
            r'(\w+)\s+(preceded|followed|came after)\s+(\w+)',
        ]

        # Community detection parameters
        self.community_min_size = 2
        self.community_confidence_threshold = 0.6

    async def extract_nodes_and_relationships(self, texts: list[str]) -> tuple[list[EntityNode], list[Relationship]]:
        """
        Extract entities and relationships from texts.
        This is the first stage of GraphRAG pipeline.
        """
        print("🔍 Starting entity and relationship extraction...")

        all_entities = []
        all_relationships = []

        for i, text in enumerate(texts):
            print(f"Processing text chunk {i+1}/{len(texts)}")

            # Extract entities from text
            entities = await self._extract_entities_from_text(text, i)
            all_entities.extend(entities)

            # Extract relationships from text
            relationships = await self._extract_relationships_from_text(text, entities, i)
            all_relationships.extend(relationships)

        print(f"✅ Extracted {len(all_entities)} entities and {len(all_relationships)} relationships")
        return all_entities, all_relationships

    async def _extract_entities_from_text(self, text: str, chunk_id: int) -> list[EntityNode]:
        """Extract entities from a single text chunk."""
        entities = []

        # Use spaCy for entity extraction (if available)
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)

            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART"]:
                    # Map spaCy labels to our entity types
                    label_mapping = {
                        "PERSON": "Person",
                        "ORG": "Organization",
                        "GPE": "Location",
                        "LOC": "Location",
                        "EVENT": "Event",
                        "WORK_OF_ART": "Work"
                    }

                    entity = EntityNode(
                        name=ent.text.strip(),
                        label=label_mapping.get(ent.label_, "Entity"),
                        description=f"Entity mentioned in text chunk {chunk_id}",
                        properties={
                            "start_char": ent.start_char,
                            "end_char": ent.end_char,
                            "chunk_id": chunk_id,
                            "spacy_label": ent.label_
                        },
                        confidence=0.8
                    )
                    entities.append(entity)

        except ImportError:
            # Fallback to simple pattern-based extraction
            entities = await self._extract_entities_pattern_based(text, chunk_id)

        return entities

    async def _extract_entities_pattern_based(self, text: str, chunk_id: int) -> list[EntityNode]:
        """Fallback entity extraction using patterns."""
        entities = []

        # Pattern for capitalized words (potential proper nouns)
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.finditer(capitalized_pattern, text)

        for match in matches:
            entity_text = match.group().strip()

            # Filter out common words
            if len(entity_text) > 2 and not self._is_common_word(entity_text):
                entity = EntityNode(
                    name=entity_text,
                    label="Entity",  # Generic label for pattern-based extraction
                    description=f"Entity extracted from text chunk {chunk_id}",
                    properties={
                        "start_char": match.start(),
                        "end_char": match.end(),
                        "chunk_id": chunk_id,
                        "extraction_method": "pattern_based"
                    },
                    confidence=0.5
                )
                entities.append(entity)

        return entities

    async def _extract_relationships_from_text(self, text: str, entities: list[EntityNode], chunk_id: int) -> list[Relationship]:
        """Extract relationships between entities in text."""
        relationships = []
        entity_names = [e.name for e in entities]

        for pattern in self.relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    source = groups[0].strip()
                    target = groups[-1].strip()

                    # Check if both entities exist in our entity list
                    if source in entity_names and target in entity_names:
                        relationship = Relationship(
                            source=source,
                            target=target,
                            relationship_type=self._classify_relationship_type(match.group(0)),
                            properties={
                                "context": match.group(0),
                                "chunk_id": chunk_id,
                                "pattern_used": pattern
                            },
                            confidence=0.7
                        )
                        relationships.append(relationship)

        return relationships

    def _classify_relationship_type(self, relationship_text: str) -> str:
        """Classify the type of relationship based on the text."""
        text_lower = relationship_text.lower()

        if any(word in text_lower for word in ['said', 'told', 'asked', 'replied', 'answered']):
            return 'SPEAKS_TO'
        elif any(word in text_lower for word in ['met', 'encountered', 'saw', 'visited']):
            return 'MEETS'
        elif any(word in text_lower for word in ['lived', 'resided', 'dwelt']):
            return 'LIVES_IN'
        elif any(word in text_lower for word in ['worked', 'served']):
            return 'WORKS_AT'
        elif any(word in text_lower for word in ['belonged', 'part of', 'member']):
            return 'MEMBER_OF'
        elif any(word in text_lower for word in ['located', 'lies', 'sits']):
            return 'LOCATED_IN'
        elif any(word in text_lower for word in ['traveled', 'went', 'journeyed']):
            return 'TRAVELS_TO'
        elif any(word in text_lower for word in ['created', 'built', 'founded']):
            return 'CREATES'
        elif any(word in text_lower for word in ['owned', 'possessed', 'controlled']):
            return 'OWNS'
        elif any(word in text_lower for word in ['happened', 'occurred', 'took place']):
            return 'HAPPENS_IN'
        elif any(word in text_lower for word in ['caused', 'led to', 'resulted in']):
            return 'CAUSES'
        else:
            return 'RELATED_TO'

    def _is_common_word(self, word: str) -> bool:
        """Check if a word is a common word that shouldn't be treated as an entity."""
        common_words = {
            'The', 'A', 'An', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For',
            'Of', 'With', 'By', 'From', 'Up', 'About', 'Into', 'Through', 'During',
            'Before', 'After', 'Above', 'Below', 'Between', 'Among', 'Under',
            'This', 'That', 'These', 'Those', 'I', 'You', 'He', 'She', 'It',
            'We', 'They', 'Me', 'Him', 'Her', 'Us', 'Them', 'My', 'Your', 'His',
            'Its', 'Our', 'Their', 'Mine', 'Yours', 'Hers', 'Ours', 'Theirs'
        }
        return word in common_words

    async def summarize_nodes_and_relationships(self, entities: list[EntityNode],
                                              relationships: list[Relationship]) -> tuple[list[EntityNode], list[Relationship]]:
        """
        Summarize and deduplicate entities and relationships.
        This is the second stage of GraphRAG pipeline.
        """
        print("📝 Summarizing entities and relationships...")

        # Deduplicate and merge entities
        summarized_entities = await self._summarize_entities(entities)

        # Deduplicate and merge relationships
        summarized_relationships = await self._summarize_relationships(relationships)

        print(f"✅ Summarized to {len(summarized_entities)} entities and {len(summarized_relationships)} relationships")
        return summarized_entities, summarized_relationships

    async def _summarize_entities(self, entities: list[EntityNode]) -> list[EntityNode]:
        """Summarize and deduplicate entities."""
        # Group entities by name (case-insensitive)
        entity_groups = defaultdict(list)

        for entity in entities:
            normalized_name = entity.name.lower().strip()
            entity_groups[normalized_name].append(entity)

        summarized_entities = []

        for _normalized_name, entity_group in entity_groups.items():
            if len(entity_group) == 1:
                # Single entity, use as-is
                summarized_entities.append(entity_group[0])
            else:
                # Multiple entities with same name, merge them
                merged_entity = await self._merge_entities(entity_group)
                summarized_entities.append(merged_entity)

        return summarized_entities

    async def _merge_entities(self, entities: list[EntityNode]) -> EntityNode:
        """Merge multiple entities with the same name."""
        # Use the entity with highest confidence as base
        base_entity = max(entities, key=lambda e: e.confidence)

        # Merge descriptions
        descriptions = [e.description for e in entities if e.description]
        merged_description = " | ".join(descriptions) if descriptions else base_entity.description

        # Merge properties
        merged_properties = base_entity.properties.copy()
        for entity in entities:
            for key, value in entity.properties.items():
                if key not in merged_properties:
                    merged_properties[key] = value
                elif key == "chunk_id":
                    # Collect all chunk IDs
                    if isinstance(merged_properties[key], list):
                        merged_properties[key].append(value)
                    else:
                        merged_properties[key] = [merged_properties[key], value]

        # Calculate average confidence
        avg_confidence = sum(e.confidence for e in entities) / len(entities)

        return EntityNode(
            name=base_entity.name,
            label=base_entity.label,
            description=merged_description,
            properties=merged_properties,
            confidence=avg_confidence
        )

    async def _summarize_relationships(self, relationships: list[Relationship]) -> list[Relationship]:
        """Summarize and deduplicate relationships."""
        # Group relationships by source, target, and type
        relationship_groups = defaultdict(list)

        for rel in relationships:
            key = (rel.source.lower(), rel.target.lower(), rel.relationship_type)
            relationship_groups[key].append(rel)

        summarized_relationships = []

        for _key, rel_group in relationship_groups.items():
            if len(rel_group) == 1:
                # Single relationship, use as-is
                summarized_relationships.append(rel_group[0])
            else:
                # Multiple relationships with same key, merge them
                merged_rel = await self._merge_relationships(rel_group)
                summarized_relationships.append(merged_rel)

        return summarized_relationships

    async def _merge_relationships(self, relationships: list[Relationship]) -> Relationship:
        """Merge multiple relationships with the same source, target, and type."""
        # Use the relationship with highest confidence as base
        base_rel = max(relationships, key=lambda r: r.confidence)

        # Merge properties
        merged_properties = base_rel.properties.copy()
        for rel in relationships:
            for key, value in rel.properties.items():
                if key not in merged_properties:
                    merged_properties[key] = value
                elif key == "chunk_id":
                    # Collect all chunk IDs
                    if isinstance(merged_properties[key], list):
                        merged_properties[key].append(value)
                    else:
                        merged_properties[key] = [merged_properties[key], value]

        # Calculate average confidence
        avg_confidence = sum(r.confidence for r in relationships) / len(relationships)

        return Relationship(
            source=base_rel.source,
            target=base_rel.target,
            relationship_type=base_rel.relationship_type,
            properties=merged_properties,
            confidence=avg_confidence
        )

    async def identify_entity_communities(self, entities: list[EntityNode],
                                        relationships: list[Relationship]) -> list[EntityCommunity]:
        """
        Identify communities of related entities using graph algorithms.
        This is the third stage of GraphRAG pipeline.
        """
        print("🔗 Identifying entity communities...")

        # Build adjacency list
        adjacency = defaultdict(set)
        entity_names = {e.name for e in entities}

        for rel in relationships:
            if rel.source in entity_names and rel.target in entity_names:
                adjacency[rel.source].add(rel.target)
                adjacency[rel.target].add(rel.source)

        # Find connected components (simple community detection)
        visited = set()
        communities = []

        for entity_name in entity_names:
            if entity_name not in visited:
                community_entities = self._dfs_community(entity_name, adjacency, visited)
                if len(community_entities) >= self.community_min_size:
                    community = EntityCommunity(
                        community_id=f"community_{len(communities)}",
                        entities=community_entities,
                        summary="",  # Will be generated later
                        relationships=[r for r in relationships
                                     if r.source in community_entities and r.target in community_entities],
                        confidence=0.8
                    )
                    communities.append(community)

        print(f"✅ Identified {len(communities)} entity communities")
        return communities

    def _dfs_community(self, start_entity: str, adjacency: dict[str, set], visited: set) -> list[str]:
        """Depth-first search to find connected entities."""
        community = []
        stack = [start_entity]

        while stack:
            entity = stack.pop()
            if entity not in visited:
                visited.add(entity)
                community.append(entity)

                # Add neighbors to stack
                for neighbor in adjacency.get(entity, set()):
                    if neighbor not in visited:
                        stack.append(neighbor)

        return community

    async def generate_community_summaries(self, communities: list[EntityCommunity],
                                         original_texts: list[str]) -> list[EntityCommunity]:
        """
        Generate summaries for entity communities.
        This is the final stage of GraphRAG pipeline.
        """
        print("📄 Generating community summaries...")

        for community in communities:
            # Collect all text chunks that mention entities in this community
            relevant_chunks = []

            for entity_name in community.entities:
                for i, text in enumerate(original_texts):
                    if entity_name.lower() in text.lower():
                        relevant_chunks.append({
                            'chunk_id': i,
                            'text': text,
                            'entity': entity_name
                        })

            # Generate summary based on relevant chunks
            if relevant_chunks:
                summary = await self._generate_community_summary(community, relevant_chunks)
                community.summary = summary
            else:
                community.summary = f"Community of {len(community.entities)} related entities: {', '.join(community.entities[:3])}{'...' if len(community.entities) > 3 else ''}"

        print(f"✅ Generated summaries for {len(communities)} communities")
        return communities

    async def _generate_community_summary(self, community: EntityCommunity,
                                        relevant_chunks: list[dict[str, Any]]) -> str:
        """Generate a summary for a community based on relevant text chunks."""
        # Simple summary generation - in a real implementation, this would use an LLM
        entity_names = ', '.join(community.entities[:3])
        if len(community.entities) > 3:
            entity_names += f" and {len(community.entities) - 3} others"

        # Count relationship types
        rel_types = defaultdict(int)
        for rel in community.relationships:
            rel_types[rel.relationship_type] += 1

        # Create summary
        summary_parts = [f"This community includes {entity_names}."]

        if rel_types:
            main_rel_type = max(rel_types, key=rel_types.get)
            summary_parts.append(f"The main relationship type is {main_rel_type.replace('_', ' ').lower()}.")

        summary_parts.append(f"The community has {len(community.relationships)} total relationships.")

        return " ".join(summary_parts)

    async def store_in_neo4j(self, entities: list[EntityNode],
                           relationships: list[Relationship],
                           communities: list[EntityCommunity]) -> bool:
        """Store extracted entities, relationships, and communities in Neo4j."""
        try:
            print("💾 Storing entities, relationships, and communities in Neo4j...")

            # Store entities
            for entity in entities:
                self.graph_service.create_entity_node(
                    name=entity.name,
                    label=entity.label,
                    description=entity.description
                )

            # Store relationships
            for rel in relationships:
                self.graph_service.create_relationship(
                    source_entity=rel.source,
                    source_label="Entity",  # Would need to look up actual label
                    target_entity=rel.target,
                    target_label="Entity",  # Would need to look up actual label
                    relationship_type=rel.relationship_type
                )

            # Store communities (as special nodes)
            for community in communities:
                self.graph_service.create_entity_node(
                    name=community.community_id,
                    label="Community",
                    description=community.summary
                )

                # Link entities to community
                for entity_name in community.entities:
                    self.graph_service.create_relationship(
                        source_entity=entity_name,
                        source_label="Entity",
                        target_entity=community.community_id,
                        target_label="Community",
                        relationship_type="MEMBER_OF"
                    )

            print("✅ Successfully stored in Neo4j")
            return True

        except Exception as e:
            print(f"❌ Error storing in Neo4j: {e}")
            return False

    async def query_with_graph_rag(self, query: str, entities: list[EntityNode],
                                 communities: list[EntityCommunity]) -> dict[str, Any]:
        """
        Query the GraphRAG system using both semantic and graph search.
        This implements the hybrid retrieval approach.
        """
        print(f"🔍 Querying GraphRAG with: {query}")

        # Step 1: Extract entities from query
        query_entities = await self._extract_entities_from_text(query, -1)
        query_entity_names = [e.name for e in query_entities]

        # Step 2: Find relevant entities in our knowledge graph
        relevant_entities = []
        for entity in entities:
            if any(qe.lower() in entity.name.lower() for qe in query_entity_names):
                relevant_entities.append(entity)

        # Step 3: Find communities containing relevant entities
        relevant_communities = []
        for community in communities:
            if any(entity.name in community.entities for entity in relevant_entities):
                relevant_communities.append(community)

        # Step 4: Use Azure Search for semantic search
        semantic_results = []
        try:
            # This would integrate with your existing Azure Search implementation
            # For now, we'll return a placeholder
            semantic_results = [{
                'content': f"Semantic search results for query: {query}",
                'score': 0.8,
                'source': 'azure_search'
            }]
        except Exception as e:
            print(f"Warning: Semantic search failed: {e}")

        # Step 5: Combine results
        result = {
            'query': query,
            'query_entities': [e.name for e in query_entities],
            'relevant_entities': [e.name for e in relevant_entities],
            'relevant_communities': [c.community_id for c in relevant_communities],
            'community_summaries': [c.summary for c in relevant_communities],
            'semantic_results': semantic_results,
            'hybrid_score': 0.8
        }

        print(f"✅ GraphRAG query completed with {len(relevant_entities)} entities and {len(relevant_communities)} communities")
        return result
