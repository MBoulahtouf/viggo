# viggo/core/services/graph_service.py
from neo4j import GraphDatabase
from typing import List, Dict, Any

class GraphService:
    def __init__(self, uri, user, password, clear_on_startup: bool = False):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        if clear_on_startup:
            self.clear_database()

    def close(self):
        self.driver.close()

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_document_node(self, filename: str, path: str) -> str:
        with self.driver.session() as session:
            query = (
                "MERGE (d:Document {filename: $filename}) "
                "ON CREATE SET d.path = $path "
                "RETURN d.filename AS filename"
            )
            session.run(query, filename=filename, path=path)
            return filename

    def create_page_node(self, document_filename: str, page_number: int) -> int:
        with self.driver.session() as session:
            query = (
                "MATCH (d:Document {filename: $document_filename}) "
                "MERGE (p:Page {page_number: $page_number}) "
                "MERGE (d)-[:HAS_PAGE]->(p) "
                "RETURN p.page_number AS page_number"
            )
            session.run(query, document_filename=document_filename, page_number=page_number)
            return page_number

    def create_chunk_node(self, page_number: int, chunk_id: str, content: str) -> str:
        with self.driver.session() as session:
            query = (
                "MATCH (p:Page {page_number: $page_number}) "
                "MERGE (c:Chunk {chunk_id: $chunk_id}) "
                "ON CREATE SET c.content = $content "
                "MERGE (p)-[:HAS_CHUNK]->(c) "
                "RETURN c.chunk_id AS chunk_id"
            )
            session.run(query, page_number=page_number, chunk_id=chunk_id, content=content)
            return chunk_id

    def create_entity_node(self, name: str, label: str, description: str = "") -> str:
        print(f"[DEBUG] Creating entity node: Name='{name}', Label='{label}'")
        with self.driver.session() as session:
            if label == "PERSON":
                first_name = name.split(" ")[0] if " " in name else name
                query = (
                    "MERGE (e:Character {name: $name}) "
                    "ON CREATE SET e.description = $description, e.first_name = $first_name "
                    "RETURN e.name AS name"
                )
                session.run(query, name=name, description=description, first_name=first_name)
            else:
                query = (
                    f"MERGE (e:{label} {{name: $name}}) "
                    "RETURN e.name AS name"
                )
                session.run(query, name=name)
            return name

    def link_chunk_to_entity(self, chunk_id: str, entity_name: str, entity_label: str):
        print(f"[DEBUG] Linking chunk '{chunk_id}' to entity '{entity_name}' ({entity_label})")
        with self.driver.session() as session:
            query = (
                f"MATCH (c:Chunk {{chunk_id: $chunk_id}}) "
                f"MATCH (e:{entity_label} {{name: $entity_name}}) "
                f"MERGE (c)-[:MENTIONS_{entity_label.upper()}]->(e)"
            )
            session.run(query, chunk_id=chunk_id, entity_name=entity_name)

    def extract_and_load_graph(self, filename: str, processed_chunks_with_metadata: List[Dict]):
        doc_filename = self.create_document_node(filename, filename) # Using filename as path for simplicity

        for i, chunk_data in enumerate(processed_chunks_with_metadata):
            page_number = chunk_data.get("page")
            content = chunk_data.get("content")
            entities = chunk_data.get("entities", [])
            chunk_id = f"{doc_filename}_page{page_number}_chunk{i}"

            self.create_page_node(doc_filename, page_number)
            self.create_chunk_node(page_number, chunk_id, content)

            for entity in entities:
                entity_name = entity["text"]
                entity_label = entity["label"]
                self.create_entity_node(entity_name, entity_label, "")
                self.link_chunk_to_entity(chunk_id, entity_name, entity_label)

    def get_related_info_for_entity(self, entity_name: str, entity_label: str) -> str:
        with self.driver.session() as session:
            query = (
                f"MATCH (e:{entity_label} {{name: $entity_name}})-[r]-(n) "
                "RETURN e.name AS entityName, labels(e) AS entityLabels, type(r) AS relationshipType, n.name AS relatedNodeName, labels(n) AS relatedNodeLabels "
                "LIMIT 5"
            )
            result = session.run(query, entity_name=entity_name)
            
            info_parts = []
            for record in result:
                info_parts.append(
                    f"  - {record["entityName"]} ({', '.join(record["entityLabels"])}) "
                    f"{record["relationshipType"]} {record["relatedNodeName"]} ({', '.join(record["relatedNodeLabels"])})"
                )
            return "\n".join(info_parts)

    def get_entity_graph_data(self, entity_name: str, entity_label: str = "") -> Dict[str, Any]:
        print(f"[DEBUG] get_entity_graph_data called with entity_name='{entity_name}', entity_label='{entity_label}'")
        with self.driver.session() as session:
            entity_result = None
            found_entity_name = None

            # Prioritize Character search if entity_label is Character
            if entity_label == "Character":
                character_query = (
                    "MATCH (e:Character) "
                    "WHERE toLower(e.name) CONTAINS toLower($entity_name) OR toLower(e.first_name) CONTAINS toLower($entity_name) "
                    "RETURN properties(e) AS properties, labels(e) AS labels, e.name AS name"
                )
                print(f"[DEBUG] Attempting Character-specific query: {character_query}")
                entity_result = session.run(character_query, entity_name=entity_name).single()
                if entity_result:
                    found_entity_name = entity_result["name"]
                    print(f"[DEBUG] Character-specific query result: {entity_result}")

            # General search fallback if no Character found or different label
            if not entity_result and entity_label != "Character":
                general_query = f"MATCH (e:{entity_label}) WHERE toLower(e.name) CONTAINS toLower($entity_name) RETURN properties(e) AS properties, labels(e) AS labels, e.name AS name"
                print(f"[DEBUG] Attempting general query with label: {general_query}")
                entity_result = session.run(general_query, entity_name=entity_name).single()
                if entity_result:
                    found_entity_name = entity_result["name"]
                    print(f"[DEBUG] General query with label result: {entity_result}")

            # Final fallback: search any node with CONTAINS
            if not entity_result:
                any_label_query = f"MATCH (e) WHERE toLower(e.name) CONTAINS toLower($entity_name) RETURN properties(e) AS properties, labels(e) AS labels, e.name AS name"
                print(f"[DEBUG] Attempting any label CONTAINS query: {any_label_query}")
                entity_result = session.run(any_label_query, entity_name=entity_name).single()
                if entity_result:
                    found_entity_name = entity_result["name"]
                    print(f"[DEBUG] Any label CONTAINS query result: {entity_result}")

            if not entity_result:
                print(f"[DEBUG] No entity found for '{entity_name}' with label '{entity_label}'.")
                return {}

            entity_data = {
                "name": found_entity_name,
                "labels": entity_result["labels"],
                "properties": entity_result["properties"],
                "relationships": []
            }

            # Get direct relationships (using the found entity's name for consistency)
            relationships_query_str = (
                f"MATCH (e:{entity_label}) WHERE toLower(e.name) = toLower($found_entity_name) MATCH (e)-[r]-(n) "
                "RETURN type(r) AS relationshipType, properties(r) AS relationshipProperties, "
                "labels(n) AS targetNodeLabels, properties(n) AS targetNodeProperties, n.name AS targetNodeName"
            )
            print(f"[DEBUG] Relationships query: {relationships_query_str}")
            relationships_result = session.run(relationships_query_str, found_entity_name=found_entity_name)

            for record in relationships_result:
                entity_data["relationships"].append({
                    "type": record["relationshipType"],
                    "properties": record["relationshipProperties"],
                    "target_node": {
                        "name": record["targetNodeName"],
                        "labels": record["targetNodeLabels"],
                        "properties": record["targetNodeProperties"]
                    }
                })
            print(f"[DEBUG] Entity data returned: {entity_data}")
            return entity_data