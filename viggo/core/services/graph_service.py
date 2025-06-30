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
        name = name.strip()
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

    def create_relationship(self, source_entity: str, target_entity: str, relationship_type: str):
        with self.driver.session() as session:
            query = (
                "MATCH (a), (b) "
                "WHERE a.name = $source_entity AND b.name = $target_entity "
                f"MERGE (a)-[:{relationship_type}]->(b)"
            )
            session.run(query, source_entity=source_entity, target_entity=target_entity)

    def extract_and_load_graph(self, filename: str, processed_chunks_with_metadata: List[Dict]):
        doc_filename = self.create_document_node(filename, filename) # Using filename as path for simplicity

        for i, chunk_data in enumerate(processed_chunks_with_metadata):
            page_number = chunk_data.get("page")
            content = chunk_data.get("content")
            entities = chunk_data.get("entities", [])
            relationships = chunk_data.get("relationships", [])
            chunk_id = f"{doc_filename}_page{page_number}_chunk{i}"

            self.create_page_node(doc_filename, page_number)
            self.create_chunk_node(page_number, chunk_id, content)

            for entity in entities:
                entity_name = entity["text"]
                entity_label = entity["label"]
                self.create_entity_node(entity_name, entity_label, "")
                self.link_chunk_to_entity(chunk_id, entity_name, entity_label)

            for rel in relationships:
                self.create_relationship(rel["source"], rel["target"], rel["type"])

    def get_related_info_for_entity(self, entity_name: str, entity_label: str = "") -> List[Dict[str, Any]]:
        print(f"[DEBUG] get_related_info_for_entity called with entity_name='{entity_name}', entity_label='{entity_label}'")
        with self.driver.session() as session:
            query_parts = [
                f"MATCH (e) WHERE toLower(e.name) CONTAINS toLower($entity_name)"
            ]
            if entity_label:
                query_parts.append(f"AND '{entity_label}' IN labels(e)")
            
            query_parts.append(
                "OPTIONAL MATCH (e)-[r]-(n) "
                "RETURN e.name AS entityName, labels(e) AS entityLabels, properties(e) AS entityProperties, "
                "type(r) AS relationshipType, properties(r) AS relationshipProperties, "
                "n.name AS relatedNodeName, labels(n) AS relatedNodeLabels, properties(n) AS relatedNodeProperties "
                "LIMIT 10"
            )
            cypher_query = " ".join(query_parts)
            print(f"[DEBUG] get_related_info_for_entity Cypher query: {cypher_query}")
            result = session.run(cypher_query, entity_name=entity_name)
            
            structured_info = []
            for record in result:
                info = {
                    "entity": {
                        "name": record["entityName"],
                        "labels": record["entityLabels"],
                        "properties": record["entityProperties"]
                    }
                }
                if record["relationshipType"]:
                    info["relationship"] = {
                        "type": record["relationshipType"],
                        "properties": record["relationshipProperties"]
                    }
                    info["related_node"] = {
                        "name": record["relatedNodeName"],
                        "labels": record["relatedNodeLabels"],
                        "properties": record["relatedNodeProperties"]
                    }
                structured_info.append(info)
            print(f"[DEBUG] get_related_info_for_entity result: {structured_info}")
            return structured_info

    def get_entity_graph_data(self, entity_name: str, entity_label: str = "") -> Dict[str, Any]:
        print(f"[DEBUG] get_entity_graph_data called with entity_name='{entity_name}', entity_label='{entity_label}'")
        with self.driver.session() as session:
            entity_result = None
            found_entity_name = None
            found_entity_labels = []

            # Build the initial node lookup query
            node_lookup_query_parts = [
                "MATCH (e)"
            ]
            if entity_label == "Character":
                node_lookup_query_parts.append("WHERE (toLower(e.name) CONTAINS toLower($entity_name) OR toLower(e.first_name) CONTAINS toLower($entity_name)) AND 'Character' IN labels(e)")
            elif entity_label:
                node_lookup_query_parts.append(f"WHERE toLower(e.name) CONTAINS toLower($entity_name) AND '{entity_label}' IN labels(e)")
            else:
                node_lookup_query_parts.append("WHERE toLower(e.name) CONTAINS toLower($entity_name)")
            node_lookup_query_parts.append("RETURN properties(e) AS properties, labels(e) AS labels, e.name AS name")
            node_lookup_query = " ".join(node_lookup_query_parts)

            print(f"[DEBUG] Node lookup query: {node_lookup_query}")
            entity_result = session.run(node_lookup_query, entity_name=entity_name).single()
            print(f"[DEBUG] Node lookup result: {entity_result}")

            if not entity_result:
                print(f"[DEBUG] No entity found for '{entity_name}' with label '{entity_label}'.")
                return {}

            found_entity_name = entity_result["name"]
            found_entity_labels = entity_result["labels"]

            entity_data = {
                "name": found_entity_name,
                "labels": found_entity_labels,
                "properties": entity_result["properties"],
                "relationships": []
            }

            # Get direct relationships using the found entity's exact name and labels
            relationships_query_str = (
                "MATCH (e) WHERE e.name = $found_entity_name AND any(label IN labels(e) WHERE label IN $found_entity_labels) "
                "MATCH (e)-[r]-(n) "
                "RETURN type(r) AS relationshipType, properties(r) AS relationshipProperties, "
                "labels(n) AS targetNodeLabels, properties(n) AS targetNodeProperties, n.name AS targetNodeName"
            )
            print(f"[DEBUG] Relationships query: {relationships_query_str}")
            relationships_result = session.run(relationships_query_str, found_entity_name=found_entity_name, found_entity_labels=found_entity_labels)

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