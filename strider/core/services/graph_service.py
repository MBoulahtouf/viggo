# strider/core/services/graph_service.py
import spacy
from neo4j import GraphDatabase
from strider.core.config import settings

# Initialize model here
nlp_model = spacy.load("en_core_web_sm")

def get_db_driver():
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))

def add_node(tx, label, name):
    tx.run(f"MERGE (n:{label} {{name: $name}})", name=name)

def add_relationship(tx, name1, name2, relationship="RELATED_TO"):
    query = ("MATCH (a {name: $name1}), (b {name: $name2}) WHERE a <> b MERGE (a)-[r:%s]->(b)" % relationship)
    tx.run(query, name1=name1, name2=name2)

def extract_and_load_graph(text_data: list):
    driver = get_db_driver()
    nodes_added, rels_added = 0, 0
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        for page in text_data:
            doc = nlp_model(page['content'])
            entities = [(ent.text.strip(), ent.label_) for ent in doc.ents if ent.label_ in ["PERSON", "GPE", "ORG", "LOC"]]
            for name, label in set(entities):
                if len(name) > 2:
                    session.execute_write(add_node, label, name)
                    nodes_added += 1
            for sentence in doc.sents:
                sent_entities = [ent.text.strip() for ent in sentence.ents if ent.label_ in ["PERSON", "GPE", "ORG", "LOC"]]
                if len(sent_entities) > 1:
                    for i in range(len(sent_entities)):
                        for j in range(i + 1, len(sent_entities)):
                            session.execute_write(add_relationship, sent_entities[i], sent_entities[j])
                            rels_added += 1
    driver.close()
    return nodes_added, rels_added
