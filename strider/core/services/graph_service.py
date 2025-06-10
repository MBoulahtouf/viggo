# strider/core/services/graph_service.py

import spacy
from neo4j import GraphDatabase
from strider.core.config import settings
from collections import Counter

nlp_model = spacy.load("en_core_web_sm")

def get_db_driver():
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))

def add_node(tx, label, name):
    tx.run(f"MERGE (n:{label} {{name: $name}})", name=name)

def add_relationship(tx, name1, label1, name2, label2, relationship):
    rel_type = relationship.upper().replace("'", "")
    query = (
        f"MATCH (a:{label1} {{name: $name1}}), (b:{label2} {{name: $name2}}) "
        "WHERE a <> b "
        "MERGE (a)-[r:%s]->(b)" % rel_type
    )
    tx.run(query, name1=name1, name2=name2)

def extract_graph_and_features(text_data: list):
    """
    A single, fast pass to extract the knowledge graph AND other features 
    like character journeys.
    """
    driver = get_db_driver()
    
    # --- Feature 1: Character Journey dictionary ---
    character_journeys = {}

    # --- Entity Pre-processing ---
    all_entities = [ent.text.strip() for page in text_data for ent in nlp_model(page['content']).ents if len(ent.text) > 2 and not ent.text.isupper()]
    entity_counts = Counter(all_entities)
    valid_entities = {name for name, count in entity_counts.items() if count < 50}
    
    nodes_added, rels_added = 0, 0
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        for page in text_data:
            doc = nlp_model(page['content'])
            page_entities = {}
            for ent in doc.ents:
                name = ent.text.strip()
                if name in valid_entities:
                    label = ent.label_
                    # Populate page_entities for relationship extraction
                    page_entities[name] = label
                    
                    # If this is a person, track their journey
                    if label == 'PERSON':
                        if name not in character_journeys:
                            character_journeys[name] = []
                        character_journeys[name].append(page['page'])

            # Add nodes for this page's valid entities
            for name, label in page_entities.items():
                session.execute_write(add_node, label, name)
                nodes_added += 1

            # Dependency Parsing for Relationships
            for sent in doc.sents:
                for token in sent:
                    if token.pos_ == "VERB":
                        subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                        objects = [child for child in token.children if child.dep_ in ("dobj", "pobj")]
                        if subjects and objects:
                            for subj_token in subjects:
                                for obj_token in objects:
                                    subj = subj_token.text.strip()
                                    obj = obj_token.text.strip()
                                    if subj in page_entities and obj in page_entities:
                                        subj_label, obj_label = page_entities[subj], page_entities[obj]
                                        if subj_label == 'PERSON':
                                            relationship = token.lemma_
                                            session.execute_write(add_relationship, subj, subj_label, obj, obj_label, relationship)
                                            rels_added += 1
    driver.close()
    
    # Clean up journeys to have unique, sorted page numbers
    for name, pages in character_journeys.items():
        character_journeys[name] = sorted(list(set(pages)))
        
    return nodes_added, rels_added, character_journeys

