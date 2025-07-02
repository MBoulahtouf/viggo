import os

def clear_indexes_and_graph(rag_service, graph_service):
    """Remove old index files and clear the Neo4j graph."""
    graph_service.clear_database()
    index_path = getattr(rag_service, 'index_path', 'faiss_index.bin')
    doc_data_path = getattr(rag_service, 'doc_data_path', 'document_data.pkl')
    for path in [index_path, doc_data_path]:
        if os.path.exists(path):
            os.remove(path) 