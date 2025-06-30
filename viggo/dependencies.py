from fastapi import Depends
from viggo.core.services.rag_service import RAGService
from viggo.core.services.graph_service import GraphService
from viggo.core.config import settings

def get_graph_service() -> GraphService:
    return GraphService(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, clear_on_startup=False)

def get_rag_service(graph_service: GraphService = Depends(get_graph_service)) -> RAGService:
    return RAGService(graph_service=graph_service)