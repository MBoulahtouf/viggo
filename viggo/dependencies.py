from fastapi import Depends
from viggo.core.services import get_rag_service
from viggo.core.services.interfaces.rag import RAGService as IRAGService
from viggo.core.services.implementations.graph_service_impl import GraphService
from viggo.core.config import settings

# SOLID-compliant dependencies
def get_solid_rag_service() -> IRAGService:
    """Get the SOLID-compliant RAG service."""
    return get_rag_service()

def get_graph_service() -> GraphService:
    """Get the graph service."""
    return GraphService(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, clear_on_startup=False)