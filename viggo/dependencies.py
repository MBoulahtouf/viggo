from fastapi import Depends
from viggo.core.services import (
    get_rag_service as get_new_rag_service,
    get_legacy_compatible_service,
    GraphService,  # Legacy service
    RAGService     # Legacy service
)
from viggo.core.services.interfaces.rag import RAGService as IRAGService
from viggo.core.config import settings

# Legacy dependencies (maintained for backward compatibility)
def get_graph_service() -> GraphService:
    return GraphService(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, clear_on_startup=False)

def get_rag_service(graph_service: GraphService = Depends(get_graph_service)) -> RAGService:
    return RAGService(graph_service=graph_service)

# New SOLID-compliant dependencies
def get_solid_rag_service() -> IRAGService:
    """Get the new SOLID-compliant RAG service."""
    return get_new_rag_service()

def get_legacy_rag_service():
    """Get a legacy-compatible RAG service wrapper."""
    return get_legacy_compatible_service()