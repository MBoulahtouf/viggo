# viggo/api/v1/router.py
from fastapi import APIRouter
from viggo.api.v1.endpoints import document, query, graph, rag, content, health

api_router = APIRouter()

# Health and system endpoints (always available)
api_router.include_router(health.router, tags=["Health & System Info"])

# Legacy endpoints (maintained for backward compatibility)
api_router.include_router(document.router, prefix="/documents", tags=["Document Processing (Legacy)"])
api_router.include_router(query.router, prefix="/query", tags=["Q&A (Legacy)"])
api_router.include_router(graph.router, prefix="/graph", tags=["Graph Exploration"])

# New SOLID-compliant endpoints
api_router.include_router(rag.router, tags=["RAG Operations (New)"])
api_router.include_router(content.router, tags=["Content Processing (New)"])
