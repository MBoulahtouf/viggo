# viggo/api/v1/router.py
from fastapi import APIRouter
from viggo.api.v1.endpoints import graph, rag, content, health

api_router = APIRouter()

# Health and system endpoints (always available)
api_router.include_router(health.router, tags=["Health & System Info"])

# SOLID-compliant endpoints
api_router.include_router(rag.router, tags=["RAG Operations"])
api_router.include_router(content.router, tags=["Content Processing"])
api_router.include_router(graph.router, prefix="/graph", tags=["Graph Exploration"])
