# viggo/api/v1/router.py
from fastapi import APIRouter
from viggo.api.v1.endpoints import document, query, graph

api_router = APIRouter()
api_router.include_router(document.router, prefix="/documents", tags=["Document Processing"])
api_router.include_router(query.router, prefix="/query", tags=["Q&A"])
api_router.include_router(graph.router, prefix="/graph", tags=["Graph Exploration"])
