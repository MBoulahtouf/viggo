from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from viggo.core.services.graph_service import GraphService
from viggo.dependencies import get_graph_service

router = APIRouter()

@router.get("/entity/{entity_name}")
async def get_entity_graph_data(
    entity_name: str,
    entity_label: str = "", # Optional: to specify the type of entity (e.g., Character, Location)
    graph_service: GraphService = Depends(get_graph_service)
) -> Dict[str, Any]:
    """
    Retrieves graph data for a given entity, including its properties and direct relationships.
    """
    graph_data = graph_service.get_entity_graph_data(entity_name, entity_label)
    if not graph_data:
        raise HTTPException(status_code=404, detail="Entity not found or no related data.")
    return {"entity_name": entity_name, "graph_data": graph_data}
