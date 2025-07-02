from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List
from viggo.core.services.graph_service import GraphService
from viggo.dependencies import get_graph_service
from viggo.models.schemas import NodeListResponse, GroupedNodeListResponse, EntityGraphResponse

router = APIRouter()

@router.get("/entity/{entity_name}", response_model=EntityGraphResponse)
async def get_entity_graph_data(
    entity_name: str,
    entity_label: str = "", # Optional: to specify the type of entity (e.g., Character, Location)
    excluded_rel_types: List[str] = Query(None), # Optional: list of relationship types to exclude
    excluded_node_labels: List[str] = Query(None), # Optional: list of node labels to exclude
    graph_service: GraphService = Depends(get_graph_service)
) -> EntityGraphResponse:
    """
    Retrieves graph data for a given entity, including its properties and direct relationships.
    """
    graph_data = graph_service.get_related_info_for_entity(
        entity_name, 
        entity_label, 
        excluded_rel_types=excluded_rel_types, 
        excluded_node_labels=excluded_node_labels
    )
    if not graph_data:
        raise HTTPException(status_code=404, detail="Entity not found or no related data.")
    return EntityGraphResponse(entity_name=entity_name, graph_data=graph_data)

@router.get("/nodes", response_model=NodeListResponse)
def list_all_nodes(label: str = Query(None, description="Filter by label: Character, Location, Organization"), graph_service: GraphService = Depends(get_graph_service)):
    """List all nodes (entities) in the graph, optionally filtered by label."""
    return NodeListResponse(nodes=graph_service.list_all_nodes(label=label))

@router.get("/grouped_nodes", response_model=GroupedNodeListResponse)
def grouped_nodes(label: str = Query(None, description="Filter by label: Character, Location, Organization"), graph_service: GraphService = Depends(get_graph_service)):
    """List all nodes grouped by canonical name, showing all aliases and labels."""
    return GroupedNodeListResponse(grouped_nodes=graph_service.grouped_nodes(label=label))
