from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List, Optional
from viggo.core.services.implementations.graph_service_impl import GraphService, PaginationParams
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
def list_all_nodes(
    label: Optional[str] = Query(None, description="Filter by label: Character, Location, Organization"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of nodes to return"),
    offset: int = Query(0, ge=0, description="Number of nodes to skip"),
    graph_service: GraphService = Depends(get_graph_service)
):
    """List all nodes (entities) in the graph, optionally filtered by label with pagination."""
    try:
        pagination = PaginationParams(limit=limit, offset=offset)
        nodes = graph_service.list_all_nodes(label=label, pagination=pagination)
        # Convert NodeResult objects to dictionaries for response
        node_dicts = [
            {"name": node.name, "labels": node.labels, "properties": node.properties}
            for node in nodes
        ]
        return NodeListResponse(nodes=node_dicts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list nodes: {str(e)}")

@router.get("/grouped_nodes", response_model=GroupedNodeListResponse)
def grouped_nodes(
    label: Optional[str] = Query(None, description="Filter by label: Character, Location, Organization"), 
    graph_service: GraphService = Depends(get_graph_service)
):
    """List all nodes grouped by canonical name, showing all aliases and labels."""
    try:
        grouped_data = graph_service.grouped_nodes(label=label)
        return GroupedNodeListResponse(grouped_nodes=grouped_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to group nodes: {str(e)}")

@router.get("/entity/{entity_name}/aliases")
def get_entity_with_aliases(
    entity_name: str,
    entity_label: Optional[str] = Query(None, description="Filter by label: Character, Location, Organization"),
    graph_service: GraphService = Depends(get_graph_service)
):
    """Get entity data including all its aliases and canonical name."""
    try:
        result = graph_service.get_entity_with_aliases(entity_name, entity_label)
        if not result:
            raise HTTPException(status_code=404, detail="Entity not found or no related data.")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get entity with aliases: {str(e)}")

@router.post("/aliases")
def add_alias_mapping(
    alias: str,
    canonical: str,
    confidence: float = Query(1.0, ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)"),
    source: str = Query("manual", description="Source of the mapping"),
    graph_service: GraphService = Depends(get_graph_service)
):
    """Add a new alias mapping from alias to canonical name."""
    try:
        graph_service.add_alias_mapping(alias, canonical, confidence, source)
        return {"message": f"Added alias mapping: '{alias}' -> '{canonical}'", "confidence": confidence, "source": source}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add alias mapping: {str(e)}")

@router.get("/entity/{entity_name}/suggest-aliases")
def suggest_aliases(
    entity_name: str,
    graph_service: GraphService = Depends(get_graph_service)
):
    """Suggest potential aliases for an entity based on similarity."""
    try:
        suggestions = graph_service.suggest_aliases_for_entity(entity_name)
        return {"entity_name": entity_name, "suggested_aliases": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to suggest aliases: {str(e)}")
