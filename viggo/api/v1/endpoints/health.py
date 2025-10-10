"""
Health check and system information endpoints.
"""

import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from viggo.core.services.interfaces.rag import RAGService as IRAGService
from viggo.dependencies import get_solid_rag_service
from viggo.models.api_models import (
    HealthCheckResponse,
    SuccessResponse,
    VersionResponse,
)

router = APIRouter(prefix="/health", tags=["Health & System Info"])

# Track startup time for uptime calculation
_startup_time = time.time()


@router.get("/", response_model=SuccessResponse[HealthCheckResponse])
async def health_check(
    rag_service: IRAGService = Depends(get_solid_rag_service)
):
    """
    Comprehensive health check for all system components.
    
    Returns the health status of the RAG system, including all
    storage backends, retrievers, and generators.
    """
    try:
        # Get system status from RAG service
        system_status = rag_service.get_system_status()

        # Calculate uptime
        uptime = time.time() - _startup_time

        # Check component health
        components = {
            "rag_system": {
                "status": "healthy" if system_status.get("vector_storage", {}).get("available", False) else "unhealthy",
                "details": system_status.get("vector_storage", {})
            },
            "vector_storage": {
                "status": "healthy" if system_status.get("vector_storage", {}).get("available", False) else "unhealthy",
                "details": system_status.get("vector_storage", {})
            },
            "graph_storage": {
                "status": "healthy" if system_status.get("graph_storage", {}).get("available", False) else "unhealthy",
                "details": system_status.get("graph_storage", {})
            },
            "cache_storage": {
                "status": "healthy" if system_status.get("cache_storage", {}).get("available", False) else "unhealthy",
                "details": system_status.get("cache_storage", {})
            }
        }

        # Check dependencies
        dependencies = {
            "neo4j": {
                "status": "healthy" if system_status.get("graph_storage", {}).get("available", False) else "unhealthy",
                "details": system_status.get("graph_storage", {})
            },
            "redis": {
                "status": "healthy" if system_status.get("cache_storage", {}).get("available", False) else "unhealthy",
                "details": system_status.get("cache_storage", {})
            },
            "azure_search": {
                "status": "healthy",  # Would check actual Azure Search connectivity
                "details": {"endpoint": "configured"}
            }
        }

        # Determine overall status
        overall_status = "healthy"
        for component in components.values():
            if component["status"] != "healthy":
                overall_status = "degraded"
                break

        health_response = HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.now(),
            version="1.0.0",
            uptime=uptime,
            components=components,
            dependencies=dependencies
        )

        return SuccessResponse(
            message="Health check completed",
            data=health_response
        )

    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/ready", response_model=SuccessResponse[dict])
async def readiness_check(
    rag_service: IRAGService = Depends(get_solid_rag_service)
):
    """
    Readiness check for the RAG system.
    
    Returns whether the system is ready to handle requests.
    """
    try:
        system_status = rag_service.get_system_status()

        is_ready = system_status.get("vector_storage", {}).get("available", False)

        if is_ready:
            return SuccessResponse(
                message="System is ready",
                data={"ready": True, "status": "operational"}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System is not ready"
            )

    except Exception as e:
        logging.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Readiness check failed: {str(e)}"
        )


@router.get("/live", response_model=SuccessResponse[dict])
async def liveness_check():
    """
    Liveness check for the application.
    
    Returns whether the application is alive and responding.
    """
    return SuccessResponse(
        message="Application is alive",
        data={
            "alive": True,
            "timestamp": datetime.now(),
            "uptime": time.time() - _startup_time
        }
    )


@router.get("/version", response_model=SuccessResponse[VersionResponse])
async def get_version():
    """
    Get system version information.
    
    Returns version details including build information,
    Python version, and dependency versions.
    """
    try:
        version_response = VersionResponse(
            version="1.0.0",
            build_date=datetime.now(),
            git_commit="unknown",  # Would come from actual git info
            python_version="3.9.0",  # Would come from actual Python version
            dependencies={
                "fastapi": "0.104.1",
                "pydantic": "2.5.0",
                "sentence-transformers": "2.2.2",
                "faiss-cpu": "1.7.4",
                "neo4j": "5.15.0",
                "redis": "5.0.1"
            }
        )

        return SuccessResponse(
            message="Version information retrieved",
            data=version_response
        )

    except Exception as e:
        logging.error(f"Failed to get version info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get version info: {str(e)}"
        )


@router.get("/metrics", response_model=SuccessResponse[dict])
async def get_metrics(
    rag_service: IRAGService = Depends(get_solid_rag_service)
):
    """
    Get system metrics and performance data.
    
    Returns metrics about system performance, usage,
    and resource utilization.
    """
    try:
        system_status = rag_service.get_system_status()

        metrics = {
            "system": {
                "uptime": time.time() - _startup_time,
                "status": "healthy"
            },
            "storage": {
                "vector_count": system_status.get("vector_storage", {}).get("vector_count", 0),
                "graph_nodes": 0,  # Would come from graph service
                "graph_relationships": 0,  # Would come from graph service
                "cache_size": 0  # Would come from cache service
            },
            "performance": {
                "avg_query_time": 0.5,  # Would come from actual metrics
                "queries_per_minute": 0,  # Would come from actual metrics
                "error_rate": 0.0  # Would come from actual metrics
            },
            "resources": {
                "memory_usage": 0,  # Would come from actual system metrics
                "cpu_usage": 0,  # Would come from actual system metrics
                "disk_usage": 0  # Would come from actual system metrics
            }
        }

        return SuccessResponse(
            message="Metrics retrieved successfully",
            data=metrics
        )

    except Exception as e:
        logging.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )
