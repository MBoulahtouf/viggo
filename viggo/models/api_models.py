"""
API response models and utilities for the Viggo system.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Generic, TypeVar
from datetime import datetime
from enum import Enum

T = TypeVar('T')


class ResponseStatus(str, Enum):
    """Response status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class BaseResponse(BaseModel, Generic[T]):
    """Base response model."""
    status: ResponseStatus = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    data: Optional[T] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


class SuccessResponse(BaseResponse[T]):
    """Success response model."""
    status: ResponseStatus = Field(ResponseStatus.SUCCESS, description="Response status")
    data: T = Field(..., description="Response data")


class ErrorResponse(BaseResponse[None]):
    """Error response model."""
    status: ResponseStatus = Field(ResponseStatus.ERROR, description="Response status")
    error_code: str = Field(..., description="Error code")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    data: None = Field(None, description="No data for error responses")


class WarningResponse(BaseResponse[T]):
    """Warning response model."""
    status: ResponseStatus = Field(ResponseStatus.WARNING, description="Response status")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    data: Optional[T] = Field(None, description="Response data")


class InfoResponse(BaseResponse[T]):
    """Info response model."""
    status: ResponseStatus = Field(ResponseStatus.INFO, description="Response status")
    data: Optional[T] = Field(None, description="Response data")


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Page size")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: str = Field("asc", description="Sort order: asc or desc")
    
    @property
    def offset(self) -> int:
        """Calculate offset from page and page_size."""
        return (self.page - 1) * self.page_size


class PaginationInfo(BaseModel):
    """Pagination information."""
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response model."""
    items: List[T] = Field(..., description="List of items")
    pagination: PaginationInfo = Field(..., description="Pagination information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    version: str = Field(..., description="System version")
    uptime: float = Field(..., description="System uptime in seconds")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health status")
    dependencies: Dict[str, Dict[str, Any]] = Field(..., description="Dependency health status")


class VersionResponse(BaseModel):
    """Version response model."""
    version: str = Field(..., description="System version")
    build_date: datetime = Field(..., description="Build date")
    git_commit: Optional[str] = Field(None, description="Git commit hash")
    python_version: str = Field(..., description="Python version")
    dependencies: Dict[str, str] = Field(..., description="Dependency versions")


class ValidationError(BaseModel):
    """Validation error model."""
    field: str = Field(..., description="Field name")
    message: str = Field(..., description="Error message")
    value: Any = Field(..., description="Invalid value")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response model."""
    error_code: str = Field("VALIDATION_ERROR", description="Error code")
    validation_errors: List[ValidationError] = Field(..., description="Validation errors")


class NotFoundErrorResponse(ErrorResponse):
    """Not found error response model."""
    error_code: str = Field("NOT_FOUND", description="Error code")
    resource_type: str = Field(..., description="Type of resource not found")
    resource_id: str = Field(..., description="ID of resource not found")


class ConflictErrorResponse(ErrorResponse):
    """Conflict error response model."""
    error_code: str = Field("CONFLICT", description="Error code")
    conflict_type: str = Field(..., description="Type of conflict")
    conflicting_resource: str = Field(..., description="Conflicting resource")


class RateLimitErrorResponse(ErrorResponse):
    """Rate limit error response model."""
    error_code: str = Field("RATE_LIMIT_EXCEEDED", description="Error code")
    retry_after: int = Field(..., description="Seconds to wait before retrying")
    limit: int = Field(..., description="Rate limit")
    remaining: int = Field(..., description="Remaining requests")


class InternalServerErrorResponse(ErrorResponse):
    """Internal server error response model."""
    error_code: str = Field("INTERNAL_SERVER_ERROR", description="Error code")
    request_id: str = Field(..., description="Request identifier for tracking")


class APIResponse(BaseModel):
    """Generic API response wrapper."""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    errors: Optional[List[str]] = Field(None, description="Error messages")
    warnings: Optional[List[str]] = Field(None, description="Warning messages")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


class BatchRequest(BaseModel):
    """Batch request model."""
    requests: List[Dict[str, Any]] = Field(..., description="List of requests")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    fail_fast: bool = Field(False, description="Stop on first error")
    max_concurrent: int = Field(5, ge=1, le=20, description="Maximum concurrent requests")


class BatchResponse(BaseModel):
    """Batch response model."""
    batch_id: str = Field(..., description="Batch identifier")
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    results: List[Dict[str, Any]] = Field(..., description="Request results")
    processing_time: float = Field(..., description="Total processing time in seconds")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error details")
