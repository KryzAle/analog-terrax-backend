from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Standardized error detail block."""

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error description")
    details: Any | None = Field(default=None, description="Additional error context")
    request_id: str | None = Field(default=None, description="Trace X-Request-ID")


class ErrorResponse(BaseModel):
    """Envelope returned for all HTTP 4xx/5xx responses."""

    error: ErrorDetail


class APIResponse[DataT](BaseModel):
    """Generic envelope for successful API responses."""

    data: DataT
    meta: dict[str, Any] = Field(default_factory=dict)
