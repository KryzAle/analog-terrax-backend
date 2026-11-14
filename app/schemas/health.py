from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """System health and readiness status."""

    status: Literal["ok", "degraded"] = Field(default="ok", description="Service health state")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Deployed semantic version")
    environment: str = Field(..., description="Runtime environment name")
    timestamp: datetime = Field(..., description="UTC server timestamp")
