from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from app.api.deps import SettingsDep
from app.core.cache import cache_control
from app.schemas.health import HealthCheckResponse

router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="System health and readiness check",
    dependencies=[Depends(cache_control(s_maxage=30, stale_while_revalidate=60))],
)
async def get_health(settings: SettingsDep) -> HealthCheckResponse:
    """Return service health metadata for monitoring and frontend status checks."""
    return HealthCheckResponse(
        status="ok",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(UTC),
    )
