from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.http_client import close_http_client, get_http_client
from app.core.logging import RequestContextMiddleware, setup_logging
from app.core.rate_limit import RateLimitMiddleware


def custom_generate_unique_id(route: APIRoute) -> str:
    """Produce concise operation IDs for OpenAPI TypeScript SDK generation."""
    if route.tags:
        return f"{route.tags[0]}_{route.name}"
    return route.name


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize and clean up shared infrastructure resources."""
    setup_logging()
    get_http_client()
    yield
    await close_http_client()


def create_app() -> FastAPI:
    """Application factory for Analog Terrax FastAPI backend."""
    settings = get_settings()

    is_production = settings.ENVIRONMENT == "production"

    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Scalable FastAPI backend infrastructure for Analog Terrax.",
        openapi_url="/openapi.json",
        docs_url=None if is_production else "/docs",
        redoc_url=None if is_production else "/redoc",
        generate_unique_id_function=custom_generate_unique_id,
        lifespan=lifespan,
    )

    application.add_middleware(
        RateLimitMiddleware,
        limit_per_minute=settings.RATE_LIMIT_PER_MINUTE,
        protected_prefixes=(
            f"{settings.API_V1_STR}/analogs/search",
            f"{settings.API_V1_STR}/analogs/heatmap",
        ),
    )
    application.add_middleware(RequestContextMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_origin_regex=settings.CORS_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )

    register_exception_handlers(application)
    application.include_router(api_v1_router, prefix=settings.API_V1_STR)

    return application


app = create_app()
