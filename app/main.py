from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse
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


def inject_web_analytics(html_content: str) -> str:
    """Inject Vercel Web Analytics script into HTML content.
    
    Args:
        html_content: HTML string to inject analytics into
        
    Returns:
        HTML string with analytics script injected before closing head tag
    """
    analytics_script = '<script defer src="/_vercel/insights/script.js"></script>'
    return html_content.replace("</head>", f"{analytics_script}</head>")


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
        docs_url=None,
        redoc_url=None,
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

    # Custom documentation routes with Vercel Web Analytics integration
    # Documentation is available in non-production environments
    if not is_production:

        @application.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html() -> HTMLResponse:
            """Swagger UI documentation with Vercel Web Analytics."""
            html = get_swagger_ui_html(
                openapi_url=application.openapi_url or "/openapi.json",
                title=f"{application.title} - Swagger UI",
                oauth2_redirect_url=application.swagger_ui_oauth2_redirect_url,
                swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
                swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
            )
            html_body = html.body if isinstance(html.body, bytes) else bytes(html.body)
            html_content = html_body.decode("utf-8")
            
            # Inject Vercel Web Analytics if enabled
            if settings.ENABLE_WEB_ANALYTICS:
                html_content = inject_web_analytics(html_content)
            
            return HTMLResponse(content=html_content)

        @application.get("/redoc", include_in_schema=False)
        async def custom_redoc_html() -> HTMLResponse:
            """ReDoc documentation with Vercel Web Analytics."""
            html = get_redoc_html(
                openapi_url=application.openapi_url or "/openapi.json",
                title=f"{application.title} - ReDoc",
                redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
            )
            html_body = html.body if isinstance(html.body, bytes) else bytes(html.body)
            html_content = html_body.decode("utf-8")
            
            # Inject Vercel Web Analytics if enabled
            if settings.ENABLE_WEB_ANALYTICS:
                html_content = inject_web_analytics(html_content)
            
            return HTMLResponse(content=html_content)

    return application


app = create_app()
