# Analog Terrax — Backend (`analog-terrax-backend`)

Scalable, stateless FastAPI backend infrastructure for **Analog Terrax** (NASA Space Apps Challenge), configured for **Vercel Python Functions (Fluid Compute)**.

## Architecture Highlights
- **`app/core/`**: Cross-cutting infrastructure (Pydantic settings, structured JSON request logging, `X-Request-ID` tracing, RFC 7807 exception handling, pooled `httpx.AsyncClient` with `tenacity` retries, and Vercel CDN `Cache-Control` helpers).
- **`app/api/v1/`**: Versioned REST routers with clean OpenAPI operation IDs for TypeScript SDK generation.
- **`app/clients/`**: Reusable async base adapter (`BaseUpstreamClient`) with in-instance `TTLCache` and exponential backoff for external NASA APIs.
- **`app/services/`**: Domain logic layer (ready for future challenge features).
- **`app/schemas/`**: Strict Pydantic v2 models that drive `/openapi.json`.

## Local Development

```bash
# 1. Install dependencies using uv
uv sync --dev

# 2. Copy environment variables
cp .env.example .env

# 3. Start development server with hot-reload on port 8000
uv run uvicorn app.main:app --reload --port 8000

# 4. Run quality checks & tests
uv run ruff check .
uv run mypy app
uv run pytest

# 5. Export OpenAPI schema for the frontend
uv run python scripts/export_openapi.py --output openapi.json
```

## Deploying to Vercel

```bash
vercel link --yes --project analog-terrax-backend
vercel deploy --prod
```
