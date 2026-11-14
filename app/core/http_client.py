import httpx

from app.core.config import get_settings

_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Get or lazily initialize a shared async HTTP client (serverless-safe)."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        settings = get_settings()
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.UPSTREAM_TIMEOUT_SECONDS),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            headers={"User-Agent": f"AnalogTerrax/{settings.VERSION}"},
        )
    return _http_client


async def close_http_client() -> None:
    """Cleanly close the shared async HTTP client on shutdown."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None
