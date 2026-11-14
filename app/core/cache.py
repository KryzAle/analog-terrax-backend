from collections.abc import Callable
from typing import Any

from cachetools import TTLCache
from fastapi import Response

# Shared in-instance warm memory cache for Vercel Fluid Compute concurrency reuse
_warm_cache: TTLCache[str, Any] = TTLCache(maxsize=512, ttl=300)


def get_warm_cache() -> TTLCache[str, Any]:
    """Return the process-level warm TTLCache."""
    return _warm_cache


def cache_control(
    s_maxage: int = 300,
    stale_while_revalidate: int = 600,
    public: bool = True,
) -> Callable[[Response], None]:
    """FastAPI dependency that sets Vercel CDN Cache-Control headers on a route."""

    def _dependency(response: Response) -> None:
        visibility = "public" if public else "private"
        response.headers["Cache-Control"] = (
            f"{visibility}, s-maxage={s_maxage}, stale-while-revalidate={stale_while_revalidate}"
        )

    return _dependency
