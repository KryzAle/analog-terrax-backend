from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.cache import get_warm_cache
from app.core.exceptions import UpstreamServiceError


class BaseUpstreamClient:
    """Reusable base client with automatic retries, warm TTL caching, and error mapping."""

    def __init__(self, service_name: str, base_url: str, http_client: httpx.AsyncClient) -> None:
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self._client = http_client
        self._cache = get_warm_cache()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    )
    async def _get_raw(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as exc:
            raise UpstreamServiceError(
                service=self.service_name,
                message=f"Upstream {self.service_name} returned HTTP {exc.response.status_code}",
                status_code=502,
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamServiceError(
                service=self.service_name,
                message=f"Failed to reach {self.service_name}: {exc!s}",
                status_code=503,
            ) from exc

    async def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        use_warm_cache: bool = True,
    ) -> dict[str, Any]:
        """Fetch JSON from upstream with optional warm in-instance TTLCache."""
        cache_key = f"{self.service_name}:{path}:{sorted((params or {}).items())}"
        if use_warm_cache and cache_key in self._cache:
            cached: dict[str, Any] = self._cache[cache_key]
            return cached

        data = await self._get_raw(path=path, params=params)
        if use_warm_cache:
            self._cache[cache_key] = data
        return data
