import time
from collections import deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

_MAX_TRACKED_CLIENTS = 10_000
_WINDOW_SECONDS = 60.0


def _client_ip(request: Request) -> str:
    # Vercel sets x-forwarded-for to the real client IP at the edge.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Best-effort per-instance sliding-window limiter for compute-heavy endpoints.

    State is in-memory, so on serverless it is per instance. It is a cheap first
    line of defense; use Vercel Firewall rate limiting for a global limit.
    """

    def __init__(self, app, limit_per_minute: int, protected_prefixes: tuple[str, ...]) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.limit = limit_per_minute
        self.protected_prefixes = protected_prefixes
        self._hits: dict[str, deque[float]] = {}

    def _is_protected(self, request: Request) -> bool:
        return request.method != "OPTIONS" and request.url.path.startswith(self.protected_prefixes)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if self.limit <= 0 or not self._is_protected(request):
            return await call_next(request)

        now = time.monotonic()
        if len(self._hits) > _MAX_TRACKED_CLIENTS:
            self._hits = {
                k: v for k, v in self._hits.items() if v and now - v[-1] < _WINDOW_SECONDS
            }

        hits = self._hits.setdefault(_client_ip(request), deque())
        while hits and now - hits[0] >= _WINDOW_SECONDS:
            hits.popleft()

        if len(hits) >= self.limit:
            retry_after = max(1, int(_WINDOW_SECONDS - (now - hits[0])))
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": "Too many requests. Please retry later.",
                        "details": None,
                        "request_id": getattr(request.state, "request_id", None),
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )

        hits.append(now)
        return await call_next(request)
