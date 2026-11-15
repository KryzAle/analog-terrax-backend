import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import create_app


@pytest.fixture
def _restore_settings():
    yield
    get_settings.cache_clear()


async def test_malicious_request_id_is_replaced(async_client: AsyncClient) -> None:
    response = await async_client.get(
        "/api/v1/health", headers={"X-Request-ID": "bad\tid with spaces"}
    )
    assert response.headers["X-Request-ID"] != "bad\tid with spaces"
    ok = await async_client.get("/api/v1/health", headers={"X-Request-ID": "abc-123"})
    assert ok.headers["X-Request-ID"] == "abc-123"


async def test_oversized_site_id_rejected(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/analogs/" + "a" * 200 + "/profile")
    assert response.status_code == 422


async def test_cors_does_not_allow_credentials_or_foreign_origin(async_client: AsyncClient) -> None:
    allowed = await async_client.get(
        "/api/v1/health", headers={"Origin": "https://analog-terrax-frontend.vercel.app"}
    )
    assert (
        allowed.headers.get("access-control-allow-origin")
        == "https://analog-terrax-frontend.vercel.app"
    )
    assert "access-control-allow-credentials" not in allowed.headers
    evil = await async_client.get(
        "/api/v1/health", headers={"Origin": "https://analog-terrax-frontend.evil.com"}
    )
    assert "access-control-allow-origin" not in evil.headers


async def test_rate_limit_returns_429(monkeypatch: pytest.MonkeyPatch, _restore_settings) -> None:
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    get_settings.cache_clear()
    limited_app = create_app()
    transport = ASGITransport(app=limited_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        statuses = [(await client.get("/api/v1/analogs/heatmap")).status_code for _ in range(5)]
        health = await client.get("/api/v1/health")
    assert statuses[:3] == [200, 200, 200]
    assert statuses[3:] == [429, 429]
    assert health.status_code == 200


async def test_docs_hidden_in_production(
    monkeypatch: pytest.MonkeyPatch, _restore_settings
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        assert (await client.get("/docs")).status_code == 404
        assert (await client.get("/redoc")).status_code == 404
