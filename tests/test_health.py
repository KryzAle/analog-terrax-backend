from httpx import AsyncClient


async def test_get_health_returns_ok(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "Analog Terrax API"
    assert "version" in body
    assert "timestamp" in body
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers
    assert "s-maxage=30" in response.headers.get("Cache-Control", "")


async def test_not_found_returns_standardized_error_envelope(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/nonexistent-endpoint")
    assert response.status_code == 404
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "HTTP_404"
    assert body["error"]["request_id"] is not None
