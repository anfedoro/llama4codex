import asyncio

import httpx

RealAsyncClient = httpx.AsyncClient

from llama4codex import app as app_module

app = app_module.app


def test_health() -> None:
    async def request() -> httpx.Response:
        async with RealAsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get("/health")

    response = asyncio.run(request())

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_llama_health_proxies_response(monkeypatch) -> None:
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url):
            assert url == "http://llama:8080/health"
            return httpx.Response(200, json={"status": "ok", "model": "loaded"})

    monkeypatch.setenv("L4C_UPSTREAM_URL", "http://llama:8080")
    monkeypatch.setattr(app_module.httpx, "AsyncClient", lambda: FakeClient())

    async def request() -> httpx.Response:
        async with RealAsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get("/health/llama")

    response = asyncio.run(request())

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "loaded"}
