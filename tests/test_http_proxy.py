import asyncio
import json

import httpx

from llama4codex import app as app_module

app = app_module.app
RealAsyncClient = httpx.AsyncClient


def test_responses_flattens_tools_and_restores_call(monkeypatch) -> None:
    captured = {}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            alias = json["tools"][0]["name"]
            return httpx.Response(
                200,
                json={"output": [{"type": "function_call", "name": alias, "arguments": "{}"}]},
            )

    monkeypatch.setenv("L4C_UPSTREAM_URL", "http://llama:8080")
    monkeypatch.setattr(app_module.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())

    async def request() -> httpx.Response:
        async with RealAsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/v1/responses",
                json={
                    "input": "find it",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "mcp__codebase-memory-mcp",
                            "tools": [{"name": "search_graph", "parameters": {}}],
                        }
                    ],
                },
            )

    response = asyncio.run(request())

    assert captured["url"] == "http://llama:8080/v1/responses"
    assert captured["json"]["tools"][0]["type"] == "function"
    assert response.json()["output"][0]["namespace"] == "mcp__codebase-memory-mcp"
    assert response.json()["output"][0]["name"] == "search_graph"


def test_multiple_namespaces_and_function_calls_preserve_fields(monkeypatch) -> None:
    captured = {}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, json):
            captured["json"] = json
            aliases = [tool["name"] for tool in json["tools"]]
            return httpx.Response(
                200,
                json={
                    "output": [
                        {"type": "function_call", "name": aliases[0], "arguments": "{\"q\":1}", "call_id": "call-1"},
                        {"type": "function_call", "name": aliases[1], "arguments": "{}", "call_id": "call-2"},
                        {"type": "function_call", "name": "ordinary", "arguments": "{}", "call_id": "call-3"},
                    ]
                },
            )

    monkeypatch.setattr(app_module.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())

    async def request() -> httpx.Response:
        async with RealAsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post(
                "/v1/responses",
                json={
                    "tools": [
                        {"type": "namespace", "name": "ns-one", "tools": [{"name": "same", "parameters": {}}]},
                        {"type": "namespace", "name": "ns-two", "tools": [{"name": "same", "parameters": {}}]},
                        {"type": "function", "name": "ordinary", "parameters": {}},
                    ]
                },
            )

    response = asyncio.run(request())
    output = response.json()["output"]
    aliases = [tool["name"] for tool in captured["json"]["tools"]]
    assert len(set(aliases[:2])) == 2
    assert output[0]["namespace"] == "ns-one"
    assert output[1]["namespace"] == "ns-two"
    assert output[0]["arguments"] == "{\"q\":1}"
    assert output[0]["call_id"] == "call-1"
    assert output[2]["name"] == "ordinary"


def test_streaming_responses_passthrough_sse(monkeypatch) -> None:
    captured = {}

    class FakeStreamResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def aiter_bytes(self):
            yield b"event: response.output_text.delta\n"
            yield b'data: {"delta":"hello"}\n\n'
            yield b"data: [DONE]\n\n"

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        def stream(self, method, url, json):
            captured.update(method=method, url=url, json=json)
            return FakeStreamResponse()

    monkeypatch.setenv("L4C_UPSTREAM_URL", "http://llama:8080")
    monkeypatch.setattr(app_module.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())

    async def request() -> httpx.Response:
        async with RealAsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"stream": True, "input": "hello", "tools": []},
            )

    response = asyncio.run(request())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == (
        b"event: response.output_text.delta\n"
        b'data: {"delta":"hello"}\n\n'
        b"data: [DONE]\n\n"
    )
    assert captured == {
        "method": "POST",
        "url": "http://llama:8080/v1/responses",
        "json": {"stream": True, "input": "hello", "tools": []},
    }


def test_streaming_tool_search_is_lowered_and_lifted(monkeypatch) -> None:
    captured = {}
    event = (
        b"event: response.output_item.done\n"
        b'data: {"item":{"type":"function_call","name":"tool_search",'
        b'"arguments":"{\\"query\\":\\"memory\\"}","call_id":"call-1"}}\n\n'
        b"data: [DONE]\n\n"
    )

    class FakeStreamResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def aiter_bytes(self):
            remaining = event
            for index in (5, 29, len(event)):
                yield remaining[:index]
                remaining = remaining[index:]
                if not remaining:
                    break

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        def stream(self, method, url, json):
            captured.update(method=method, url=url, json=json)
            return FakeStreamResponse()

    monkeypatch.setenv("L4C_UPSTREAM_URL", "http://llama:8080")
    monkeypatch.setattr(app_module.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())

    async def request() -> httpx.Response:
        async with RealAsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/v1/responses",
                json={
                    "stream": True,
                    "tools": [
                        {
                            "type": "tool_search",
                            "execution": "client",
                            "description": "ORIGINAL",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
            )

    response = asyncio.run(request())

    assert captured["json"]["tools"] == [
        {
            "type": "function",
            "name": "tool_search",
            "description": "ORIGINAL",
            "parameters": {"type": "object"},
        }
    ]
    assert b'"type":"tool_search_call"' in response.content
    assert b'"execution":"client"' in response.content
    assert b'"arguments":{"query":"memory"}' in response.content
    assert response.content.endswith(b"data: [DONE]\n\n")


def test_followup_tool_search_history_is_lowered_for_llama(monkeypatch) -> None:
    captured = {}
    discovered = {
        "type": "function",
        "name": "search_graph",
        "description": "ORIGINAL",
        "parameters": {"type": "object"},
    }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, json):
            captured.update(url=url, json=json)
            return httpx.Response(200, json={"output": []})

    monkeypatch.setenv("L4C_UPSTREAM_URL", "http://llama:8080")
    monkeypatch.setattr(app_module.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())

    async def request() -> httpx.Response:
        async with RealAsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/v1/responses",
                json={
                    "tools": [
                        {
                            "type": "tool_search",
                            "execution": "client",
                            "description": "ORIGINAL",
                            "parameters": {"type": "object"},
                        }
                    ],
                    "input": [
                        {
                            "type": "tool_search_call",
                            "id": "fc-1",
                            "call_id": "call-1",
                            "status": "completed",
                            "execution": "client",
                            "arguments": {"query": "memory"},
                        },
                        {
                            "type": "tool_search_output",
                            "id": "tso-1",
                            "call_id": "call-1",
                            "status": "completed",
                            "execution": "client",
                            "tools": [discovered],
                        },
                    ],
                },
            )

    response = asyncio.run(request())

    assert response.status_code == 200
    assert [item["type"] for item in captured["json"]["input"]] == [
        "function_call",
        "function_call_output",
    ]
    assert captured["json"]["input"][0]["name"] == "tool_search"
    assert captured["json"]["input"][0]["call_id"] == "call-1"
    assert json.loads(captured["json"]["input"][1]["output"]) == {"tools": [discovered]}
    assert [tool["name"] for tool in captured["json"]["tools"]] == [
        "tool_search",
        "search_graph",
    ]


def test_catch_all_forwards_binary_body_query_status_and_headers(monkeypatch) -> None:
    captured = {}
    binary = b"\x00\x01html-bytes\xff"

    class FakeStreamResponse:
        status_code = 206
        headers = httpx.Headers(
            {
                "content-type": "text/html",
                "x-upstream": "preserved",
                "connection": "close",
            }
        )

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def aiter_raw(self):
            yield binary[:5]
            yield binary[5:]

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        def stream(self, method, url, content, headers):
            captured.update(method=method, url=url, content=content, headers=headers)
            return FakeStreamResponse()

    monkeypatch.setenv("L4C_UPSTREAM_URL", "http://llama:8080")
    monkeypatch.setattr(app_module.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())

    async def request() -> httpx.Response:
        async with RealAsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/assets/page.html?theme=dark",
                content=b"arbitrary-body",
                headers={"content-type": "application/octet-stream", "host": "client"},
            )

    response = asyncio.run(request())

    assert captured["method"] == "POST"
    assert captured["url"] == "http://llama:8080/assets/page.html?theme=dark"
    assert captured["content"] == b"arbitrary-body"
    assert "host" not in captured["headers"]
    assert response.status_code == 206
    assert response.content == binary
    assert response.headers["content-type"] == "text/html"
    assert response.headers["x-upstream"] == "preserved"
    assert "connection" not in response.headers


def test_explicit_routes_win_over_catch_all(monkeypatch) -> None:
    captured = []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url):
            captured.append(url)
            return httpx.Response(200, json={"data": []})

    monkeypatch.setenv("L4C_UPSTREAM_URL", "http://llama:8080")
    monkeypatch.setattr(app_module.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())

    async def request() -> tuple[httpx.Response, httpx.Response]:
        async with RealAsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get("/health"), await client.get("/v1/models")

    health, models = asyncio.run(request())

    assert health.json() == {"status": "ok"}
    assert models.json() == {"data": []}
    assert captured == ["http://llama:8080/v1/models"]
