"""HTTP entry point for the Llama4Codex adapter."""

import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .compat import transform_request, transform_response, transform_sse_stream

app = FastAPI(title="Llama4Codex")

HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def proxy_headers(headers: httpx.Headers) -> dict[str, str]:
    """Return headers safe to forward across the proxy boundary."""
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS and name.lower() != "host"
    }


def bind_config() -> tuple[str, int]:
    """Return the externally configurable bind address and port."""
    return os.getenv("L4C_HOST", "0.0.0.0"), int(os.getenv("L4C_PORT", "8081"))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/llama")
async def llama_health() -> JSONResponse:
    """Proxy the upstream llama-server health response."""
    upstream = os.getenv("L4C_UPSTREAM_URL", "http://127.0.0.1:8080")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{upstream.rstrip('/')}/health")
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "component": "llama-server", "detail": str(exc)},
        )

    try:
        content = response.json()
    except ValueError:
        content = {"status": response.text}
    return JSONResponse(status_code=response.status_code, content=content)


def upstream_url(path: str) -> str:
    base = os.getenv("L4C_UPSTREAM_URL", "http://127.0.0.1:8080").rstrip("/")
    return f"{base}{path}"


@app.get("/v1/models")
async def models() -> JSONResponse:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(upstream_url("/v1/models"))
    return JSONResponse(status_code=response.status_code, content=response.json())


@app.post("/v1/responses")
async def responses(request: Request) -> Response:
    payload = await request.json()
    transformed = transform_request(payload)
    if payload.get("stream"):
        async def stream_body():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", upstream_url("/v1/responses"), json=transformed.payload
                ) as upstream:
                    async for chunk in transform_sse_stream(
                        upstream.aiter_bytes(), transformed.tool_reverse_map
                    ):
                        yield chunk

        return StreamingResponse(stream_body(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=None) as client:
        upstream = await client.post(upstream_url("/v1/responses"), json=transformed.payload)
    try:
        response_payload = upstream.json()
    except ValueError:
        return JSONResponse(status_code=upstream.status_code, content={"detail": upstream.text})
    return JSONResponse(
        status_code=upstream.status_code,
        content=transform_response(response_payload, transformed.tool_reverse_map),
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def passthrough(request: Request, path: str) -> StreamingResponse:
    """Proxy routes not owned by the adapter without interpreting their payloads."""
    body = await request.body()
    target = upstream_url(f"/{path}")
    if request.url.query:
        target = f"{target}?{request.url.query}"
    request_headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS and name.lower() != "host"
    }

    async def body_iterator():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                request.method,
                target,
                content=body,
                headers=request_headers,
            ) as upstream:
                response_headers = proxy_headers(upstream.headers)
                yield (upstream.status_code, response_headers, upstream)
                async for chunk in upstream.aiter_raw():
                    yield chunk

    iterator = body_iterator()
    status_code, response_headers, upstream = await iterator.__anext__()

    async def response_body():
        try:
            async for chunk in iterator:
                yield chunk
        finally:
            await iterator.aclose()

    return StreamingResponse(
        response_body(),
        status_code=status_code,
        headers=response_headers,
        media_type=None,
    )


def main() -> None:
    import uvicorn

    host, port = bind_config()
    uvicorn.run("llama4codex.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
