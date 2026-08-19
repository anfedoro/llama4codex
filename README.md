# llama4codex

`llama4codex` is a stateless compatibility adapter that exposes an
OpenAI Responses-compatible HTTP surface in front of a local
[`llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
It is intended for local development and self-hosted inference.

## Overview

The adapter accepts Responses API requests, translates the parts that
`llama-server` represents differently, forwards the request, and translates
the response back. It does not store conversations, proxy authentication, or
manage model weights.

```text
OpenAI Responses client
          |
          v
     llama4codex :8081
          |
          v
     llama.cpp server :8080
          |
          v
       local LLM
```

The default container publishes only the adapter on `http://127.0.0.1:8000`.
The llama.cpp server remains on the container network.

## Motivation / Why this exists

Local llama.cpp deployments and OpenAI Responses clients have overlapping but
not identical request and response contracts. The adapter keeps that
compatibility logic in one small, stateless boundary so that a client can use
local models without changing the model server or embedding translation logic
in every client.

## Supported capabilities

| Capability | Status | Notes |
| --- | --- | --- |
| Responses API | Supported | `POST /v1/responses` |
| Streaming SSE | Supported | Stream bytes are translated without buffering the whole response |
| MCP tool discovery | Supported | Client-side `tool_search` is lowered to a callable function |
| Deferred tool search | Supported | Search results and follow-up history are normalized |
| WebUI/API passthrough | Supported | Unowned routes are forwarded opaquely |
| Multi-model routing | Partial | Delegated to llama.cpp; the adapter does not select models |
| Reasoning effort | Supported | Preserves `reasoning.effort` and adds a safety budget when absent |
| Reasoning budget guard | Supported | `thinking_budget_tokens` is added only for known effort values |

The adapter also exposes `GET /v1/models`, `GET /health`, and
`GET /health/llama`.

## Deployment

### Prerequisites

- Linux with Docker and an NVIDIA GPU runtime for the CUDA image;
- a local GGUF model or another llama.cpp-compatible model source;
- enough GPU memory for the selected model;
- `curl` for the smoke test.

### Full flow

```bash
git clone https://github.com/anfedoro/llama4codex.git
cd llama4codex
export MODEL_DIR="$PWD/local-models"
mkdir -p "$MODEL_DIR"
cp examples/models.ini.example "$MODEL_DIR/models.ini"
```

Put the model file referenced by `models.ini` in `MODEL_DIR`, edit the preset,
then build and run:

```bash
docker build -t llama4codex .
docker run --rm --gpus all \
  -e GGML_CUDA_P2P=1 \
  -p 8000:8081 \
  -v "$MODEL_DIR:/root" \
  --name llama4codex \
  llama4codex
```

For a long-running container, add `-d --restart unless-stopped`. The repository
launcher performs the same build/run flow and health checks:

```bash
MODEL_DIR="$MODEL_DIR" ./scripts/llama4codex_start.sh
```

## Docker build

The Dockerfile is based on the CUDA-enabled llama.cpp server image. It installs
the Python runtime dependencies, copies the adapter, and starts both processes
through `docker/entrypoint.sh`.

```bash
docker build --pull -t llama4codex .
```

The image exposes port `8081`; map it to any host port you prefer. The model
directory is mounted at `/root`, so the default preset path is
`/root/models.ini`.

## Runtime configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `L4C_HOST` | `0.0.0.0` | Adapter bind address |
| `L4C_PORT` | `8081` | Adapter port |
| `L4C_UPSTREAM_URL` | `http://127.0.0.1:8080` | llama-server base URL |
| `LLAMA_HOST` | `0.0.0.0` | llama-server bind address inside the container |
| `LLAMA_PORT` | `8080` | llama-server port |
| `MODELS_PRESET` | `/root/models.ini` | llama.cpp preset file |
| `LLAMA_EXTRA_ARGS` | empty | Additional llama-server arguments |
| `GGML_CUDA_P2P` | unset | Optional CUDA runtime setting passed to the container |

To run the adapter directly against an existing server:

```bash
L4C_UPSTREAM_URL=http://127.0.0.1:8080 \
L4C_PORT=18080 \
uv run python -m llama4codex.app
```

## Model configuration

`models.ini` is consumed by llama.cpp, not parsed by the adapter. Each section
defines a model preset; keys correspond to llama-server command-line options.
The exact options depend on the llama.cpp image version. See the upstream
[model preset documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#model-presets).

Minimal local-GGUF example:

```ini
version = 1

[*]
ctx-size = 8192
n-gpu-layers = 999

[example-model]
model = /root/models/example-model.gguf
alias = example-model
```

The model name in a request must match a model accepted by llama.cpp. Verify
the effective names with:

```bash
curl http://127.0.0.1:8000/v1/models
```

## Codex CLI configuration

Configure the client to use the adapter's OpenAI-compatible base URL and the
model name returned by `/v1/models`. Client configuration keys vary by CLI
version; the essential values are:

```text
Base URL: http://127.0.0.1:8000/v1
Protocol: Responses API
Model:    example-model
```

For clients that support environment-based endpoint configuration:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_MODEL=example-model
```

For a CLI that uses provider profiles, the equivalent configuration is:

```toml
[model_providers.local]
name = "local-llama"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"

[profiles.local]
model_provider = "local"
model = "example-model"
```

Treat this as a template: retain the provider/profile keys required by your
CLI version.

Use the CLI's current provider/base-URL setting when it does not honor these
environment variables. Never put API keys or bearer tokens in this repository.

## Test request

After startup, run the health checks and a minimal Responses request:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/health/llama
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/responses \
  -H 'content-type: application/json' \
  -d @examples/responses-request.json
```

See [examples/responses-response.json](examples/responses-response.json) for a
sanitized non-streaming response shape.

## Troubleshooting

### The container exits during startup

Check that Docker can access the GPU and that `/root/models.ini` exists:

```bash
docker logs llama4codex
docker exec llama4codex sh -c 'ls -l /root/models.ini'
```

### `/health` works but `/health/llama` fails

The adapter is running but cannot reach llama.cpp. Check `LLAMA_HOST`,
`LLAMA_PORT`, `L4C_UPSTREAM_URL`, and the llama-server startup logs.

### `/v1/models` is empty or the model is rejected

Check the preset section name, model path, file permissions, and the model name
sent by the client. The adapter does not download or rename model files.

### Tool discovery does not work

Confirm that the client sends Responses `tool_search` items and that the server
response is streamed as SSE when streaming is requested. Function and
namespace tool definitions are supported; unsupported custom tool types are
rejected.

### CUDA out-of-memory

Use a smaller or more aggressively quantized model, reduce context size, or
lower the number of concurrently loaded models. The adapter does not change
the model's memory requirements.

## Limitations

- This is a stateless adapter; conversation storage, authentication, and
  multi-tenant isolation are outside its scope.
- Model routing and model lifecycle are delegated to llama.cpp.
- The compatibility layer covers the request/response shapes tested here, not
  every future Responses API field.
- Unsupported custom tool types and malformed tool definitions are rejected.
- The default deployment is plain HTTP on localhost. It does not provide TLS,
  identity, authorization, or internet-facing hardening.
- GPU/CUDA support depends on the selected upstream llama.cpp image and host
  runtime.

## Security notes

- Bind the published port to localhost or place it behind an authenticated,
  TLS-terminating reverse proxy before exposing it to a network.
- Treat prompts, model outputs, and Docker logs as potentially sensitive.
- Keep model files, credentials, caches, captures, and runtime logs outside Git.
- Never commit API keys, bearer tokens, MCP credentials, client metadata, or
  real request captures.
- Review upstream llama.cpp image tags and model licenses before redistribution.

## Development

```bash
uv sync
uv run pytest
```

All files under `tests/fixtures/codex/` are synthetic and sanitized. They are
protocol examples, not production traffic.
