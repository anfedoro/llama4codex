#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda13"
IMAGE="llama4codex"
CONTAINER_NAME="llama-core"
MODEL_ROOT="${MODEL_DIR:-}"

if [ -z "${MODEL_ROOT}" ]; then
  echo "Set MODEL_DIR to the local model directory" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed" >&2
  exit 1
fi

if ! docker manifest inspect "${UPSTREAM_IMAGE}" >/dev/null 2>&1; then
  echo "Image not available: ${UPSTREAM_IMAGE}. Keeping current container unchanged." >&2
  exit 1
fi

docker pull "${UPSTREAM_IMAGE}"
docker build --pull -t "${IMAGE}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d --gpus all \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e GGML_CUDA_P2P=1 \
  -p 8000:8081 \
  -v "${MODEL_ROOT}:/root" \
  --restart always \
  --name "${CONTAINER_NAME}" \
  "${IMAGE}" >/dev/null

for attempt in $(seq 1 60); do
  if curl --silent --fail http://127.0.0.1:8000/health >/dev/null; then
    break
  fi
  if [ "${attempt}" -eq 60 ]; then
    echo "Llama4Codex did not become ready" >&2
    docker logs --tail 80 "${CONTAINER_NAME}" >&2 || true
    exit 1
  fi
  sleep 1
done

if ! curl --silent --fail http://127.0.0.1:8000/health/llama >/dev/null; then
  echo "llama-server health check failed" >&2
  docker logs --tail 80 "${CONTAINER_NAME}" >&2 || true
  exit 1
fi

image_id="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
llama_version="$(docker exec "${CONTAINER_NAME}" /app/llama-server --version 2>&1 | head -n 1)"
llama_hashes="$(docker exec "${CONTAINER_NAME}" sha256sum /app/llama-server /app/libggml-cuda.so 2>/dev/null || true)"

echo "container=${CONTAINER_NAME}"
echo "image=${IMAGE}"
echo "image_id=${image_id}"
echo "llama_server_version=${llama_version}"
echo "adapter_health=$(curl --silent http://127.0.0.1:8000/health)"
echo "llama_health=$(curl --silent http://127.0.0.1:8000/health/llama)"
echo "models_url=http://127.0.0.1:8000/v1/models"
printf '%s\n' "${llama_hashes}"
