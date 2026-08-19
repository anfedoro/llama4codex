#!/usr/bin/env bash
set -euo pipefail

L4C_HOST="${L4C_HOST:-0.0.0.0}"
L4C_PORT="${L4C_PORT:-8081}"
LLAMA_HOST="${LLAMA_HOST:-0.0.0.0}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
MODELS_PRESET="${MODELS_PRESET:-/root/models.ini}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"

build_llama_args() {
  local -n out="$1"
  out=(
    --host "${LLAMA_HOST}"
    --port "${LLAMA_PORT}"
    --ui
    --models-preset "${MODELS_PRESET}"
    --models-max 1
    -fa on
  )
  if [ -n "${LLAMA_EXTRA_ARGS}" ]; then
    local extra=()
    # shellcheck disable=SC2206
    extra=(${LLAMA_EXTRA_ARGS})
    out+=("${extra[@]}")
  fi
}

llama_args=()
build_llama_args llama_args
/app/llama-server "${llama_args[@]}" &
llama_pid=$!

cleanup() {
  kill "${llama_pid}" >/dev/null 2>&1 || true
  kill "${adapter_pid:-}" >/dev/null 2>&1 || true
  wait "${llama_pid}" >/dev/null 2>&1 || true
  wait "${adapter_pid:-}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

export L4C_HOST L4C_PORT
export L4C_UPSTREAM_URL="${L4C_UPSTREAM_URL:-http://127.0.0.1:${LLAMA_PORT}}"

for attempt in $(seq 1 30); do
  if curl --silent --fail "http://${LLAMA_HOST}:${LLAMA_PORT}/health" >/dev/null; then
    break
  fi
  if [ "${attempt}" -eq 30 ]; then
    echo "llama-server did not become ready" >&2
    exit 1
  fi
  sleep 1
done

python3 -m llama4codex.app &
adapter_pid=$!

wait -n "${llama_pid}" "${adapter_pid}"
status=$?
cleanup
trap - EXIT INT TERM
exit "${status}"
