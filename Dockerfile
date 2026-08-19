FROM ghcr.io/ggml-org/llama.cpp:server-cuda13

USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
COPY docker/entrypoint.sh /app/entrypoint.sh

RUN python3 -m venv /opt/l4c-venv \
    && /opt/l4c-venv/bin/pip install --no-cache-dir uv \
    && cd /app \
    && /opt/l4c-venv/bin/uv sync --frozen --no-dev \
    && chmod +x /app/entrypoint.sh

ENV PATH="/app/.venv/bin:/opt/l4c-venv/bin:${PATH}"
ENV PYTHONPATH="/app/src"

EXPOSE 8081

ENTRYPOINT ["/app/entrypoint.sh"]
