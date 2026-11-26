FROM python:3.11-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY config ./config
RUN uv sync --frozen --no-dev

# Artifacts are built by the pipeline and are not in git.
ARG ARTIFACTS_DIR=artifacts
COPY ${ARTIFACTS_DIR} ./artifacts

RUN useradd --create-home app && chown -R app /app/artifacts
USER app

# Resolve the project root from /app rather than from the installed package location.
ENV FRAUD_ROOT=/app
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "fraud.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
