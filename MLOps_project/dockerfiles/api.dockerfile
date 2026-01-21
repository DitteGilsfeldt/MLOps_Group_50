# 1. Switch from 'alpine' to 'slim-bookworm' for PyTorch compatibility
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# 2. Set working directory
WORKDIR /app

# 3. Copy only dependency files first to speed up builds
COPY uv.lock pyproject.toml ./

# 4. Install dependencies (This will now include torch correctly)
RUN uv sync --frozen --no-install-project

# 5. Copy your source code
COPY src ./src

# 6. Final sync to include your local package
RUN uv sync --frozen

# 7. Default for API (Vertex AI will override this during training)
ENTRYPOINT ["sh", "-c", "uv run python -m uvicorn src.group50.api:app --host 0.0.0.0 --port $PORT"]
