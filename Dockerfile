# One image serves the whole app: FastAPI on $PORT serves /api and the built frontend.

# --- 1. Build the React frontend ---------------------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY frontend/ ./
RUN npm run build

# --- 2. Python backend -------------------------------------------------------------
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# Dependencies first, so code changes don't invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY config ./config
RUN uv sync --frozen --no-dev

COPY --from=frontend /web/dist ./frontend/dist

# Run as an unprivileged user.
RUN useradd --create-home app && chown -R app /app
USER app

# 7860 is Hugging Face Spaces' default; Render and others inject $PORT.
ENV PORT=7860
EXPOSE 7860
# Single worker on purpose: runs are held in this process's memory.
CMD ["sh", "-c", "uv run --no-sync uvicorn promptloop.api.app:create_app --factory --host 0.0.0.0 --port ${PORT} --workers 1"]
