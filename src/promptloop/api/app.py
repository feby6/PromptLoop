"""FastAPI app: JSON API under /api, live run progress over Server-Sent Events, and the
built React frontend served from / (so one container serves the whole app).

Run with `promptloop serve`, or `uvicorn promptloop.api.app:create_app --factory`.
Must run as a single worker: runs live in this process's memory.
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from promptloop import __version__
from promptloop.api.runs import (
    ClientFactory,
    RunManager,
    RunNotFound,
    RunRecord,
    TooManyRuns,
    default_client_factory,
)
from promptloop.api.schemas import (
    AppConfig,
    CreateRunRequest,
    GenerateExamplesRequest,
    GenerateExamplesResponse,
    RunCreated,
    RunStatus,
)
from promptloop.config import PROJECT_ROOT, Settings, load_settings, provider_of
from promptloop.llm import LLMError
from promptloop.optimizer.common import OptimizerError
from promptloop.optimizer.synthesize import synthesize_examples

logger = logging.getLogger(__name__)

DEFAULT_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def create_app(
    settings: Settings | None = None,
    client_factory: ClientFactory = default_client_factory,
    frontend_dist: Path | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    manager = RunManager(
        settings,
        max_active=int(os.environ.get("PROMPTLOOP_MAX_ACTIVE_RUNS", "4")),
        client_factory=client_factory,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await manager.shutdown()

    app = FastAPI(title="PromptLoop", version=__version__, lifespan=lifespan)
    app.state.manager = manager

    # Only needed when the frontend is hosted separately (e.g. Vercel + Render).
    if origins := os.environ.get("PROMPTLOOP_CORS_ORIGINS"):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in origins.split(",") if o.strip()],
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type", "Last-Event-ID"],
        )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/config")
    async def config() -> AppConfig:
        return AppConfig(
            suggested_models=settings.suggested_models,
            min_examples=settings.synthesis.min_examples,
            target_total_examples=settings.synthesis.target_total,
            default_candidates=settings.loop.n_candidates,
            default_max_iterations=settings.loop.max_iterations,
        )

    @app.post("/api/examples/generate")
    async def generate_examples(req: GenerateExamplesRequest) -> GenerateExamplesResponse:
        """Draft extra examples for the user to review. Uses the user's model and key."""
        llm = client_factory(settings, {provider_of(req.model): req.api_key.get_secret_value()})
        try:
            examples = await synthesize_examples(
                llm,
                req.model,
                req.task_description,
                [e.to_example() for e in req.seed_examples],
                req.n,
            )
        except (LLMError, OptimizerError) as e:
            # 502: the failure is upstream (the user's provider), not in their request.
            raise HTTPException(status_code=502, detail=str(e)) from None
        finally:
            llm.close()
        return GenerateExamplesResponse(examples=examples)

    @app.post("/api/runs", status_code=201)
    async def create_run(req: CreateRunRequest) -> RunCreated:
        try:
            prepared = manager.prepare(req)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from None
        try:
            record = manager.start(prepared, req)
        except TooManyRuns as e:
            raise HTTPException(status_code=429, detail=str(e)) from None
        return RunCreated(
            run_id=record.id,
            scorer=prepared.task.scorer,
            n_train=len(prepared.train),
            n_val=len(prepared.val),
        )

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> RunStatus:
        return _get_or_404(manager, run_id).to_status()

    @app.delete("/api/runs/{run_id}")
    async def cancel_run(run_id: str) -> RunStatus:
        try:
            return manager.cancel(run_id).to_status()
        except RunNotFound:
            raise HTTPException(status_code=404, detail="run not found") from None

    @app.get("/api/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        request: Request,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        """SSE stream of RunEvents. Browsers' EventSource reconnects automatically and
        sends Last-Event-ID, which we use to resume without replaying duplicates."""
        _get_or_404(manager, run_id)
        after = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0

        async def stream() -> AsyncIterator[str]:
            async for event in manager.subscribe(run_id, after=after):
                if await request.is_disconnected():
                    break
                if event is None:
                    yield ": keep-alive\n\n"  # SSE comment line; ignored by clients
                    continue
                yield f"id: {event.seq}\nevent: {event.type}\ndata: {event.model_dump_json()}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            # no-transform/X-Accel-Buffering stop proxies from buffering the stream.
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    dist = frontend_dist or Path(os.environ.get("PROMPTLOOP_FRONTEND_DIST", DEFAULT_FRONTEND_DIST))
    if (dist / "index.html").is_file():
        # Mounted last so /api routes take precedence.
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:
        logger.info("frontend build not found at %s; serving the API only", dist)

    return app


def _get_or_404(manager: RunManager, run_id: str) -> RunRecord:
    try:
        return manager.get(run_id)
    except RunNotFound:
        raise HTTPException(status_code=404, detail="run not found") from None
