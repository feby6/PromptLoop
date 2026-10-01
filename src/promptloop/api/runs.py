"""In-memory run manager: starts optimisation runs as background tasks and fans their
events out to any number of SSE subscribers.

Deliberately in-memory: runs are short-lived, the app has no accounts, and keeping user
data off disk is a design goal. The trade-off is that a server restart loses runs in
flight, and the app must run as a single worker process.
"""

import asyncio
import logging
import secrets
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from promptloop.api.schemas import CreateRunRequest, RunStatus, RunStatusName
from promptloop.config import LoopSettings, Settings, provider_of
from promptloop.data import split_examples
from promptloop.llm import LLMClient, LLMError, MemoryCache
from promptloop.models import TERMINAL_EVENTS, Example, RunEvent, RunResult, Task
from promptloop.optimizer.common import OptimizerError
from promptloop.optimizer.graph import RunContext, optimise
from promptloop.report import LineageStep, build_lineage
from promptloop.scorers import auto_scorer_name, scorer_for_task
from promptloop.scorers.judge import DEFAULT_RUBRIC

logger = logging.getLogger(__name__)

# How often an idle SSE stream sends a keep-alive, so proxies don't close it.
HEARTBEAT_S = 15.0

ClientFactory = Callable[[Settings, dict[str, str]], LLMClient]


def default_client_factory(settings: Settings, api_keys: dict[str, str]) -> LLMClient:
    """Per-run client: the user's key only, never the server's, and nothing on disk."""
    return LLMClient(settings, cache=MemoryCache(), api_keys=api_keys, allow_env_keys=False)


class TooManyRuns(RuntimeError):
    pass


class RunNotFound(KeyError):
    pass


@dataclass
class RunRecord:
    id: str
    created_at: datetime
    status: RunStatusName = "running"
    events: list[RunEvent] = field(default_factory=list)
    result: RunResult | None = None
    lineage: list[LineageStep] | None = None
    error: str | None = None
    finished_at: float | None = None  # monotonic, for expiry
    task: asyncio.Task[None] | None = None
    subscribers: set[asyncio.Queue[RunEvent]] = field(default_factory=set)

    def emit(self, event: RunEvent) -> None:
        event = event.model_copy(update={"seq": len(self.events) + 1})
        self.events.append(event)
        for queue in self.subscribers:
            queue.put_nowait(event)

    def to_status(self) -> RunStatus:
        return RunStatus(
            run_id=self.id,
            status=self.status,
            created_at=self.created_at,
            error=self.error,
            result=self.result,
            lineage=self.lineage,
            events=self.events,
        )


@dataclass
class PreparedRun:
    """A validated run request, ready to start. Holds the key only until the run starts."""

    task: Task
    train: list[Example]
    val: list[Example]
    api_key: str


class RunManager:
    def __init__(
        self,
        settings: Settings,
        max_active: int = 4,
        ttl_s: float = 3600.0,
        client_factory: ClientFactory = default_client_factory,
    ) -> None:
        self.settings = settings
        self.max_active = max_active
        self.ttl_s = ttl_s
        self.client_factory = client_factory
        self._runs: dict[str, RunRecord] = {}

    # --- lifecycle -------------------------------------------------------------------

    def prepare(self, req: CreateRunRequest) -> PreparedRun:
        """Validate and split the request. Raises ValueError with a user-facing message."""
        synth = self.settings.synthesis
        examples = [e.to_example() for e in req.examples]
        if len(examples) < synth.min_examples:
            raise ValueError(
                f"need at least {synth.min_examples} examples (got {len(examples)}); "
                "use 'Generate more examples' to add some"
            )
        if len({e.input for e in examples}) != len(examples):
            raise ValueError("examples contain duplicate inputs")
        train, val = split_examples(examples, synth.val_fraction, synth.min_val)
        scorer = auto_scorer_name(examples) if req.scorer == "auto" else req.scorer
        task = Task(
            name=req.task_name,
            description=req.task_description.strip(),
            scorer=scorer,
            # The user's model does everything: target, optimiser and judge.
            target_model=req.model,
            optimizer_model=req.model,
            # Task requires a rubric for the judge; fall back to the judge's default.
            rubric=req.rubric or (DEFAULT_RUBRIC if scorer == "judge" else None),
        )
        return PreparedRun(task=task, train=train, val=val, api_key=req.api_key.get_secret_value())

    def start(self, prepared: PreparedRun, req: CreateRunRequest) -> RunRecord:
        self._expire_old()
        active = sum(1 for r in self._runs.values() if r.status == "running")
        if active >= self.max_active:
            raise TooManyRuns(f"the server is busy ({active} runs in progress); try again soon")
        run_id = secrets.token_urlsafe(9)
        record = RunRecord(id=run_id, created_at=datetime.now(UTC))
        self._runs[run_id] = record
        loop = self.settings.loop.model_copy(
            update={"n_candidates": req.n_candidates, "max_iterations": req.max_iterations}
        )
        record.task = asyncio.create_task(self._execute(record, prepared, loop))
        record.task.add_done_callback(lambda t: self._on_task_done(record, t))
        return record

    @staticmethod
    def _on_task_done(record: RunRecord, task: asyncio.Task[None]) -> None:
        # A task cancelled before its first step never enters `_execute`'s try block,
        # so its record would otherwise stay "running" forever.
        if task.cancelled() and record.status == "running":
            record.status = "cancelled"
            record.emit(RunEvent(type="cancelled", message="Run cancelled"))
            record.finished_at = time.monotonic()

    async def _execute(self, record: RunRecord, prepared: PreparedRun, loop: LoopSettings) -> None:
        task = prepared.task
        llm = self.client_factory(self.settings, {provider_of(task.target_model): prepared.api_key})
        # Drop our reference to the key; from here only the client holds it.
        prepared.api_key = ""
        try:
            ctx = RunContext(
                llm=llm,
                task=task,
                train=prepared.train,
                val=prepared.val,
                scorer=scorer_for_task(task, llm),
                loop=loop,
                on_event=record.emit,
            )
            result = await optimise(ctx)
            record.result = result
            record.lineage = build_lineage(result)
            record.status = "succeeded"
        except asyncio.CancelledError:
            record.status = "cancelled"
            record.emit(RunEvent(type="cancelled", message="Run cancelled"))
        except (LLMError, OptimizerError) as e:
            # These messages are already key-redacted by the LLM client.
            record.status = "failed"
            record.error = str(e)
            record.emit(RunEvent(type="error", message=str(e)))
        except Exception:
            # Unexpected bug: log the traceback server-side, keep details from the user.
            logger.exception("run %s crashed", record.id)
            record.status = "failed"
            record.error = "internal error"
            record.emit(RunEvent(type="error", message="internal error"))
        finally:
            llm.close()
            record.finished_at = time.monotonic()

    def get(self, run_id: str) -> RunRecord:
        try:
            return self._runs[run_id]
        except KeyError:
            raise RunNotFound(run_id) from None

    def cancel(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        if record.task and not record.task.done():
            record.task.cancel()
        return record

    async def shutdown(self) -> None:
        tasks = [r.task for r in self._runs.values() if r.task and not r.task.done()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _expire_old(self) -> None:
        now = time.monotonic()
        expired = [
            run_id
            for run_id, r in self._runs.items()
            if r.finished_at is not None and now - r.finished_at > self.ttl_s
        ]
        for run_id in expired:
            del self._runs[run_id]

    # --- streaming ---------------------------------------------------------------------

    async def subscribe(self, run_id: str, after: int = 0) -> AsyncIterator[RunEvent | None]:
        """Events with seq > `after`: replayed history first, then live ones.

        Yields None as a heartbeat when idle. Ends after a terminal event. `after` lets
        a reconnecting EventSource resume via Last-Event-ID without duplicates.
        """
        record = self.get(run_id)
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        # Register before snapshotting: no await in between, so no event can be missed.
        record.subscribers.add(queue)
        try:
            last = after
            for event in list(record.events):
                if event.seq > last:
                    last = event.seq
                    yield event
                    if event.type in TERMINAL_EVENTS:
                        return
            if record.status != "running":
                return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_S)
                except TimeoutError:
                    yield None
                    continue
                if event.seq <= last:
                    continue  # already replayed
                last = event.seq
                yield event
                if event.type in TERMINAL_EVENTS:
                    return
        finally:
            record.subscribers.discard(queue)
