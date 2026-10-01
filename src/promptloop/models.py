"""Pydantic models shared across modules."""

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

ScorerName = Literal["exact", "json_match", "judge"]
Role = Literal["system", "user", "assistant"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Example(_Model):
    input: str
    expected_output: str


class Task(_Model):
    name: str
    description: str
    scorer: ScorerName
    target_model: str
    optimizer_model: str
    rubric: str | None = None

    @model_validator(mode="after")
    def _judge_needs_rubric(self) -> "Task":
        if self.scorer == "judge" and not self.rubric:
            raise ValueError("scorer 'judge' requires a rubric")
        return self


class Candidate(_Model):
    id: str
    prompt: str
    iteration: int = 0
    parent_ids: list[str] = Field(default_factory=list)


class ExampleOutput(_Model):
    """A target-model output for one example, before scoring."""

    input: str
    expected_output: str
    output: str
    error: str | None = None  # set when the LLM call failed; output is then ""
    cached: bool = False


class ExampleResult(_Model):
    input: str
    expected_output: str
    output: str
    score: float = Field(ge=0, le=1)
    error: str | None = None


class EvalResult(_Model):
    # `mean_score` is a computed field, so it appears in dumped JSON; ignore it on load
    # instead of rejecting it as an unknown field.
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    results: list[ExampleResult]

    @computed_field
    @property
    def mean_score(self) -> float:
        return sum(r.score for r in self.results) / len(self.results) if self.results else 0.0

    def failures(self, threshold: float = 1.0) -> list[ExampleResult]:
        """Examples scoring below `threshold` (including failed calls)."""
        return [r for r in self.results if r.score < threshold]

    def errors(self) -> list[ExampleResult]:
        """Examples whose LLM call failed."""
        return [r for r in self.results if r.error]


class ScoredCandidate(_Model):
    """A candidate together with its train-set evaluation."""

    candidate: Candidate
    evaluation: EvalResult

    @property
    def score(self) -> float:
        return self.evaluation.mean_score


class IterationRecord(_Model):
    """One scoring round: the candidates evaluated in it and the best-so-far after it."""

    iteration: int
    candidates: list[Candidate]
    evals: list[EvalResult]
    # Best over *all* rounds so far, not just this one, so the history is monotonic and
    # plateau detection is a simple comparison.
    best_candidate_id: str
    best_score: float
    # The critique that produced this round's candidates (None for the initial round).
    critique: str | None = None


StopReason = Literal["target_reached", "max_iterations", "plateau"]


class LLMUsage(_Model):
    """Counts for one LLMClient, so users can see what a run cost their quota."""

    calls: int = 0  # requests actually sent, including retries
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class RunResult(_Model):
    task_name: str
    target_model: str
    optimizer_model: str
    scorer: ScorerName
    best_candidate: Candidate
    train_score: float
    # Headline number: the best prompt on examples the optimiser never saw.
    val_score: float
    val_eval: EvalResult
    # The task description used verbatim as the prompt; None when the baseline is disabled.
    baseline_train_score: float | None = None
    baseline_val_score: float | None = None
    stop_reason: StopReason
    history: list[IterationRecord] = Field(default_factory=list)
    n_train: int
    n_val: int
    usage: LLMUsage = Field(default_factory=LLMUsage)

    @property
    def val_failures(self) -> list[ExampleResult]:
        return self.val_eval.failures()


EventType = Literal["phase", "candidates", "iteration", "result", "error", "cancelled"]
TERMINAL_EVENTS: frozenset[str] = frozenset({"result", "error", "cancelled"})


class RunEvent(_Model):
    """Progress update emitted by the optimisation loop (streamed to the web UI)."""

    seq: int = 0  # assigned by whoever records the event stream
    type: EventType
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class Message(_Model):
    role: Role
    content: str


class LLMRequest(_Model):
    model: str
    messages: list[Message]
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    # Distinguishes otherwise-identical requests so N samples aren't all one cache hit.
    sample_id: int = 0

    def cache_key(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()


class LLMResponse(_Model):
    content: str
    model: str  # model that actually answered (may be a fallback)
    requested_model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
