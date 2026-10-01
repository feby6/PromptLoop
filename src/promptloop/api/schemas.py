"""Request/response models for the HTTP API.

Size limits here are the first line of defence for a public demo: they bound how much of
the user's quota (and our server's memory) one request can consume.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from promptloop.models import Example, RunEvent, RunResult, ScorerName
from promptloop.report import LineageStep

MAX_TEXT = 8000
MAX_TASK = 4000
MAX_EXAMPLES = 60
# LiteLLM routes on the "<provider>/" prefix, and we look up rate limits by it too.
MODEL_PATTERN = r"^[A-Za-z0-9_-]+/\S+$"


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExampleIn(_Schema):
    input: str = Field(min_length=1, max_length=MAX_TEXT)
    expected_output: str = Field(min_length=1, max_length=MAX_TEXT)

    def to_example(self) -> Example:
        return Example(input=self.input.strip(), expected_output=self.expected_output.strip())


class ModelAccess(_Schema):
    """The user's model and key. SecretStr keeps the key out of reprs, logs and errors."""

    model: str = Field(pattern=MODEL_PATTERN, max_length=200, examples=["groq/qwen/qwen3.8-27b"])
    api_key: SecretStr = Field(min_length=1, max_length=500)


class GenerateExamplesRequest(ModelAccess):
    task_description: str = Field(min_length=10, max_length=MAX_TASK)
    seed_examples: list[ExampleIn] = Field(min_length=1, max_length=20)
    n: int = Field(default=10, ge=1, le=30)


class GenerateExamplesResponse(_Schema):
    examples: list[Example]


class CreateRunRequest(ModelAccess):
    task_name: str = Field(default="web-task", min_length=1, max_length=80)
    task_description: str = Field(min_length=10, max_length=MAX_TASK)
    examples: list[ExampleIn] = Field(min_length=2, max_length=MAX_EXAMPLES)
    scorer: Literal["auto"] | ScorerName = "auto"
    rubric: str | None = Field(default=None, max_length=2000)
    n_candidates: int = Field(default=3, ge=1, le=5)
    max_iterations: int = Field(default=4, ge=1, le=8)


class RunCreated(_Schema):
    run_id: str
    scorer: ScorerName
    n_train: int
    n_val: int


RunStatusName = Literal["running", "succeeded", "failed", "cancelled"]


class RunStatus(_Schema):
    run_id: str
    status: RunStatusName
    created_at: datetime
    error: str | None = None
    result: RunResult | None = None
    lineage: list[LineageStep] | None = None
    events: list[RunEvent] = Field(default_factory=list)


class AppConfig(_Schema):
    """What the frontend needs to render the form."""

    suggested_models: list[str]
    min_examples: int
    target_total_examples: int
    max_examples: int = MAX_EXAMPLES
    default_candidates: int
    default_max_iterations: int
