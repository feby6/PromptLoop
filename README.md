# PromptLoop

PromptLoop finds a strong prompt for a task automatically. Give it a task description and
input → expected-output examples; it runs an iterative loop (generate → execute → evaluate →
critique → refine) against a chosen model and returns the best prompt, its score on held-out
examples, a per-iteration score history, prompt diffs, and the cases that still fail.

Runs on free hosted APIs (Groq, OpenRouter, Gemini) via LiteLLM; any LiteLLM-supported
provider works if you supply a key.

> Status: early development.

## Setup

```bash
uv sync
cp .env.example .env   # then paste your API keys
uv run promptloop --help
```

## Usage

Score the unoptimised baseline (the task description used as the prompt) on train and
held-out val examples:

```bash
uv run promptloop baseline examples/tasks/sentiment
```

A task is a folder with `task.yaml`, `train.jsonl` and `val.jsonl`; see
[`examples/tasks/sentiment`](examples/tasks/sentiment). Scorers: `exact` (normalised string
match) and `json_match` (JSON validity + per-field accuracy).

Provider rate limits, fallbacks and default models live in
[`config/providers.yaml`](config/providers.yaml). Every LLM call is cached in `.cache/`,
so reruns are free.

## Development

```bash
uv run ruff check . && uv run pytest   # offline; LLM calls are mocked
uv run pytest -m live                  # real API calls, needs a key in .env
```
