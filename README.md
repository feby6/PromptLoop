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

## Development

```bash
uv run ruff check . && uv run pytest
```
