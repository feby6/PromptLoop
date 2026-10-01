# PromptLoop — Automatic Prompt Optimisation Web App

## What this project is
PromptLoop is a web app that finds the best prompt for a user's task. The user fills in a
form: a task description, one or more input → expected-output examples, their own API
key, and their preferred model. The backend then:

1. Optionally generates extra examples (user reviews/edits them before the run).
2. Generates 3 candidate prompts.
3. Runs each candidate on the examples with the user's model and scores the outputs.
4. Critiques the best candidate's failures and refines new candidates (beam search).
5. Repeats until the score stops improving, then reports the best prompt, its score on
   held-out examples vs. the unoptimised baseline, a score-per-iteration history,
   prompt diffs, and the cases that still fail.

**Everything runs on the user's key and model** (generation, execution, critique,
refinement, judging). The project's own keys in `.env` are for development, tests and
benchmarks only.

Built by Feby Das (AI/ML Engineer) as a public portfolio project for GitHub and LinkedIn.
Code quality, a clear README, and honest, reproducible benchmark numbers matter more than
feature count.

## Hard constraints
- **Free hosted APIs only — no local models.** Do not use or install Ollama.
  Free providers used for development (all via LiteLLM):
  - Groq free tier (verified 2026-10-01: `qwen/qwen3.8-27b`)
  - OpenRouter free models (`:free` suffix; low daily cap without credits)
  - Google Gemini Flash / Flash-Lite free tier (AI Studio key)
  Check each provider's current model list and limits before hard-coding model names;
  keep model names in config, never inline.
  Design must stay provider-agnostic (Claude/OpenAI must work if a user supplies a key).
- **User API keys are secrets in transit only.** Held in memory for one request/run;
  never written to disk, logged, cached, put in a cache key, or echoed in errors
  (`llm.py` redacts them). Use `SecretStr` at the API boundary.
- **No user data at rest in web mode.** Web runs use an in-memory cache and do not write
  `runs/`; the disk cache and run logs are for CLI/dev use.
- **Rate limits come from the user's provider, not from us.** Don't tune defaults to the
  project's own free-tier key. React to 429s (exponential backoff honouring
  `Retry-After`, a provider-wide pause), cap concurrency, and stop the run with a clear
  error when quota is exhausted (never score it as wrong answers). Local RPM/RPD/TPM
  limits and fallback models exist in config but are off by default.
- **Cache every LLM call** (keyed on model + messages + params) so reruns are free.
- **No secrets in code or git.** Keys come from `.env` (gitignored); keep `.env.example`
  updated with placeholders.
- Dev machine: MacBook Air (Apple Silicon). No GPU, no local inference.

## Stack
- Python 3.12, managed with `uv` (use `uv add` / `uv run`, never bare pip)
- LangGraph — the optimisation loop as a cyclic graph
- LiteLLM — single interface for Groq / OpenRouter / Gemini / OpenAI / Anthropic / ...
  (model strings like `groq/<model>`, `openrouter/<model>`, `gemini/<model>`)
- Pydantic v2 — all data models
- diskcache — LLM response cache (CLI/dev); in-memory cache for web runs
- FastAPI + uvicorn — backend API; Server-Sent Events for live run progress
- React + TypeScript + Vite — frontend (`frontend/`), served by FastAPI in production
- Typer + Rich — CLI
- pytest — tests; ruff — lint/format
- Docker — single image (frontend build + backend) for free-tier hosting

## Project layout
```
src/promptloop/
  config.py          # settings from config/providers.yaml + .env
  models.py          # Task, Example, Candidate, EvalResult, IterationRecord, RunResult, RunEvent
  llm.py             # LiteLLM wrapper: caching, retries, rate limiting, per-run keys
  data.py            # load task folders; split examples into train/val
  prompts/           # optimiser prompt templates (*.txt) + render()
  scorers/
    base.py          # Scorer / AsyncScorer protocols, normalise()
    exact.py         # exact / normalised string match
    json_match.py    # JSON validity + per-field accuracy
    judge.py         # LLM-as-judge with rubric (noisier; use only when needed)
  optimizer/
    synthesize.py    # generate extra examples from the user's seeds
    generate.py      # initial candidate prompts from task description
    execute.py       # run a candidate prompt on examples with the target model
    evaluate.py      # score outputs per candidate
    critique.py      # critic LLM explains failures ("textual gradient")
    refine.py        # new candidates from top-k + critique
    graph.py         # LangGraph wiring + stopping logic; emits RunEvents
  report.py          # lineage, prompt diffs, markdown report, run logging
  api/               # FastAPI app: schemas, run manager, SSE events, static frontend
  cli.py             # `promptloop baseline|run|serve`
frontend/            # React + Vite app
benchmarks/          # public-dataset tasks, prepare + run scripts, RESULTS.md
examples/tasks/<task_name>/{task.yaml, train.jsonl, val.jsonl}
tests/
```

## Data format (CLI tasks)
`task.yaml`:
```yaml
name: sentiment
description: Classify a product review as positive, negative, or neutral.
scorer: exact            # exact | json_match | judge
target_model: groq/<model>        # optional; defaults from config/providers.yaml
optimizer_model: groq/<model>     # optional; defaults from config/providers.yaml
rubric: null             # only for judge scorer
```
`train.jsonl` / `val.jsonl`, one example per line:
```json
{"input": "...", "expected_output": "..."}
```
Minimum 10 train examples for CLI tasks. Web runs need at least 6 examples in total and
are split into train/val automatically. **Val examples are NEVER shown to the optimiser.**

## The loop
1. Generate N (default 3) candidate prompts from the task description; the task
   description itself is also scored as the "baseline" candidate.
2. Execute each candidate on train examples with the target model.
3. Score outputs with the task's scorer; average per candidate.
4. Critique: send the best candidate's failed cases to the critic LLM for a diagnosis.
5. Refine: produce new candidates from top-k (default 2) + critique (beam search).
6. Stop on: max rounds (default 5), target score reached, or no improvement for
   2 rounds (plateau).
7. Final: evaluate the best train candidate (and the baseline) on val; report val score
   vs baseline val score as the headline number.

CLI runs log every iteration (prompts, scores, critique) to `runs/<timestamp>-<task>/`.

## Conventions
- Type hints everywhere; Pydantic models for anything crossing module boundaries.
- Small, pure functions; LLM calls only through `llm.py`.
- Every scorer and loop step gets unit tests; mock LLM calls in tests.
- Keep prompts used by the optimiser itself in `src/promptloop/prompts/` as text files,
  not inline strings.
- Comments explain *why*, not what.
- Run `uv run ruff check . && uv run pytest` before committing.

## Milestones (build in order)
- [x] M0: project setup (uv package, deps, .gitignore, .env.example, git init)
- [x] M1: config, models, data loading, LLM wrapper with cache + retries
- [x] M2: scorers (exact, json_match) + baseline evaluation of a single prompt
- [x] M3: full optimisation loop in LangGraph (3 candidates, baseline candidate),
      example synthesis, per-run API keys, `promptloop run` CLI
- [x] M4: held-out validation vs baseline, run logging, report (history, diffs, failures)
- [x] M5: FastAPI backend: generate examples, start run, live SSE events, results,
      cancel; keys in memory only; serves the built frontend
- [x] M6: React frontend: task form, example review/edit, live progress chart,
      results with prompt diffs and failures
- [x] M7: LLM-as-judge scorer; benchmarks on public datasets (baseline vs optimised
      val score); README with results table + architecture diagram
- [x] M8: deployment: Dockerfile, Render blueprint, HF Spaces / Vercel instructions
      (user supplies own key in the UI)

## Out of scope for v1
Multi-turn prompts, tool-using prompts, fine-tuning, user accounts, persistent run
history in web mode, paid-API-only features.
