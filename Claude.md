# PromptLoop — Automatic Prompt Optimisation Framework

## What this project is
PromptLoop takes a task description plus input → expected-output examples and runs an
iterative loop (generate → execute → evaluate → critique → refine) to find the best prompt
for that task on a chosen model. It returns the best prompt, its score on held-out
examples, a score-per-iteration history, prompt diffs, and the cases that still fail.

Built by Feby Das (AI/ML Engineer) as a public portfolio project for GitHub and LinkedIn.
Code quality, a clear README, and honest, reproducible benchmark numbers matter more than
feature count.

## Hard constraints
- **Free hosted APIs only — no local models.** Do not use or install Ollama.
  Supported free providers (all via LiteLLM):
  - Groq free tier (fast open-source models, e.g. Llama / Qwen / GPT-OSS)
  - OpenRouter free models (`:free` suffix; low daily cap without credits)
  - Google Gemini Flash / Flash-Lite free tier (AI Studio key)
  Check each provider's current model list and limits before hard-coding model names;
  keep model names in config, never inline.
  Design must stay provider-agnostic (Claude/OpenAI must work if a user supplies a key).
- **Free-tier limits are the main engineering constraint.** A run can need hundreds of
  calls. Support per-provider rate limits (RPM/RPD from config), and optional fallback
  to another configured provider when one returns 429 or hits its daily cap.
- **No secrets in code or git.** Keys come from `.env` (gitignored). Keep `.env.example`
  updated with placeholders.
- **Cache every LLM call** (keyed on model + messages + params) so reruns are free.
- **Respect rate limits:** retry with exponential backoff on 429s; configurable concurrency.
- Dev machine: MacBook Air (Apple Silicon). No GPU, no local inference.

## Stack
- Python 3.12, managed with `uv` (use `uv add` / `uv run`, never bare pip)
- LangGraph — the optimisation loop as a cyclic graph
- LiteLLM — single interface for Groq / OpenRouter / Gemini / others
  (model strings like `groq/<model>`, `openrouter/<model>`, `gemini/<model>`)
- Pydantic v2 — all data models
- diskcache (or SQLite) — LLM response cache
- Typer + Rich — CLI
- Streamlit — web UI (later milestone)
- pytest — tests; ruff — lint/format

## Project layout
```
src/promptloop/
  config.py          # settings from env (.env via python-dotenv)
  models.py          # Task, Example, Candidate, EvalResult, IterationRecord, RunResult
  llm.py             # LiteLLM wrapper: caching, retries, rate limiting
  data.py            # load task.yaml + train/val JSONL
  scorers/
    base.py          # Scorer protocol: score(output, expected) -> float in [0,1]
    exact.py         # exact / normalised string match
    json_match.py    # JSON validity + per-field accuracy
    judge.py         # LLM-as-judge with rubric (noisier; use only when needed)
  optimizer/
    generate.py      # initial candidate prompts from task description
    execute.py       # run a candidate prompt on examples with the target model
    evaluate.py      # aggregate scores per candidate
    critique.py      # critic LLM explains failures ("textual gradient")
    refine.py        # new candidates from top-k + critique
    graph.py         # LangGraph wiring + stopping logic
  report.py          # final report: best prompt, history, diffs, failures
  cli.py             # `promptloop run examples/tasks/<task>`
app/streamlit_app.py
examples/tasks/<task_name>/{task.yaml, train.jsonl, val.jsonl}
tests/
```

## Data format
`task.yaml`:
```yaml
name: sentiment
description: Classify a product review as positive, negative, or neutral.
scorer: exact            # exact | json_match | judge
target_model: groq/<model>        # model whose prompt is being optimised
optimizer_model: gemini/<model>   # model that generates, critiques, refines
rubric: null             # only for judge scorer
```
`train.jsonl` / `val.jsonl`, one example per line:
```json
{"input": "...", "expected_output": "..."}
```
Minimum ~10 train examples; val examples are NEVER shown to the optimiser.

## The loop
1. Generate N (default 4) candidate prompts from the task description.
2. Execute each candidate on train examples with the target model.
3. Score outputs with the task's scorer; average per candidate.
4. Critique: send the best candidate's failed cases to the critic LLM for a diagnosis.
5. Refine: produce new candidates from top-k (default 2) + critique (beam search).
6. Stop on: max iterations (default 5), target score reached, or no improvement for
   2 iterations (plateau).
7. Final: evaluate the best train candidate on val set; report val score as the
   headline number.

Log every iteration (prompts, scores, critique) to `runs/<timestamp>/` as JSON.

## Conventions
- Type hints everywhere; Pydantic models for anything crossing module boundaries.
- Small, pure functions; LLM calls only through `llm.py`.
- Every scorer and loop step gets unit tests; mock LLM calls in tests.
- Keep prompts used by the optimiser itself in `src/promptloop/prompts/` as text files,
  not inline strings.
- Run `uv run ruff check . && uv run pytest` before committing.

## Milestones (build in order; finish and test each before the next)
- [x] M0: project setup — run it yourself, the user will not set anything up manually:
      `uv init --package promptloop` layout (or adapt the current folder), add deps
      (`langgraph litellm pydantic python-dotenv diskcache typer rich pyyaml`,
      dev: `pytest ruff`), create `.gitignore` (.env, runs/, .cache/, __pycache__/,
      .venv/), `.env.example` with `GROQ_API_KEY=`, `OPENROUTER_API_KEY=`,
      `GEMINI_API_KEY=` placeholders, `git init`, and a first commit. Then tell the
      user to copy `.env.example` to `.env` and paste their keys.
- [x] M1: project setup, config, models, data loading, LLM wrapper with cache + retries
- [ ] M2: scorers (exact, json_match) + baseline evaluation of a single prompt
- [ ] M3: full optimisation loop in LangGraph + CLI
- [ ] M4: held-out validation, run logging, report (history + diffs + failures)
- [ ] M5: Streamlit UI (upload task + examples, live iteration view, results)
- [ ] M6: LLM-as-judge scorer; benchmark on 2–3 public tasks (baseline vs optimised
      val score); README with results table + architecture diagram
- [ ] M7: deploy demo (Streamlit Community Cloud / HF Spaces, user supplies own key)

## Out of scope for v1
Multi-turn prompts, tool-using prompts, fine-tuning, paid-API-only features.