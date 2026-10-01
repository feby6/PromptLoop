# Benchmark results

_Last updated 2026-10-01. Held-out (val) scores: the optimiser never sees these examples. Baseline = the task description used as the prompt._

| task | model | scorer | train / val | baseline val | optimised val | Δ | rounds |
|---|---|---|---|---|---|---|---|
| ag_news | `groq/qwen/qwen3.8-27b` | exact | 12 / 12 | 25% | 83% | +58 pts | 1 |
| gsm8k | `groq/qwen/qwen3.8-27b` | exact | 12 / 12 | 0% | 42% | +42 pts | 3 |
| invoice_extraction | `groq/qwen/qwen3.8-27b` | json_match | 12 / 6 | 67% | 97% | +30 pts | 1 |

Small val sets (a dozen examples) mean each example is worth ~8 points: treat differences of one or two examples as noise.
