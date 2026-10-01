"""Build benchmark task folders from public datasets.

Rows are fetched from the Hugging Face datasets-server API (no `datasets` dependency) at
fixed offsets, so anyone re-running this gets the same examples.

    uv run python benchmarks/prepare.py
"""

import json
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROWS_URL = "https://datasets-server.huggingface.co/rows"
TASKS_DIR = Path(__file__).parent / "tasks"
# Small on purpose: free-tier quotas allow roughly 1000 calls/day, and a run costs about
# (candidates + 1) x n_train x rounds calls.
N_TRAIN = 12
N_VAL = 12


# The datasets-server API returns at most this many rows per request.
PAGE_SIZE = 100


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    rows: list[dict] = []
    while len(rows) < length:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset + len(rows),
                "length": min(PAGE_SIZE, length - len(rows)),
            }
        )
        with urllib.request.urlopen(f"{ROWS_URL}?{query}", timeout=60) as resp:
            page = [r["row"] for r in json.load(resp)["rows"]]
        if not page:
            break
        rows += page
    return rows


def write_task(name: str, task_yaml: str, train: list[dict], val: list[dict]) -> None:
    root = TASKS_DIR / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.yaml").write_text(task_yaml, encoding="utf-8")
    for filename, rows in [("train.jsonl", train), ("val.jsonl", val)]:
        with (root / filename).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{name}: {len(train)} train / {len(val)} val -> {root}")


def gsm8k() -> None:
    """Grade-school maths word problems; the answer is the number after '####'.

    Exact match on the bare number is strict on purpose: the baseline model tends to show
    its working, and a good prompt has to get it to answer with the number alone.
    """
    rows = fetch_rows("openai/gsm8k", "main", "test", offset=0, length=N_TRAIN + N_VAL)
    examples = [
        {
            "input": r["question"],
            "expected_output": r["answer"].split("####")[-1].strip().replace(",", ""),
        }
        for r in rows
    ]
    write_task(
        "gsm8k",
        "name: gsm8k\n"
        "description: Solve the grade-school maths word problem and give the final answer.\n"
        "scorer: exact\n",
        examples[:N_TRAIN],
        examples[N_TRAIN:],
    )


AG_NEWS_LABELS = {0: "world", 1: "sports", 2: "business", 3: "sci/tech"}


def ag_news() -> None:
    """News headline + lede -> topic. Balanced across the four labels, so a prompt
    can't score well by always guessing the majority class."""
    rows = fetch_rows("fancyzhx/ag_news", "default", "test", offset=0, length=200)
    per_label: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        per_label[r["label"]].append(
            {"input": r["text"], "expected_output": AG_NEWS_LABELS[r["label"]]}
        )
    per_class = (N_TRAIN + N_VAL) // len(AG_NEWS_LABELS)
    if any(len(v) < per_class for v in per_label.values()):
        raise SystemExit("ag_news: not enough rows per label; increase the fetch length")
    # Interleave labels so both splits stay balanced.
    examples = [per_label[label][i] for i in range(per_class) for label in sorted(per_label)]
    write_task(
        "ag_news",
        "name: ag_news\ndescription: Classify the topic of a news article.\nscorer: exact\n",
        examples[:N_TRAIN],
        examples[N_TRAIN : N_TRAIN + N_VAL],
    )


if __name__ == "__main__":
    gsm8k()
    ag_news()
