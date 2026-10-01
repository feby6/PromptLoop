from promptloop.models import (
    Candidate,
    EvalResult,
    ExampleResult,
    IterationRecord,
    RunResult,
)
from promptloop.report import build_lineage, lineage, render_markdown, word_diff


def _eval(cid: str, score: float) -> EvalResult:
    return EvalResult(
        candidate_id=cid,
        results=[ExampleResult(input="i", expected_output="yes", output="?", score=score)],
    )


def make_result() -> RunResult:
    c0 = Candidate(id="it0-c0", prompt="Answer the question.")
    c1 = Candidate(
        id="it1-c0", prompt="Answer the question with yes.", iteration=1, parent_ids=["it0-c0"]
    )
    c2 = Candidate(
        id="it2-c0", prompt="Always answer with yes.", iteration=2, parent_ids=["it1-c0", "it0-c0"]
    )
    history = [
        IterationRecord(
            iteration=0,
            candidates=[c0],
            evals=[_eval("it0-c0", 0.0)],
            best_candidate_id="it0-c0",
            best_score=0.0,
        ),
        IterationRecord(
            iteration=1,
            candidates=[c1],
            evals=[_eval("it1-c0", 0.5)],
            best_candidate_id="it1-c0",
            best_score=0.5,
        ),
        IterationRecord(
            iteration=2,
            candidates=[c2],
            evals=[_eval("it2-c0", 1.0)],
            best_candidate_id="it2-c0",
            best_score=1.0,
            critique="be firm",
        ),
    ]
    return RunResult(
        task_name="toy",
        target_model="groq/t",
        optimizer_model="groq/o",
        scorer="exact",
        best_candidate=c2,
        train_score=1.0,
        val_score=0.5,
        val_eval=EvalResult(
            candidate_id="it2-c0",
            results=[
                ExampleResult(input="v0", expected_output="yes", output="yes", score=1.0),
                ExampleResult(input="v1", expected_output="yes", output="no", score=0.0),
            ],
        ),
        baseline_train_score=0.0,
        baseline_val_score=0.0,
        stop_reason="target_reached",
        history=history,
        n_train=1,
        n_val=2,
    )


def test_word_diff_round_trips() -> None:
    old, new = "Answer the question.\nBe brief.", "Answer the hard question.\nBe very brief."
    chunks = word_diff(old, new)
    assert "".join(c.text for c in chunks if c.op != "insert") == old
    assert "".join(c.text for c in chunks if c.op != "delete") == new
    assert any(c.op == "insert" and "hard" in c.text for c in chunks)


def test_word_diff_identical_text() -> None:
    assert [c.op for c in word_diff("same", "same")] == ["equal"]


def test_lineage_follows_first_parent() -> None:
    assert [c.id for c in lineage(make_result())] == ["it0-c0", "it1-c0", "it2-c0"]


def test_build_lineage_scores_and_diffs() -> None:
    steps = build_lineage(make_result())
    assert [s.train_score for s in steps] == [0.0, 0.5, 1.0]
    assert steps[0].diff_from_parent == []
    assert any(c.op == "insert" for c in steps[1].diff_from_parent)


def test_render_markdown_sections() -> None:
    md = render_markdown(make_result())
    assert "| baseline (task description) | 0.000 | 0.000 |" in md
    assert "| optimised (`it2-c0`) | 1.000 | 0.500 |" in md
    assert "### it0-c0 → it1-c0" in md
    assert "- **Input:** v1" in md and "v0" not in md.split("still failing")[1]


def test_run_result_json_round_trip() -> None:
    # Dumped results include computed fields (mean_score); loading must accept them.
    result = make_result()
    again = RunResult.model_validate_json(result.model_dump_json())
    assert again.val_eval.mean_score == result.val_eval.mean_score
    assert [f.input for f in again.val_failures] == ["v1"]
