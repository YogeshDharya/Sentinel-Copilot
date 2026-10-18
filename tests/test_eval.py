import pytest

from app.eval.runner import _aggregate_scores


def test_aggregates_expected_ragas_metrics() -> None:
    scores = _aggregate_scores(
        [
            {"faithfulness": 0.8, "context_recall": 0.6, "answer_relevancy": 1.0},
            {"faithfulness": 0.6, "context_recall": 0.8, "answer_relevancy": 0.8},
        ]
    )

    assert scores == {
        "faithfulness": 0.7,
        "context_recall": 0.7,
        "answer_relevancy": 0.9,
    }


def test_raises_when_ragas_does_not_return_a_metric() -> None:
    with pytest.raises(ValueError, match="no scores for faithfulness"):
        _aggregate_scores([{"context_recall": 0.5, "answer_relevancy": 0.6}])


def test_raises_when_ragas_returns_no_rows() -> None:
    with pytest.raises(ValueError, match="no evaluation scores"):
        _aggregate_scores([])
