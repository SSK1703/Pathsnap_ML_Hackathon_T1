import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from pathsnap.evaluation.model_evaluation import (
    FEATURE_COLS,
    compute_observation_top1,
    evaluate_model,
    run_group_cv,
)


def _training_rows(group_count=5):
    rows = []
    for trip_index in range(group_count):
        for label, distance in ((0, 10.0), (1, 0.0)):
            row = {column: 0.0 for column in FEATURE_COLS}
            row.update(
                {
                    "trip_id": f"trip-{trip_index}",
                    "obs_id": 0,
                    "dist_perp": distance,
                    "is_match": label,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_observation_top1_uses_highest_scoring_candidate():
    candidates = pd.DataFrame(
        {
            "trip_id": ["trip-a", "trip-a", "trip-a", "trip-a"],
            "obs_id": [0, 0, 1, 1],
            "is_match": [0, 1, 1, 0],
        }
    )

    assert compute_observation_top1(candidates, np.array([0.2, 0.8, 0.7, 0.3])) == 1.0


def test_evaluate_model_allows_unlabeled_student_test_data():
    train_df = _training_rows()
    test_df = pd.DataFrame(
        [{column: 0.0 for column in FEATURE_COLS} for _ in range(2)]
    )
    model = LogisticRegression()

    result = evaluate_model(
        model_name="test logistic regression",
        algorithm_family="linear",
        key_config="test",
        model=model,
        train_df=train_df,
        test_df=test_df,
    )

    assert result["Student-Test Macro F1-score"] == "N/A (unlabeled)"
    assert result["Top-1 Accuracy"] == "N/A (unlabeled)"
    assert result["Log Loss"] == "N/A (unlabeled)"
    assert hasattr(model, "classes_")


def test_evaluate_model_scores_labeled_test_data():
    train_df = _training_rows()
    test_df = _training_rows(group_count=1)
    model = LogisticRegression()

    result = evaluate_model(
        model_name="test logistic regression",
        algorithm_family="linear",
        key_config="test",
        model=model,
        train_df=train_df,
        test_df=test_df,
    )

    assert result["Student-Test Macro F1-score"] != "N/A (unlabeled)"
    assert result["Top-1 Accuracy"] != "N/A (unlabeled)"
    assert result["Log Loss"] != "N/A (unlabeled)"


def test_group_cv_requires_enough_distinct_trips():
    with pytest.raises(ValueError, match="at least 5 distinct trips"):
        run_group_cv(LogisticRegression(), _training_rows(group_count=4))
