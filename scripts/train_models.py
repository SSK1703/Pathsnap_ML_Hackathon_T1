from pathlib import Path

import pandas as pd

from pathsnap.models.decision_tree import train_and_evaluate as evaluate_decision_tree
from pathsnap.models.knn import train_and_evaluate as evaluate_knn
from pathsnap.models.logistic_regression import (
    train_and_evaluate as evaluate_logistic_regression,
)
from pathsnap.models.random_forest import train_and_evaluate as evaluate_random_forest
from pathsnap.models.xgboost_model import train_and_evaluate as evaluate_xgboost


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "src" / "pathsnap" / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "reports" / "generated"


def main():
    train_path = PROCESSED_DIR / "train_candidates.parquet"
    test_path = PROCESSED_DIR / "student_test_candidates.parquet"
    missing = [path for path in (train_path, test_path) if not path.is_file()]
    if missing:
        missing_paths = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Processed candidate data is missing:\n"
            f"{missing_paths}\n"
            "Build it first with `python scripts/build_dataset.py`."
        )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    if "is_match" not in train_df.columns:
        raise ValueError("Training candidates must contain an is_match column.")
    if test_df.empty:
        raise ValueError("Student-test candidates are empty.")

    results = []
    evaluators = [
        ("logistic_regression", evaluate_logistic_regression),
        ("decision_tree", evaluate_decision_tree),
        ("random_forest", evaluate_random_forest),
        ("knn", evaluate_knn),
        ("xgboost", evaluate_xgboost),
    ]
    for model_name, evaluator in evaluators:
        result, _ = evaluator(
            train_df,
            test_df,
            save_path=ARTIFACTS_DIR / f"{model_name}.joblib",
        )
        results.append(result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "member2_model_results.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"Model comparison saved to {results_path}")


if __name__ == "__main__":
    main()
