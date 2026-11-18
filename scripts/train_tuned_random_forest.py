
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "pathsnap" / "data" / "processed"
OUT = ROOT / "reports" / "generated"
ARTIFACTS = ROOT / "artifacts"

FEATURE_COLS = [
    "dist_perp",
    "dist_to_start",
    "dist_to_end",
    "arc_length",
    "heading_diff",
    "speed",
    "proj_ratio"
]

RANDOM_STATE = 42


def main():
    print("=" * 65)
    print("PATHSNAP - FINAL BAYESIAN-TUNED RANDOM FOREST")
    print("=" * 65)

    train_path = DATA / "train_candidates.parquet"
    student_path = DATA / "student_test_candidates.parquet"
    params_path = OUT / "best_parameters.json"

    for path in [train_path, student_path, params_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    train_df = pd.read_parquet(train_path)
    student_df = pd.read_parquet(student_path)

    with open(params_path, "r") as f:
        best_params = json.load(f)

    print(f"Training rows: {len(train_df):,}")
    print(f"Training trips: {train_df['trip_id'].nunique()}")
    print(f"Student-test rows: {len(student_df):,}")
    print(f"Best parameters: {best_params}")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["is_match"]
    X_student = student_df[FEATURE_COLS]

    model = RandomForestClassifier(
        **best_params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Train once on the complete labelled training dataset.
    print("\nTraining final model on all training data...")
    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start
    print(f"Training completed in {training_time:.2f} seconds")

    # Generate predictions for the unlabeled student-test data.
    print("\nGenerating student-test predictions...")
    start = time.perf_counter()

    probabilities = model.predict_proba(X_student)
    positive_class_index = list(model.classes_).index(1)
    match_probabilities = probabilities[:, positive_class_index]

    predictions = (match_probabilities >= 0.5).astype(int)

    inference_time = time.perf_counter() - start

    # Save the trained model.
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS / "random_forest_bayesian_tuned.joblib"
    joblib.dump(model, model_path)

    # Save candidate-level predictions.
    prediction_df = student_df[
        ["trip_id", "obs_id", "candidate_arc_id"]
    ].copy()

    prediction_df["match_probability"] = match_probabilities
    prediction_df["predicted_match"] = predictions

    prediction_path = OUT / "student_test_rf_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False)

    # Record results without inventing metrics for unlabeled data.
    result = {
        "Model": "Random Forest - Bayesian Tuned",
        "Algorithm Family": "Ensemble",
        "Best Parameters": json.dumps(best_params),
        "Tuning CV Macro F1 (10% sample)": 0.8398891139871439,
        "Training Rows": len(train_df),
        "Training Trips": train_df["trip_id"].nunique(),
        "Student-Test Rows": len(student_df),
        "Student-Test Macro F1": "N/A (unlabeled)",
        "Student-Test Top-1 Accuracy": "N/A (unlabeled)",
        "Training Time (seconds)": round(training_time, 2),
        "Inference Time (seconds)": round(inference_time, 2)
    }

    result_df = pd.DataFrame([result])

    evaluation_path = OUT / "final_tuned_random_forest_evaluation.csv"
    result_df.to_csv(evaluation_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL MODEL TRAINING COMPLETED")
    print("=" * 65)

    print("\nModel saved:")
    print(model_path)

    print("\nStudent-test predictions saved:")
    print(prediction_path)

    print("\nEvaluation saved:")
    print(evaluation_path)

    print("\nFINAL RESULTS")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
