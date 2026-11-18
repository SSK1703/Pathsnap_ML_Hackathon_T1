
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from pathsnap.evaluation.model_evaluation import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "pathsnap" / "data" / "processed"
OUT = ROOT / "reports" / "generated"

RANDOM_STATE = 42
N_SPLITS = 3
N_TRIALS = 8
SAMPLE_FRACTION = 0.10


def cv_score(params, X, y, groups):
    cv = GroupKFold(n_splits=N_SPLITS)
    scores = []

    for train_idx, val_idx in cv.split(X, y, groups):
        model = RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(X[train_idx], y[train_idx])
        predictions = model.predict(X[val_idx])

        scores.append(
            f1_score(
                y[val_idx],
                predictions,
                average="macro",
                zero_division=0,
            )
        )

    return float(np.mean(scores))


def objective(trial, X, y, groups):
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 50, 125, step=25
        ),
        "max_depth": trial.suggest_int("max_depth", 6, 16),
        "min_samples_split": trial.suggest_int(
            "min_samples_split", 5, 25
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 2, 12
        ),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
    }

    return cv_score(params, X, y, groups)


def main():
    print("PATHSNAP RANDOM FOREST BAYESIAN OPTIMISATION")

    train_path = DATA / "train_candidates.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing dataset: {train_path}")

    df = pd.read_parquet(train_path)

    required = FEATURE_COLS + ["trip_id", "obs_id", "is_match"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing values found in training data.")

    # Sample observations within every trip.
    # All candidate rows for each selected observation are retained.
    observation_keys = df[["trip_id", "obs_id"]].drop_duplicates()

    sampled_keys = observation_keys.groupby(
        "trip_id", group_keys=False
    ).sample(frac=SAMPLE_FRACTION, random_state=RANDOM_STATE)

    tune_df = df.merge(
        sampled_keys,
        on=["trip_id", "obs_id"],
        how="inner",
    )

    X = tune_df[FEATURE_COLS].to_numpy()
    y = tune_df["is_match"].to_numpy()
    groups = tune_df["trip_id"].to_numpy()

    print(f"Original rows: {len(df):,}")
    print(f"Tuning rows: {len(tune_df):,}")
    print(f"Trips represented: {tune_df['trip_id'].nunique()}")
    print(f"Positive label fraction: {y.mean():.4f}")
    print(f"CV folds: {N_SPLITS}")
    print(f"Optuna trials: {N_TRIALS}")

    OUT.mkdir(parents=True, exist_ok=True)

    # Match the original random_forest.py configuration.
    baseline_params = {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_split": 15,
        "min_samples_leaf": 5,
    }

    print("\nEvaluating baseline with GroupKFold...")
    baseline_score = cv_score(
        baseline_params, X, y, groups
    )

    print(f"Baseline CV Macro F1: {baseline_score:.4f}")

    # TPE: sequential Bayesian hyperparameter optimisation.
    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
        n_startup_trials=4,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="pathsnap_random_forest_tpe",
    )

    start = time.time()

    study.optimize(
        lambda trial: objective(trial, X, y, groups),
        n_trials=N_TRIALS,
    )

    elapsed = time.time() - start

    best_score = study.best_value
    best_params = study.best_params

    print("\nOPTIMISATION COMPLETE")
    print(f"Time: {elapsed / 60:.2f} minutes")
    print(f"Baseline CV Macro F1: {baseline_score:.4f}")
    print(f"Best CV Macro F1: {best_score:.4f}")
    print(f"Best parameters: {best_params}")

    # Save every trial.
    study.trials_dataframe().to_csv(
        OUT / "optimization_trials.csv",
        index=False,
    )

    with open(OUT / "best_parameters.json", "w") as f:
        json.dump(best_params, f, indent=4)

    comparison = pd.DataFrame([
        {
            "Model": "Random Forest - Untuned",
            "CV Macro F1": baseline_score,
        },
        {
            "Model": "Random Forest - Bayesian Tuned",
            "CV Macro F1": best_score,
        },
    ])

    comparison.to_csv(
        OUT / "bayesian_tuned_vs_untuned.csv",
        index=False,
    )

    metadata = {
        "optimizer": "Optuna TPESampler",
        "objective": "Mean 3-fold GroupKFold Macro F1",
        "grouping": "trip_id",
        "trials": N_TRIALS,
        "sampling_fraction_per_trip": SAMPLE_FRACTION,
        "original_training_rows": len(df),
        "tuning_rows": len(tune_df),
        "training_trips": int(tune_df["trip_id"].nunique()),
        "baseline_cv_macro_f1": baseline_score,
        "best_cv_macro_f1": best_score,
        "optimization_time_seconds": elapsed,
    }

    with open(OUT / "optimization_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nSAVED RESULTS:")
    print(OUT / "optimization_trials.csv")
    print(OUT / "best_parameters.json")
    print(OUT / "bayesian_tuned_vs_untuned.csv")
    print(OUT / "optimization_metadata.json")


if __name__ == "__main__":
    main()
