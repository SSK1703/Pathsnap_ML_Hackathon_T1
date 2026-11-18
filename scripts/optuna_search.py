import json
import time
from pathlib import Path

import optuna
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss

from pathsnap.evaluation.model_evaluation import FEATURE_COLS, evaluate_model

# Ensure Optuna is less verbose
optuna.logging.set_verbosity(optuna.logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "src" / "pathsnap" / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "reports" / "generated"

def objective(trial, X, y, groups):
    """
    Bayesian optimization objective function using GroupKFold on training trips.
    Strictly uses Bayesian Sequential Model-Based Optimization (Optuna TPESampler).
    """
    # 1. Define justified hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
        "max_depth": trial.suggest_int("max_depth", 6, 16),
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 25),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 12),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }

    gkf = GroupKFold(n_splits=5)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf = RandomForestClassifier(**params)
        clf.fit(X_tr, y_tr)

        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        score = f1_score(y_val, preds, average="macro", zero_division=0)
        cv_scores.append(score)

    return float(np.mean(cv_scores))

def run_bayesian_optimization(n_trials=20):
    print("=" * 60)
    print("PATHSNAP BAYESIAN HYPERPARAMETER OPTIMISATION (TPE SAMPLER)")
    print("=" * 60)
    print("Strict rule check: Sequential Bayesian Optimization via Tree-structured Parzen Estimator (TPE).")
    print("Grid search and random search are strictly excluded.\n")

    train_path = PROCESSED_DIR / "train_candidates.parquet"
    test_path = PROCESSED_DIR / "student_test_candidates.parquet"

    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "Processed candidate datasets are missing. "
            "Run `python scripts/build_dataset.py` first."
        )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["is_match"].values
    groups = train_df["trip_id"].values

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)

    # Strictly Bayesian TPE Sampler
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="pathsnap_bayesian_optimization"
    )

    t0 = time.time()
    study.optimize(lambda trial: objective(trial, X_train, y_train, groups), n_trials=n_trials)
    optimization_time = time.time() - t0

    print(f"\nCompleted {len(study.trials)} Bayesian optimization trials in {optimization_time:.2f}s")
    print(f"Best Trial Score (CV Macro F1): {study.best_value:.4f}")
    print(f"Best Parameters: {json.dumps(study.best_params, indent=2)}")

    # 1. Save all trials to CSV
    trials_df = study.trials_dataframe()
    trials_csv = RESULTS_DIR / "optimization_trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    print(f"Saved optimization trials log to: {trials_csv}")

    # 2. Save best parameters JSON
    best_params_path = RESULTS_DIR / "best_parameters.json"
    with best_params_path.open("w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Saved best parameters to: {best_params_path}")

    # 3. Refit tuned model on entire train set and evaluate on student-test
    best_config = study.best_params.copy()
    best_config["class_weight"] = "balanced"
    best_config["random_state"] = 42
    best_config["n_jobs"] = -1

    tuned_model = RandomForestClassifier(**best_config)

    print("\n--- Evaluating Untuned Baseline Model ---")
    untuned_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight="balanced", n_jobs=-1)
    untuned_eval = evaluate_model("Random Forest (Untuned)", "Ensemble", "Default config", untuned_model, train_df, test_df)

    print("\n--- Evaluating Tuned Model (Bayesian Opt) ---")
    tuned_eval = evaluate_model("Random Forest (Bayesian Tuned)", "Ensemble", str(study.best_params), tuned_model, train_df, test_df)

    # 4. Save Final Tuned Model Artifact
    final_model_path = PROJECT_ROOT / "artifacts" / "final_model.joblib"
    joblib.dump(tuned_model, final_model_path)
    print(f"\nSaved final tuned model artifact to: {final_model_path}")

    # 5. Output Comparison Table for Report Section 4
    comp_df = pd.DataFrame([untuned_eval, tuned_eval])
    comp_csv = RESULTS_DIR / "bayesian_tuned_vs_untuned.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"\n=== Tuned vs Untuned Comparison (Saved to {comp_csv}) ===")
    print(comp_df.to_string(index=False))

    return study, tuned_model

if __name__ == "__main__":
    run_bayesian_optimization(n_trials=15)
