import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss

FEATURE_COLS = [
    "dist_perp",
    "dist_to_start",
    "dist_to_end",
    "arc_length",
    "heading_diff",
    "speed",
    "proj_ratio"
]

def compute_observation_top1(df_subset, probas):
    """
    Computes Observation-Level Top-1 Accuracy.
    For each (trip_id, obs_id), select candidate with the highest predicted score.
    Returns the accuracy (proportion of observations where top candidate has is_match == 1).
    """
    df_eval = df_subset[["trip_id", "obs_id", "is_match"]].copy()
    df_eval["score"] = probas
    # Group by observation and pick candidate with max score
    idx_max = df_eval.groupby(["trip_id", "obs_id"])["score"].idxmax()
    top_matches = df_eval.loc[idx_max, "is_match"]
    return float(top_matches.mean())

def run_group_cv(model, train_df, n_splits=5):
    """
    Runs group-aware cross validation on train_df grouped by trip_id.
    """
    group_count = train_df["trip_id"].nunique()
    if n_splits < 2 or group_count < n_splits:
        raise ValueError(
            f"GroupKFold requires at least {n_splits} distinct trips; "
            f"received {group_count}."
        )

    gkf = GroupKFold(n_splits=n_splits)
    X = train_df[FEATURE_COLS].values
    y = train_df["is_match"].values
    groups = train_df["trip_id"].values

    cv_f1 = []
    cv_top1 = []
    cv_logloss = []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        val_df = train_df.iloc[val_idx]

        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        top1 = compute_observation_top1(val_df, probs)
        loss = log_loss(y_val, np.clip(probs, 1e-7, 1 - 1e-7))

        cv_f1.append(f1)
        cv_top1.append(top1)
        cv_logloss.append(loss)

    return {
        "cv_macro_f1_mean": float(np.mean(cv_f1)),
        "cv_macro_f1_std": float(np.std(cv_f1)),
        "cv_top1_mean": float(np.mean(cv_top1)),
        "cv_top1_std": float(np.std(cv_top1)),
        "cv_logloss_mean": float(np.mean(cv_logloss))
    }

def evaluate_model(model_name, algorithm_family, key_config, model, train_df, test_df):
    """
    Fits model on train_df, evaluates CV, and optionally scores a labeled
    test set. Unlabeled test rows are used only to measure inference time.
    """
    print(f"--- Evaluating {model_name} ({algorithm_family}) ---")
    if test_df.empty:
        raise ValueError("Test candidate data is empty.")

    # Cross Validation
    cv_res = run_group_cv(model, train_df)
    print(f"  CV Macro F1: {cv_res['cv_macro_f1_mean']:.4f} (+/- {cv_res['cv_macro_f1_std']:.4f})")
    print(f"  CV Top-1 Acc: {cv_res['cv_top1_mean']:.4f}")

    # Full Train Fit
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["is_match"].values

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Test Inference
    X_test = test_df[FEATURE_COLS].values
    t0 = time.time()
    probs = model.predict_proba(X_test)[:, 1]
    inference_time_total = time.time() - t0
    inf_time_per_1k = (inference_time_total / len(X_test)) * 1000.0 * 1000.0  # in ms per 1000 candidates

    if "is_match" in test_df.columns:
        if test_df["is_match"].isna().any():
            raise ValueError("Labeled test data contains missing is_match values.")

        y_test = test_df["is_match"].values
        preds = (probs >= 0.5).astype(int)
        test_macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        test_top1 = compute_observation_top1(test_df, probs)
        test_loss = log_loss(y_test, np.clip(probs, 1e-7, 1 - 1e-7))
        print(f"  Test Macro F1: {test_macro_f1:.4f}")
        print(f"  Test Top-1 Acc: {test_top1:.4f}")
        print(f"  Log Loss: {test_loss:.4f}")
        test_macro_f1 = f"{test_macro_f1:.4f}"
        test_top1 = f"{test_top1:.4f}"
        test_loss = f"{test_loss:.4f}"
    else:
        test_macro_f1 = "N/A (unlabeled)"
        test_top1 = "N/A (unlabeled)"
        test_loss = "N/A (unlabeled)"
        print("  Test metrics skipped: test dataset has no is_match labels.")

    print(f"  Train Time: {train_time:.2f}s, Inference: {inf_time_per_1k:.2f}ms / 1k")

    return {
        "Model": model_name,
        "Algorithm Family": algorithm_family,
        "Key Configuration": key_config,
        "CV Macro F1": f"{cv_res['cv_macro_f1_mean']:.4f}",
        "CV Top-1 Accuracy": f"{cv_res['cv_top1_mean']:.4f}",
        "Top-1 Accuracy": test_top1,
        "Route continuity": "N/A (Classifier only)",
        "Student-Test Macro F1-score": test_macro_f1,
        "Log Loss": test_loss,
        "Inference Time": f"{inf_time_per_1k:.2f} ms/1k"
    }
