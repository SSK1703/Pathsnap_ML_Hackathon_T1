
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from pathsnap.route.hmm_decoder import HMMRouteDecoder


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "src" / "pathsnap" / "data" / "processed"
TRAIN_RAW = ROOT / "src" / "pathsnap" / "data" / "raw" / "train"
PARAMS_PATH = ROOT / "reports" / "generated" / "best_parameters.json"
OUTPUT_DIR = ROOT / "reports" / "hmm_validation"

FEATURE_COLS = [
    "dist_perp",
    "dist_to_start",
    "dist_to_end",
    "arc_length",
    "heading_diff",
    "speed",
    "proj_ratio",
]

RANDOM_STATE = 42

# Ten configurations, including neutral and current settings.
CONFIGURATIONS = [
    {"name": "neutral", "p_same": 1.0, "p_connected": 1.0, "p_disconnected": 1.0},
    {"name": "current", "p_same": 0.50, "p_connected": 0.45, "p_disconnected": 0.00001},
    {"name": "current_soft_001", "p_same": 0.50, "p_connected": 0.45, "p_disconnected": 0.001},
    {"name": "current_soft_01", "p_same": 0.50, "p_connected": 0.45, "p_disconnected": 0.01},
    {"name": "current_soft_05", "p_same": 0.50, "p_connected": 0.45, "p_disconnected": 0.05},
    {"name": "current_soft_10", "p_same": 0.50, "p_connected": 0.45, "p_disconnected": 0.10},
    {"name": "connected_preferred", "p_same": 0.25, "p_connected": 0.50, "p_disconnected": 0.05},
    {"name": "strong_connected", "p_same": 0.50, "p_connected": 0.75, "p_disconnected": 0.05},
    {"name": "same_preferred", "p_same": 0.75, "p_connected": 0.50, "p_disconnected": 0.05},
    {"name": "strong_smoothness", "p_same": 0.75, "p_connected": 0.75, "p_disconnected": 0.05},
]


def get_road_arcs(trip_id):
    """
    Load the road network for one training trip.
    load_trip also reads the training reference route, but this
    function uses only the road arcs.
    """
    from pathsnap.data.legacy_loader import load_trip

    trip = load_trip(TRAIN_RAW, trip_id)
    return trip["arcs"]


def evaluate_trip(candidate_df, decoder, model):
    """Evaluate RF Top-1 and HMM predictions against candidate labels."""

    candidate_df = candidate_df.sort_values(
        ["obs_id", "candidate_arc_id"]
    ).copy()

    # RF positive-class probabilities.
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(
        candidate_df[FEATURE_COLS]
    )[:, positive_index]

    candidate_df["rf_score"] = probabilities

    # Ground-truth arc for observations with exactly one positive label.
    positive_counts = candidate_df.groupby("obs_id")["is_match"].sum()
    eligible_obs = positive_counts[positive_counts == 1].index

    positive_rows = candidate_df[
        (candidate_df["obs_id"].isin(eligible_obs))
        & (candidate_df["is_match"] == 1)
    ]

    truth = positive_rows.set_index("obs_id")["candidate_arc_id"]

    # RF Top-1 prediction per observation.
    best_indices = candidate_df.groupby("obs_id")["rf_score"].idxmax()
    rf_predictions = candidate_df.loc[
        best_indices
    ].set_index("obs_id")["candidate_arc_id"]

    # HMM prediction.
    hmm_predictions, _, continuity = decoder.decode(
        candidate_df,
        model,
    )

    ordered_obs = sorted(candidate_df["obs_id"].unique())

    hmm_series = pd.Series(
        hmm_predictions,
        index=ordered_obs,
        name="hmm_prediction",
    )

    common_obs = truth.index.intersection(rf_predictions.index)
    common_obs = common_obs.intersection(hmm_series.index)

    if len(common_obs) == 0:
        raise ValueError("No eligible observations for evaluation")

    rf_accuracy = (
        rf_predictions.loc[common_obs].to_numpy()
        == truth.loc[common_obs].to_numpy()
    ).mean()

    hmm_accuracy = (
        hmm_series.loc[common_obs].to_numpy()
        == truth.loc[common_obs].to_numpy()
    ).mean()

    return {
        "eligible_observations": len(common_obs),
        "rf_accuracy": float(rf_accuracy),
        "hmm_accuracy": float(hmm_accuracy),
        "continuity": float(continuity),
    }


def main():
    start_time = time.perf_counter()

    print("=" * 70)
    print("PATHSNAP - VALIDATION-BASED HMM TRANSITION TUNING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load training candidates and selected RF parameters.
    candidates = pd.read_parquet(
        DATA_DIR / "train_candidates.parquet"
    )

    with open(PARAMS_PATH, "r") as f:
        best_params = json.load(f)

    candidates["trip_id"] = (
        candidates["trip_id"].astype(str).str.zfill(8)
    )

    print(f"Candidate rows: {len(candidates):,}")
    print(f"Training trips: {candidates['trip_id'].nunique()}")

    # Split by trip, never by individual candidate rows.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_indices, validation_indices = next(
        splitter.split(
            candidates,
            groups=candidates["trip_id"],
        )
    )

    training_df = candidates.iloc[train_indices].copy()
    validation_df = candidates.iloc[validation_indices].copy()

    training_trips = sorted(training_df["trip_id"].unique())
    validation_trips = sorted(validation_df["trip_id"].unique())

    print(f"\nRF training trips: {len(training_trips)}")
    print(f"Validation trips: {len(validation_trips)}")
    print(f"RF training rows: {len(training_df):,}")
    print(f"Validation rows: {len(validation_df):,}")

    # Save the exact split for reproducibility.
    split_info = {
        "random_state": RANDOM_STATE,
        "training_trips": training_trips,
        "validation_trips": validation_trips,
    }

    with open(OUTPUT_DIR / "validation_split.json", "w") as f:
        json.dump(split_info, f, indent=4)

    # Train a fresh RF using only the 64 training trips.
    print("\nTraining validation Random Forest...")

    model = RandomForestClassifier(
        **best_params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        training_df[FEATURE_COLS],
        training_df["is_match"],
    )

    print("Validation Random Forest trained.")

    # Process each validation trip once per configuration.
    trip_data = {}

    print("\nPreparing validation trip road networks...")

    for trip_id in validation_trips:
        trip_candidates = validation_df[
            validation_df["trip_id"] == trip_id
        ].copy()

        arcs_df = get_road_arcs(trip_id)

        trip_data[trip_id] = {
            "candidates": trip_candidates,
            "arcs": arcs_df,
        }

        print(
            f"Prepared trip {trip_id}: "
            f"{trip_candidates['obs_id'].nunique()} observations"
        )

    # Evaluate every transition configuration.
    all_results = []

    for config_number, config in enumerate(CONFIGURATIONS, start=1):
        print("\n" + "-" * 65)
        print(
            f"Configuration {config_number}/{len(CONFIGURATIONS)}: "
            f"{config['name']}"
        )
        print(
            f"Same={config['p_same']}, "
            f"Connected={config['p_connected']}, "
            f"Disconnected={config['p_disconnected']}"
        )

        trip_results = []

        for trip_id, data in trip_data.items():
            decoder = HMMRouteDecoder(
                data["arcs"],
                p_same=config["p_same"],
                p_connected=config["p_connected"],
                p_disconnected=config["p_disconnected"],
            )

            result = evaluate_trip(
                data["candidates"],
                decoder,
                model,
            )

            result["trip_id"] = trip_id
            trip_results.append(result)

        total_observations = sum(
            r["eligible_observations"] for r in trip_results
        )

        micro_rf = sum(
            r["rf_accuracy"] * r["eligible_observations"]
            for r in trip_results
        ) / total_observations

        micro_hmm = sum(
            r["hmm_accuracy"] * r["eligible_observations"]
            for r in trip_results
        ) / total_observations

        macro_rf = np.mean([
            r["rf_accuracy"] for r in trip_results
        ])

        macro_hmm = np.mean([
            r["hmm_accuracy"] for r in trip_results
        ])

        mean_continuity = np.mean([
            r["continuity"] for r in trip_results
        ])

        summary = {
            **config,
            "observations": total_observations,
            "micro_rf_accuracy": micro_rf,
            "micro_hmm_accuracy": micro_hmm,
            "micro_accuracy_delta": micro_hmm - micro_rf,
            "macro_rf_accuracy": macro_rf,
            "macro_hmm_accuracy": macro_hmm,
            "macro_accuracy_delta": macro_hmm - macro_rf,
            "mean_continuity": mean_continuity,
        }

        all_results.append(summary)

        print(f"RF micro accuracy:  {micro_rf:.4f}")
        print(f"HMM micro accuracy: {micro_hmm:.4f}")
        print(f"Accuracy change:    {micro_hmm - micro_rf:+.4f}")
        print(f"Mean continuity:    {mean_continuity:.4f}")

    # Rank primarily by micro accuracy, then macro accuracy,
    # then continuity. Continuity alone does not determine the winner.
    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by=[
            "micro_hmm_accuracy",
            "macro_hmm_accuracy",
            "mean_continuity",
        ],
        ascending=False,
    ).reset_index(drop=True)

    results_df.to_csv(
        OUTPUT_DIR / "hmm_validation_comparison.csv",
        index=False,
    )

    best = results_df.iloc[0].to_dict()

    best_config = {
        "name": best["name"],
        "p_same": float(best["p_same"]),
        "p_connected": float(best["p_connected"]),
        "p_disconnected": float(best["p_disconnected"]),
        "validation_micro_accuracy": float(best["micro_hmm_accuracy"]),
        "validation_macro_accuracy": float(best["macro_hmm_accuracy"]),
        "validation_continuity": float(best["mean_continuity"]),
    }

    with open(OUTPUT_DIR / "best_hmm_configuration.json", "w") as f:
        json.dump(best_config, f, indent=4)

    print("\n" + "=" * 70)
    print("VALIDATION TUNING RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "name",
                "micro_hmm_accuracy",
                "micro_accuracy_delta",
                "macro_hmm_accuracy",
                "mean_continuity",
            ]
        ].to_string(index=False)
    )

    print("\nBest validation configuration:")
    print(json.dumps(best_config, indent=4))

    print(f"\nResults saved in: {OUTPUT_DIR}")
    print(f"Runtime: {time.perf_counter() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
