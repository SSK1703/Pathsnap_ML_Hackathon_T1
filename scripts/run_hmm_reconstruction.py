
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pathsnap.route import HMMRouteDecoder


FEATURE_COLS = [
    "dist_perp",
    "dist_to_start",
    "dist_to_end",
    "arc_length",
    "heading_diff",
    "speed",
    "proj_ratio",
]


def calculate_continuity(predicted_arcs, arcs):
    """
    Calculate the fraction of consecutive predictions
    that stay on the same arc or move to a directly
    connected arc.
    """
    if len(predicted_arcs) <= 1:
        return 1.0

    connected = 0

    for previous, current in zip(
        predicted_arcs[:-1],
        predicted_arcs[1:],
    ):
        same_arc = previous == current

        directly_connected = (
            arcs[previous, 1] == arcs[current, 0]
        )

        if same_arc or directly_connected:
            connected += 1

    return connected / (len(predicted_arcs) - 1)


def main():
    start_time = time.perf_counter()

    root = Path(__file__).resolve().parents[1]

    model_path = (
        root / "artifacts" /
        "random_forest_bayesian_tuned.joblib"
    )

    candidates_path = (
        root / "src" / "pathsnap" / "data" /
        "processed" / "student_test_candidates.parquet"
    )

    student_test_dir = (
        root / "src" / "pathsnap" / "data" /
        "raw" / "student_test"
    )

    output_dir = root / "reports" / "hmm_viterbi"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load trained Random Forest and candidate data
    # --------------------------------------------------

    print("Loading trained Random Forest...")
    model = joblib.load(model_path)

    if not hasattr(model, "classes_"):
        raise ValueError("Loaded model is not fitted")

    positive_class = np.flatnonzero(
        np.asarray(model.classes_) == 1
    )

    if len(positive_class) != 1:
        raise ValueError(
            "Random Forest must contain positive class 1"
        )

    positive_class = positive_class[0]

    print("Loading student-test candidates...")
    candidates = pd.read_parquet(candidates_path)

    candidates["trip_id"] = (
        candidates["trip_id"].astype(str).str.zfill(8)
    )

    print(f"Candidate rows: {len(candidates):,}")
    print(
        f"Student-test trips: "
        f"{candidates['trip_id'].nunique()}"
    )

    # --------------------------------------------------
    # 2. Process each trip independently
    # --------------------------------------------------

    all_predictions = []
    trip_metrics = []

    trip_groups = candidates.groupby("trip_id", sort=True)

    for trip_id, trip_candidates in trip_groups:

        print(f"\nProcessing trip {trip_id}...")

        # Read ONLY the road arcs.
        # We deliberately do not load the reference route.
        arc_path = (
            student_test_dir /
            trip_id /
            f"{trip_id}.arcs"
        )

        if not arc_path.is_file():
            raise FileNotFoundError(
                f"Road arc file not found: {arc_path}"
            )

        arcs_df = pd.read_csv(
            arc_path,
            sep=r"\s+",
            header=None,
            names=["source_node", "target_node"],
        )

        arcs = arcs_df.to_numpy(dtype=np.int64)

        # ----------------------------------------------
        # 3. Random Forest Top-1 baseline
        # ----------------------------------------------

        trip_candidates = trip_candidates.copy()

        X = trip_candidates[FEATURE_COLS]

        probabilities = model.predict_proba(X)[
            :, positive_class
        ]

        trip_candidates["rf_probability"] = probabilities

        # Highest RF probability for each observation.
        rf_best = (
            trip_candidates
            .sort_values(
                ["obs_id", "rf_probability", "candidate_arc_id"],
                ascending=[True, False, True],
                kind="stable",
            )
            .drop_duplicates("obs_id")
        )

        rf_predictions = dict(
            zip(
                rf_best["obs_id"].astype(int),
                rf_best["candidate_arc_id"].astype(int),
            )
        )

        # ----------------------------------------------
        # 4. HMM + Viterbi decoding
        # ----------------------------------------------

        decoder = HMMRouteDecoder(
    arcs_df,
    p_same=0.75,
    p_connected=0.50,
    p_disconnected=0.05,
)

        (
            hmm_arcs_per_obs,
            compressed_route,
            hmm_continuity,
        ) = decoder.decode(trip_candidates, model)

        observation_ids = sorted(
            trip_candidates["obs_id"].unique()
        )

        if len(observation_ids) != len(hmm_arcs_per_obs):
            raise RuntimeError(
                f"Prediction count mismatch for trip {trip_id}"
            )

        hmm_predictions = dict(
            zip(
                observation_ids,
                hmm_arcs_per_obs,
            )
        )

        # ----------------------------------------------
        # 5. Save observation-level predictions
        # ----------------------------------------------

        for obs_id in observation_ids:

            rf_arc = rf_predictions[int(obs_id)]
            hmm_arc = hmm_predictions[int(obs_id)]

            all_predictions.append(
                {
                    "trip_id": trip_id,
                    "obs_id": int(obs_id),
                    "rf_top1_arc_id": rf_arc,
                    "hmm_viterbi_arc_id": hmm_arc,
                }
            )

        # ----------------------------------------------
        # 6. Compare route continuity
        # ----------------------------------------------

        ordered_rf = [
            rf_predictions[int(obs_id)]
            for obs_id in observation_ids
        ]

        ordered_hmm = [
            hmm_predictions[int(obs_id)]
            for obs_id in observation_ids
        ]

        rf_continuity = calculate_continuity(
            ordered_rf,
            arcs,
        )

        # Independently calculate to verify decoder result.
        verified_hmm_continuity = calculate_continuity(
            ordered_hmm,
            arcs,
        )

        if not np.isclose(
            hmm_continuity,
            verified_hmm_continuity,
        ):
            raise RuntimeError(
                f"HMM continuity verification failed for {trip_id}"
            )

        trip_metrics.append(
            {
                "trip_id": trip_id,
                "observations": len(observation_ids),
                "rf_continuity": rf_continuity,
                "hmm_continuity": hmm_continuity,
                "continuity_improvement": (
                    hmm_continuity - rf_continuity
                ),
                "hmm_compressed_route_length": len(
                    compressed_route
                ),
            }
        )

        print(f"Observations: {len(observation_ids)}")
        print(f"RF continuity: {rf_continuity:.4f}")
        print(f"HMM continuity: {hmm_continuity:.4f}")

    # --------------------------------------------------
    # 7. Export results
    # --------------------------------------------------

    predictions_df = pd.DataFrame(all_predictions)
    metrics_df = pd.DataFrame(trip_metrics)

    predictions_path = (
        output_dir / "hmm_route_predictions.csv"
    )

    metrics_path = (
        output_dir / "hmm_continuity_comparison.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------
    # 8. Overall summary
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("PATHSNAP - HMM / VITERBI RECONSTRUCTION SUMMARY")
    print("=" * 55)

    print(f"Trips processed: {len(metrics_df)}")
    print(f"Observations processed: {len(predictions_df):,}")

    print(
        "Average RF continuity: "
        f"{metrics_df['rf_continuity'].mean():.4f}"
    )

    print(
        "Average HMM continuity: "
        f"{metrics_df['hmm_continuity'].mean():.4f}"
    )

    print(
        "Average continuity improvement: "
        f"{metrics_df['continuity_improvement'].mean():+.4f}"
    )

    print(
        "Runtime: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    print("\nSaved files:")
    print(predictions_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
