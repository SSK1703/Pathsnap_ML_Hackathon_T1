
import numpy as np
import pandas as pd
from pathlib import Path

from pathsnap.data.legacy_loader import load_trip
from pathsnap.data.preprocessing import preprocess_trip
from pathsnap.route.route_alignment import align_gps_to_route


def main():
    root = Path(__file__).resolve().parents[1]

    student_test_dir = (
        root / "src" / "pathsnap" / "data" /
        "raw" / "student_test"
    )

    candidates_path = (
        root / "src" / "pathsnap" / "data" /
        "processed" / "student_test_candidates.parquet"
    )

    predictions_path = (
        root / "reports" / "hmm_viterbi" /
        "hmm_route_predictions.csv"
    )

    output_dir = root / "reports" / "hmm_viterbi"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading candidate data and predictions...")

    candidates = pd.read_parquet(candidates_path)
    predictions = pd.read_csv(predictions_path)

    candidates["trip_id"] = (
        candidates["trip_id"].astype(str).str.zfill(8)
    )

    predictions["trip_id"] = (
        predictions["trip_id"].astype(str).str.zfill(8)
    )

    trip_results = []
    observation_results = []

    for trip_id in sorted(predictions["trip_id"].unique()):

        print(f"\nEvaluating trip {trip_id}...")

        # Load reference route ONLY for offline evaluation.
        trip = load_trip(student_test_dir, trip_id)
        processed = preprocess_trip(trip)

        alignment = align_gps_to_route(processed)

        reference_arcs = alignment["matched_arc_ids"]
        route_positions = alignment["route_positions"]
        alignment_distances = alignment["match_distances"]

        trip_candidates = candidates[
            candidates["trip_id"] == trip_id
        ]

        trip_predictions = predictions[
            predictions["trip_id"] == trip_id
        ].sort_values("obs_id")

        candidate_sets = (
            trip_candidates.groupby("obs_id")["candidate_arc_id"]
            .apply(lambda values: set(values.astype(int)))
        )

        jumps = np.abs(np.diff(route_positions))

        rf_correct = 0
        hmm_correct = 0
        eligible_count = 0
        candidate_covered_count = 0
        excluded_count = 0

        for row in trip_predictions.itertuples(index=False):

            obs_id = int(row.obs_id)

            reference_arc = int(reference_arcs[obs_id])
            distance = float(alignment_distances[obs_id])

            forward_jump = (
                0 if obs_id == 0
                else int(jumps[obs_id - 1])
            )

            candidate_ids = candidate_sets.loc[obs_id]

            covered = reference_arc in candidate_ids

            # Same confidence policy used during training.
            reasons = []

            if not covered:
                reasons.append("matched_arc_missing_from_candidates")

            if distance > 50.0:
                reasons.append("large_alignment_distance")

            if forward_jump > 10:
                reasons.append("implausible_route_jump")

            eligible = len(reasons) == 0

            if covered:
                candidate_covered_count += 1

            if not eligible:
                excluded_count += 1

            rf_match = (
                int(row.rf_top1_arc_id) == reference_arc
            )

            hmm_match = (
                int(row.hmm_viterbi_arc_id) == reference_arc
            )

            if eligible:
                eligible_count += 1
                rf_correct += int(rf_match)
                hmm_correct += int(hmm_match)

            observation_results.append(
                {
                    "trip_id": trip_id,
                    "obs_id": obs_id,
                    "reference_arc_id": reference_arc,
                    "rf_top1_arc_id": int(row.rf_top1_arc_id),
                    "hmm_viterbi_arc_id": int(row.hmm_viterbi_arc_id),
                    "alignment_distance": distance,
                    "route_position_jump": forward_jump,
                    "candidate_covered": covered,
                    "eligible": eligible,
                    "exclusion_reason": ";".join(reasons),
                    "rf_correct": rf_match,
                    "hmm_correct": hmm_match,
                }
            )

        total_observations = len(trip_predictions)

        rf_accuracy = (
            rf_correct / eligible_count
            if eligible_count else np.nan
        )

        hmm_accuracy = (
            hmm_correct / eligible_count
            if eligible_count else np.nan
        )

        candidate_recall = (
            candidate_covered_count / total_observations
            if total_observations else np.nan
        )

        evaluation_coverage = (
            eligible_count / total_observations
            if total_observations else np.nan
        )

        trip_results.append(
            {
                "trip_id": trip_id,
                "total_observations": total_observations,
                "eligible_observations": eligible_count,
                "excluded_observations": excluded_count,
                "candidate_recall": candidate_recall,
                "evaluation_coverage": evaluation_coverage,
                "rf_accuracy": rf_accuracy,
                "hmm_accuracy": hmm_accuracy,
                "accuracy_improvement": hmm_accuracy - rf_accuracy,
            }
        )

        print(f"Eligible observations: {eligible_count}")
        print(f"RF accuracy: {rf_accuracy:.4f}")
        print(f"HMM accuracy: {hmm_accuracy:.4f}")

    # --------------------------------------------------
    # Save detailed and per-trip results
    # --------------------------------------------------

    observations_df = pd.DataFrame(observation_results)
    trips_df = pd.DataFrame(trip_results)

    observations_df.to_csv(
        output_dir / "hmm_ground_truth_observations.csv",
        index=False,
    )

    trips_df.to_csv(
        output_dir / "hmm_ground_truth_evaluation.csv",
        index=False,
    )

    # Micro accuracy: pool all eligible observations.
    eligible_df = observations_df[
        observations_df["eligible"]
    ]

    rf_micro = eligible_df["rf_correct"].mean()
    hmm_micro = eligible_df["hmm_correct"].mean()

    # Macro accuracy: average the per-trip accuracies.
    rf_macro = trips_df["rf_accuracy"].mean()
    hmm_macro = trips_df["hmm_accuracy"].mean()

    print("\n" + "=" * 60)
    print("PATHSNAP - GROUND-TRUTH EVALUATION")
    print("=" * 60)

    print(f"Trips evaluated: {len(trips_df)}")
    print(f"Total observations: {len(observations_df):,}")
    print(f"Eligible observations: {len(eligible_df):,}")

    print("\nMICRO ACCURACY")
    print(f"RF Top-1: {rf_micro:.4f}")
    print(f"HMM:     {hmm_micro:.4f}")
    print(f"Improvement: {hmm_micro - rf_micro:+.4f}")

    print("\nMACRO ACCURACY (average per trip)")
    print(f"RF Top-1: {rf_macro:.4f}")
    print(f"HMM:     {hmm_macro:.4f}")
    print(f"Improvement: {hmm_macro - rf_macro:+.4f}")

    print("\nAverage candidate recall:")
    print(f"{trips_df['candidate_recall'].mean():.4f}")

    print("\nAverage evaluation coverage:")
    print(f"{trips_df['evaluation_coverage'].mean():.4f}")

    print("\nSaved evaluation files:")
    print(output_dir / "hmm_ground_truth_observations.csv")
    print(output_dir / "hmm_ground_truth_evaluation.csv")


if __name__ == "__main__":
    main()
