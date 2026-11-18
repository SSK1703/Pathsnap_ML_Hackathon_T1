
import numpy as np
import pandas as pd


FEATURE_COLS = [
    "dist_perp",
    "dist_to_start",
    "dist_to_end",
    "arc_length",
    "heading_diff",
    "speed",
    "proj_ratio",
]


class HMMRouteDecoder:
    """
    HMM-based route decoder using:
      - ML candidate probabilities as emission scores
      - Road-network connectivity as transition scores
      - Viterbi dynamic programming for route selection

    IMPORTANT:
    Decode one trip at a time because arc IDs are
    zero-based row indices within each trip's road network.
    """

    def __init__(
        self,
        arcs_df,
        p_same=0.50,
        p_connected=0.45,
        p_disconnected=1e-5,
    ):
        if isinstance(arcs_df, pd.DataFrame):
            required = {"source_node", "target_node"}

            if not required.issubset(arcs_df.columns):
                raise ValueError(
                    f"arcs_df must contain columns: {required}"
                )

            arcs = arcs_df[
                ["source_node", "target_node"]
            ].to_numpy(dtype=np.int64)
        else:
            arcs = np.asarray(arcs_df, dtype=np.int64)

        if arcs.ndim != 2 or arcs.shape[1] != 2:
            raise ValueError(
                "arcs must have shape (number_of_arcs, 2)"
            )

        if len(arcs) == 0:
            raise ValueError("Road network contains no arcs")

        if not np.isfinite(arcs).all() or (arcs < 0).any():
            raise ValueError("Invalid road arc endpoints")

        probabilities = [
            p_same,
            p_connected,
            p_disconnected,
        ]

        if any(
            not np.isfinite(p) or p <= 0
            for p in probabilities
        ):
            raise ValueError(
                "Transition weights must be positive and finite"
            )

        self.sources = arcs[:, 0]
        self.targets = arcs[:, 1]
        self.n_arcs = len(arcs)

        self.p_same = float(p_same)
        self.p_connected = float(p_connected)
        self.p_disconnected = float(p_disconnected)

    def _validate_candidates(self, candidate_df):
        required = {
            "obs_id",
            "candidate_arc_id",
            *FEATURE_COLS,
        }

        missing = required - set(candidate_df.columns)

        if missing:
            raise ValueError(
                f"Candidate dataframe missing columns: {sorted(missing)}"
            )

        if candidate_df.empty:
            raise ValueError("Candidate dataframe is empty")

        if candidate_df["obs_id"].isna().any():
            raise ValueError("Missing observation IDs")

        if candidate_df["candidate_arc_id"].isna().any():
            raise ValueError("Missing candidate arc IDs")

        arc_ids = candidate_df[
            "candidate_arc_id"
        ].to_numpy(dtype=float)

        if (
            not np.isfinite(arc_ids).all()
            or not np.equal(arc_ids, np.floor(arc_ids)).all()
        ):
            raise ValueError("Candidate arc IDs must be integers")

        arc_ids = arc_ids.astype(np.int64)

        if (
            (arc_ids < 0).any()
            or (arc_ids >= self.n_arcs).any()
        ):
            raise ValueError(
                "Candidate arc ID is outside this trip's road network"
            )

        features = candidate_df[
            FEATURE_COLS
        ].to_numpy(dtype=float)

        if not np.isfinite(features).all():
            raise ValueError("Candidate features contain invalid values")

    def _transition_log_scores(self, prev_arcs, curr_arcs):
        """
        Return a matrix of log transition scores.

        Rows represent previous candidate arcs.
        Columns represent current candidate arcs.

        Transition compatibility:
          Same arc        -> p_same
          Directly linked -> p_connected
          Otherwise       -> p_disconnected

        Scores are normalised across the available current
        candidates for each previous candidate.
        """

        same = prev_arcs[:, None] == curr_arcs[None, :]

        connected = (
            self.targets[prev_arcs][:, None]
            == self.sources[curr_arcs][None, :]
        )

        weights = np.full(
            (len(prev_arcs), len(curr_arcs)),
            self.p_disconnected,
            dtype=float,
        )

        weights[connected] = self.p_connected
        weights[same] = self.p_same

        row_sums = weights.sum(axis=1, keepdims=True)
        probabilities = weights / row_sums

        return np.log(np.maximum(probabilities, 1e-12))

    def decode(self, candidate_df, model):
        """
        Decode one trip.

        Returns:
            predicted_arcs_per_obs:
                Selected road arc for every observation.

            compressed_route:
                Sequential route with consecutive duplicate
                arc IDs removed.

            route_continuity:
                Fraction of selected transitions that either
                stay on the same arc or connect directly.
        """

        self._validate_candidates(candidate_df)

        candidate_df = candidate_df.copy()

        # Obtain probability belonging to class 1 explicitly.
        if not hasattr(model, "classes_"):
            raise ValueError(
                "Model must be fitted before HMM decoding"
            )

        class_indices = np.flatnonzero(
            np.asarray(model.classes_) == 1
        )

        if len(class_indices) != 1:
            raise ValueError(
                "Model must contain exactly one positive class labelled 1"
            )

        X = candidate_df[FEATURE_COLS].to_numpy(dtype=float)

        probs = model.predict_proba(X)[
            :, class_indices[0]
        ]

        if (
            not np.isfinite(probs).all()
            or (probs < 0).any()
            or (probs > 1).any()
        ):
            raise ValueError("Invalid model probabilities")

        candidate_df["model_score"] = np.clip(
            probs, 1e-12, 1.0
        )

        # Preserve chronological observation order.
        obs_ids = sorted(
            candidate_df["obs_id"].unique()
        )

        step_arcs = []
        step_emissions = []

        for obs_id in obs_ids:
            group = candidate_df[
                candidate_df["obs_id"] == obs_id
            ]

            arcs = group[
                "candidate_arc_id"
            ].to_numpy(dtype=np.int64)

            scores = group["model_score"].to_numpy(
                dtype=float
            )

            # Convert ML scores into emission probabilities
            # within the current observation's candidate set.
            emission_probs = scores / scores.sum()

            step_arcs.append(arcs)
            step_emissions.append(
                np.log(np.maximum(emission_probs, 1e-12))
            )

        # Initial Viterbi scores.
        previous_scores = step_emissions[0].copy()

        backpointers = []

        # Forward dynamic programming.
        for t in range(1, len(obs_ids)):
            prev_arcs = step_arcs[t - 1]
            curr_arcs = step_arcs[t]

            transition_scores = self._transition_log_scores(
                prev_arcs,
                curr_arcs,
            )

            total_scores = (
                previous_scores[:, None]
                + transition_scores
            )

            best_previous = np.argmax(
                total_scores,
                axis=0,
            )

            current_scores = (
                total_scores[best_previous, np.arange(len(curr_arcs))]
                + step_emissions[t]
            )

            backpointers.append(best_previous)
            previous_scores = current_scores

        # Backtrack to recover the best sequence.
        best_index = int(np.argmax(previous_scores))

        optimal_indices = [best_index]

        for t in range(len(backpointers) - 1, -1, -1):
            best_index = int(
                backpointers[t][best_index]
            )
            optimal_indices.append(best_index)

        optimal_indices.reverse()

        predicted_arcs_per_obs = [
            int(step_arcs[t][optimal_indices[t]])
            for t in range(len(obs_ids))
        ]

        # Calculate route continuity.
        total_transitions = len(predicted_arcs_per_obs) - 1
        connected_transitions = 0

        for t in range(total_transitions):
            prev_arc = predicted_arcs_per_obs[t]
            curr_arc = predicted_arcs_per_obs[t + 1]

            same_arc = prev_arc == curr_arc

            directly_connected = (
                self.targets[prev_arc]
                == self.sources[curr_arc]
            )

            if same_arc or directly_connected:
                connected_transitions += 1

        route_continuity = (
            connected_transitions / total_transitions
            if total_transitions > 0
            else 1.0
        )

        # Remove consecutive duplicate arcs.
        compressed_route = []

        for arc_id in predicted_arcs_per_obs:
            if (
                not compressed_route
                or compressed_route[-1] != arc_id
            ):
                compressed_route.append(arc_id)

        return (
            predicted_arcs_per_obs,
            compressed_route,
            float(route_continuity),
        )
