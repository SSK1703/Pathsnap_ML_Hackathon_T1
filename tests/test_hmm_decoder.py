
import numpy as np
import pandas as pd
import pytest

from pathsnap.route.hmm_decoder import HMMRouteDecoder


class DummyModel:
    """Small deterministic classifier for decoder tests."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        # Smaller dist_perp gives a higher match probability.
        positive = 1.0 / (1.0 + X[:, 0])

        return np.column_stack([
            1.0 - positive,
            positive,
        ])


def make_candidates(rows):
    return pd.DataFrame(rows)


def test_viterbi_selects_connected_route():
    # Arc 0: node 0 -> node 1
    # Arc 1: node 1 -> node 2
    # Arc 2: node 5 -> node 6 (disconnected)
    arcs = pd.DataFrame({
        "source_node": [0, 1, 5],
        "target_node": [1, 2, 6],
    })

    candidates = make_candidates([
        {
            "obs_id": 0,
            "candidate_arc_id": 0,
            "dist_perp": 0.0,
            "dist_to_start": 1.0,
            "dist_to_end": 2.0,
            "arc_length": 10.0,
            "heading_diff": 0.0,
            "speed": 5.0,
            "proj_ratio": 0.2,
        },
        {
            "obs_id": 0,
            "candidate_arc_id": 2,
            "dist_perp": 1.0,
            "dist_to_start": 2.0,
            "dist_to_end": 3.0,
            "arc_length": 10.0,
            "heading_diff": 0.0,
            "speed": 5.0,
            "proj_ratio": 0.2,
        },
        {
            "obs_id": 1,
            "candidate_arc_id": 1,
            "dist_perp": 0.0,
            "dist_to_start": 1.0,
            "dist_to_end": 2.0,
            "arc_length": 10.0,
            "heading_diff": 0.0,
            "speed": 5.0,
            "proj_ratio": 0.3,
        },
        {
            "obs_id": 1,
            "candidate_arc_id": 2,
            "dist_perp": 1.0,
            "dist_to_start": 2.0,
            "dist_to_end": 3.0,
            "arc_length": 10.0,
            "heading_diff": 0.0,
            "speed": 5.0,
            "proj_ratio": 0.3,
        },
    ])

    decoder = HMMRouteDecoder(arcs)
    model = DummyModel()

    predictions, route, continuity = decoder.decode(
        candidates,
        model,
    )

    assert predictions == [0, 1]
    assert route == [0, 1]
    assert continuity == 1.0


def test_single_observation():
    arcs = pd.DataFrame({
        "source_node": [0],
        "target_node": [1],
    })

    candidates = make_candidates([
        {
            "obs_id": 0,
            "candidate_arc_id": 0,
            "dist_perp": 0.0,
            "dist_to_start": 1.0,
            "dist_to_end": 2.0,
            "arc_length": 10.0,
            "heading_diff": 0.0,
            "speed": 0.0,
            "proj_ratio": 0.5,
        },
    ])

    decoder = HMMRouteDecoder(arcs)

    predictions, route, continuity = decoder.decode(
        candidates,
        DummyModel(),
    )

    assert predictions == [0]
    assert route == [0]
    assert continuity == 1.0


def test_invalid_arc_id_is_rejected():
    arcs = pd.DataFrame({
        "source_node": [0],
        "target_node": [1],
    })

    candidates = make_candidates([
        {
            "obs_id": 0,
            "candidate_arc_id": 5,
            "dist_perp": 0.0,
            "dist_to_start": 1.0,
            "dist_to_end": 2.0,
            "arc_length": 10.0,
            "heading_diff": 0.0,
            "speed": 0.0,
            "proj_ratio": 0.5,
        },
    ])

    decoder = HMMRouteDecoder(arcs)

    with pytest.raises(ValueError):
        decoder.decode(candidates, DummyModel())
