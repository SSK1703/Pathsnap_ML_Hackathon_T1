from pathlib import Path

import joblib

from pathsnap.data.legacy_loader import load_trip
from pathsnap.data.preprocessing import preprocess_trip
from pathsnap.features.feature_engineering import RoadCandidateGenerator
from pathsnap.route.hmm_decoder import HMMRouteDecoder


class PathSnapPipeline:
    def __init__(self, project_root=None):
        self.project_root = (
            Path(project_root)
            if project_root
            else Path(__file__).resolve().parents[3]
        )
        self.model = None
        self.is_ready = False
        self.load()

    def load(self):
        model_path = self.project_root / "artifacts" / "random_forest.joblib"

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}"
            )

        self.model = joblib.load(model_path)

        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Loaded model does not support predict_proba")

        if not hasattr(self.model, "classes_"):
            raise ValueError("Loaded model is not fitted")

        if 1 not in self.model.classes_:
            raise ValueError("Loaded model has no positive class labelled 1")

        self.is_ready = True

    def predict(self, trip_data):
        if not self.is_ready:
            raise RuntimeError("Prediction pipeline is not ready")

        track_path = Path(trip_data).resolve()

        if not track_path.is_file():
            raise FileNotFoundError(f"Track file not found: {track_path}")

        trip_id = track_path.parent.name
        partition_root = track_path.parent.parent

        # Load inference inputs only. Never read the reference route.
        raw_trip = load_trip(
            data_dir=partition_root,
            trip_id=trip_id,
            include_route=False,
        )

        processed = preprocess_trip(raw_trip)

        candidate_generator = RoadCandidateGenerator(
            nodes_xy=processed["nodes_xy"],
            arcs=processed["arcs"],
            spacing=25.0,
        )

        candidates = candidate_generator.build_feature_frame(
            gps_xy=processed["gps_xy"],
            speed=processed["speed"],
            heading=processed["heading"],
            trip_id=trip_id,
            k=10,
            search_k=128,
        )

        decoder = HMMRouteDecoder(processed["arcs"])

        predicted_per_observation, compressed_route, continuity = (
            decoder.decode(candidates, self.model)
        )

        return {
            "predicted_arc_ids": compressed_route,
            "metrics": {
                "observations": len(predicted_per_observation),
                "route_length_arcs": len(compressed_route),
                "route_continuity": continuity,
                "candidate_count": len(candidates),
            },
        }
