
from pathlib import Path

import numpy as np
import pandas as pd


def load_trip(data_dir, trip_id, include_route=True):
    """
    Load GPS observations, road nodes, road arcs,
    and optionally the reference route for one PathSnap trip.

    Arc IDs and node IDs are zero-based row indices.

    Args:
        data_dir: Dataset partition directory containing trip folders.
        trip_id: Trip identifier.
        include_route: Whether to load and validate the reference route.

    Returns:
        Dictionary containing trip_id, track, nodes, arcs, and route.
        When include_route=False, route is None.
    """
    trip_id = str(trip_id).zfill(8)
    trip_dir = Path(data_dir) / trip_id

    if not trip_dir.is_dir():
        raise FileNotFoundError(
            f"Trip directory not found: {trip_dir}"
        )

    # Required files. The reference route is optional for inference.
    files = {
        "track": ["longitude", "latitude", "timestamp"],
        "nodes": ["longitude", "latitude"],
        "arcs": ["source_node", "target_node"],
    }

    if include_route:
        files["route"] = ["arc_id"]

    data = {}

    # Load required files and, optionally, the reference route.
    for extension, columns in files.items():
        path = trip_dir / f"{trip_id}.{extension}"

        if not path.is_file():
            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=columns,
        )

        if df.empty:
            raise ValueError(f"Empty file: {path}")

        if df.isna().any().any():
            raise ValueError(
                f"Missing or invalid values in {path}"
            )

        data[extension] = df

    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]

    # Always provide a route key in the returned dictionary.
    route = data.get("route") if include_route else None

    # Validate geographic coordinates.
    for name, df in [("track", track), ("nodes", nodes)]:
        coords = df[["longitude", "latitude"]].to_numpy(
            dtype=float
        )

        if not np.isfinite(coords).all():
            raise ValueError(
                f"Non-finite coordinates in {name}"
            )

        if (
            (np.abs(coords[:, 0]) > 180).any()
            or (np.abs(coords[:, 1]) > 90).any()
        ):
            raise ValueError(
                f"Coordinates outside geographic bounds in {name}"
            )

    # Validate timestamps.
    timestamps = track["timestamp"].to_numpy(dtype=float)

    if not np.isfinite(timestamps).all():
        raise ValueError("Invalid GPS timestamps")

    if not (np.diff(timestamps) > 0).all():
        raise ValueError(
            "GPS timestamps are not strictly increasing"
        )

    # Validate integer indices in road arcs.
    arc_values = arcs.to_numpy(dtype=float)

    if (
        not np.isfinite(arc_values).all()
        or not (arc_values == np.floor(arc_values)).all()
    ):
        raise ValueError("Invalid integer indices in arcs")

    arcs = arcs.astype(np.int64)

    # Validate arc endpoints.
    arc_values = arcs.to_numpy()

    if (
        (arc_values < 0).any()
        or (arc_values >= len(nodes)).any()
    ):
        raise ValueError(
            "Arc references an invalid node index"
        )

    # Validate the reference route only when it was requested.
    if include_route:
        route_values = route.to_numpy(dtype=float)

        if (
            not np.isfinite(route_values).all()
            or not (route_values == np.floor(route_values)).all()
        ):
            raise ValueError(
                "Invalid integer indices in route"
            )

        route = route.astype(np.int64)

        route_values = route["arc_id"].to_numpy()

        if (
            (route_values < 0).any()
            or (route_values >= len(arcs)).any()
        ):
            raise ValueError(
                "Reference route contains an invalid arc ID"
            )

    return {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
    }


def list_trips(data_dir):
    """Return sorted trip IDs from a dataset partition."""
    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}"
        )

    return sorted(
        folder.name
        for folder in data_dir.iterdir()
        if folder.is_dir()
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    train_dir = project_root / "data" / "raw" / "train"

    trip_ids = list_trips(train_dir)
    print("Training trips:", len(trip_ids))

    if trip_ids:
        trip = load_trip(train_dir, trip_ids[0])

        print("Trip ID:", trip["trip_id"])
        print("GPS observations:", len(trip["track"]))
        print("Road nodes:", len(trip["nodes"]))
        print("Road arcs:", len(trip["arcs"]))
        print("Reference route arcs:", len(trip["route"]))

        print("\nGPS observations:")
        print(trip["track"].head())

        print("\nRoad arcs:")
        print(trip["arcs"].head())
