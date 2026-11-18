
import pandas as pd
import pytest

from pathsnap.data.legacy_loader import load_trip


TRIP_ID = "00000001"


def create_trip_files(trip_dir):
    """Create minimal whitespace-separated trip files."""
    trip_dir.mkdir(parents=True, exist_ok=True)

    (trip_dir / f"{TRIP_ID}.track").write_text(
        "77.5946 12.9716 1000\n"
        "77.5947 12.9717 1001\n"
    )

    (trip_dir / f"{TRIP_ID}.nodes").write_text(
        "77.5946 12.9716\n"
        "77.5947 12.9717\n"
    )

    (trip_dir / f"{TRIP_ID}.arcs").write_text(
        "0 1\n"
    )


def test_load_trip_without_route(tmp_path):
    """A trip without a route should load for inference."""
    trip_dir = tmp_path / TRIP_ID
    create_trip_files(trip_dir)

    trip = load_trip(
        data_dir=tmp_path,
        trip_id=TRIP_ID,
        include_route=False,
    )

    assert trip["route"] is None
    assert len(trip["track"]) > 0
    assert len(trip["nodes"]) > 0
    assert len(trip["arcs"]) > 0


def test_load_trip_requires_route_by_default(tmp_path):
    """The default loader should require the route file."""
    trip_dir = tmp_path / TRIP_ID
    create_trip_files(trip_dir)

    with pytest.raises(
        FileNotFoundError,
        match=r"Required file missing: .*\.route",
    ):
        load_trip(
            data_dir=tmp_path,
            trip_id=TRIP_ID,
        )


def test_load_trip_with_route(tmp_path):
    """A trip with a valid route should load normally."""
    trip_dir = tmp_path / TRIP_ID
    create_trip_files(trip_dir)

    (trip_dir / f"{TRIP_ID}.route").write_text("0\n")

    trip = load_trip(
        data_dir=tmp_path,
        trip_id=TRIP_ID,
    )

    assert trip["route"] is not None
    assert len(trip["route"]) == 1
    assert trip["route"]["arc_id"].iloc[0] == 0
