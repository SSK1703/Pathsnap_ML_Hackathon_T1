
from pathlib import Path

import pandas as pd


class TripRegistry:
    def __init__(self, manifest_path: str | Path, data_root: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        self.data_root = Path(data_root).resolve()

    def load_manifest(self) -> pd.DataFrame:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Trip manifest not found: {self.manifest_path}"
            )

        df = pd.read_csv(
           self.manifest_path,
           dtype={"trip_id": str}
        )

        required = {"trip_id", "track_path"}
        if not required.issubset(df.columns):
            raise ValueError(
                "Manifest must contain trip_id and track_path."
            )

        if df.empty:
            raise ValueError("Trip manifest cannot be empty.")

        if df["trip_id"].isna().any():
            raise ValueError("Trip IDs cannot be missing.")

        df["trip_id"] = (
           df["trip_id"]
           .astype(str)
           .str.strip()
           .str.zfill(8)
        )

        if (df["trip_id"] == "").any():
            raise ValueError("Trip IDs cannot be empty.")

        if df["trip_id"].duplicated().any():
            raise ValueError("Trip IDs must be unique.")

        if not df["track_path"].map(
            lambda x: isinstance(x, str) and bool(x.strip())
        ).all():
            raise ValueError("Track paths cannot be missing or empty.")

        df["track_path"] = df["track_path"].astype(str).str.strip()

        # Validate paths and prevent paths escaping the data directory.
        for track_path in df["track_path"]:
            path = Path(track_path)

            if path.is_absolute():
                raise ValueError("Track paths must be relative paths.")

            resolved_path = (self.data_root / path).resolve()

            if not resolved_path.is_relative_to(self.data_root):
                raise ValueError(
                    f"Track path escapes the data directory: {track_path}"
                )

            if resolved_path.suffix != ".track":
                raise ValueError(
                    f"Expected a .track file: {track_path}"
                )

        return df

    def list_trip_ids(self) -> list[str]:
        df = self.load_manifest()
        return df["trip_id"].tolist()

    def get_trip(self, trip_id: str) -> Path:
        df = self.load_manifest()

        matches = df.loc[df["trip_id"] == trip_id, "track_path"]

        if matches.empty:
            raise KeyError(f"Unknown trip ID: {trip_id}")

        return (self.data_root / matches.iloc[0]).resolve()
