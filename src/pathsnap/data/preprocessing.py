
import numpy as np

EARTH_RADIUS = 6371000.0


def project_coordinates(longitude, latitude, ref_lon, ref_lat):
    """
    Convert geographic coordinates into local
    equirectangular x/y coordinates in metres.

    The same reference point must be used for
    GPS observations and road-network nodes.
    """
    lon = np.radians(np.asarray(longitude, dtype=float))
    lat = np.radians(np.asarray(latitude, dtype=float))

    lon0 = np.radians(ref_lon)
    lat0 = np.radians(ref_lat)

    x = EARTH_RADIUS * (lon - lon0) * np.cos(lat0)
    y = EARTH_RADIUS * (lat - lat0)

    return np.column_stack((x, y))


def calculate_speed(xy, timestamps):
    """
    Calculate GPS speed in metres per second.
    First observation has speed 0.
    """
    xy = np.asarray(xy, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    if len(xy) != len(timestamps):
        raise ValueError("Coordinate and timestamp lengths differ")

    speed = np.zeros(len(xy), dtype=float)

    if len(xy) < 2:
        return speed

    distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    time_deltas = np.diff(timestamps)

    if np.any(time_deltas <= 0):
        raise ValueError("Timestamps must be strictly increasing")

    speed[1:] = distances / time_deltas

    return speed


def calculate_heading(xy):
    """
    Calculate vehicle heading in degrees.

    0 degrees = east
    90 degrees = north
    180 degrees = west
    270 degrees = south
    """
    xy = np.asarray(xy, dtype=float)
    heading = np.zeros(len(xy), dtype=float)

    if len(xy) < 2:
        return heading

    displacement = np.diff(xy, axis=0)

    angles = np.degrees(
        np.arctan2(displacement[:, 1], displacement[:, 0])
    )

    heading[1:] = (angles + 360.0) % 360.0
    heading[0] = heading[1]

    return heading


def preprocess_trip(trip):
    """
    Project GPS and road nodes into the same
    coordinate system and calculate movement features.
    """
    track = trip["track"]
    nodes = trip["nodes"]

    ref_lon = float(track["longitude"].mean())
    ref_lat = float(track["latitude"].mean())

    gps_xy = project_coordinates(
        track["longitude"],
        track["latitude"],
        ref_lon,
        ref_lat,
    )

    nodes_xy = project_coordinates(
        nodes["longitude"],
        nodes["latitude"],
        ref_lon,
        ref_lat,
    )

    timestamps = track["timestamp"].to_numpy(dtype=float)

    speed = calculate_speed(gps_xy, timestamps)
    heading = calculate_heading(gps_xy)

    return {
        "trip_id": trip["trip_id"],
        "gps_xy": gps_xy,
        "nodes_xy": nodes_xy,
        "timestamps": timestamps,
        "speed": speed,
        "heading": heading,
        "arcs": trip["arcs"],
        "route": trip.get("route"),
        "reference": (ref_lon, ref_lat),
    }
