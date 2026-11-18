
import folium
import streamlit as st
from streamlit_folium import st_folium


def normalize_points(data):
    """
    Convert nested coordinate data into a flat list of
    [latitude, longitude] pairs.
    """
    points = []

    def walk(item):
        if not isinstance(item, (list, tuple)):
            return

        # A valid coordinate pair.
        if (
            len(item) == 2
            and all(
                isinstance(value, (int, float))
                for value in item
            )
        ):
            lat = float(item[0])
            lon = float(item[1])

            if (
                -90 <= lat <= 90
                and -180 <= lon <= 180
            ):
                points.append([lat, lon])

            return

        # Recursively process nested lists.
        for child in item:
            walk(child)

    if data is not None:
        walk(data)

    return points


def display_map(
    gps_points=None,
    reference_route=None,
    predicted_route=None,
):
    st.subheader("Geographical Route Visualization")

    # Normalize all incoming geographical coordinates.
    gps_points = normalize_points(gps_points)
    reference_points = normalize_points(reference_route)

    # Keep predicted route segments separate.
    predicted_segments = []

    if predicted_route is not None:
        for segment in predicted_route:
            normalized_segment = normalize_points(segment)

            if len(normalized_segment) >= 2:
                predicted_segments.append(normalized_segment)

    # Collect all coordinates for map positioning.
    all_points = (
        gps_points
        + reference_points
    )

    for segment in predicted_segments:
        all_points.extend(segment)

    if not all_points:
        st.warning(
            "No valid geographical coordinates available."
        )
        return

    # Use the first valid coordinate as the map centre.
    # No coordinate averaging is required.
    center_lat, center_lon = all_points[0]

    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        control_scale=True,
    )

    # Original GPS trace: blue.
    if len(gps_points) >= 2:
        folium.PolyLine(
            locations=gps_points,
            color="blue",
            weight=3,
            opacity=0.8,
            tooltip="Original GPS Trace",
        ).add_to(map_object)

        folium.Marker(
            gps_points[0],
            tooltip="GPS Start",
            icon=folium.Icon(color="blue", icon="play"),
        ).add_to(map_object)

        folium.Marker(
            gps_points[-1],
            tooltip="GPS End",
            icon=folium.Icon(color="blue", icon="stop"),
        ).add_to(map_object)

    # Reference route: green.
    if len(reference_points) >= 2:
        folium.PolyLine(
            locations=reference_points,
            color="green",
            weight=5,
            opacity=0.8,
            tooltip="Reference Route",
        ).add_to(map_object)

    # Predicted road route: red.
    for index, segment in enumerate(predicted_segments):
        folium.PolyLine(
            locations=segment,
            color="red",
            weight=5,
            opacity=0.9,
            tooltip=f"Predicted Route - Segment {index + 1}",
        ).add_to(map_object)

    # Fit map to all available coordinates.
    min_lat = min(point[0] for point in all_points)
    max_lat = max(point[0] for point in all_points)
    min_lon = min(point[1] for point in all_points)
    max_lon = max(point[1] for point in all_points)

    map_object.fit_bounds(
        [
            [min_lat, min_lon],
            [max_lat, max_lon],
        ]
    )

    folium.LayerControl().add_to(map_object)

    st_folium(
        map_object,
        width=None,
        height=550,
        key="pathsnap_route_map",
    )
