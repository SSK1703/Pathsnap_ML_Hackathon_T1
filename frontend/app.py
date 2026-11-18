import streamlit as st
import requests

from frontend.api_client import PathSnapAPIClient

from frontend.components.results import (
    display_metrics,
    display_route
)

from frontend.components.map_view import display_map


st.set_page_config(
    page_title="PathSnap Evaluator",
    page_icon="🗺️",
    layout="wide"
)


@st.cache_resource
def get_client():

    return PathSnapAPIClient()


client = get_client()


st.title("🗺️ PathSnap")
st.subheader("GPS Map Matching and Route Reconstruction")

st.markdown(
    """
    Select an evaluation trip to reconstruct its route
    using our machine learning pipeline.
    """
)

# --------------------------------------------------
# Backend status
# --------------------------------------------------

st.sidebar.header("Backend Status")

try:

    health = client.check_health()

    st.sidebar.success(
        f"Backend: {health['status']}"
    )

except requests.RequestException:

    st.sidebar.error(
        "Backend is unavailable."
    )

    st.error(
        "Cannot connect to the PathSnap API. "
        "Ensure the backend is running."
    )

    st.stop()


# --------------------------------------------------
# Load available trips
# --------------------------------------------------

try:

    trip_data = client.get_trips()

    trips = trip_data["trips"]

except requests.RequestException as error:

    st.error(
        f"Could not load trips: {error}"
    )

    st.stop()


if not trips:

    st.warning(
        "No evaluator trips have been configured yet."
    )

    st.stop()


# --------------------------------------------------
# Trip selection
# --------------------------------------------------

selected_trip = st.selectbox(
    "Select an evaluation trip",
    trips
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "Run Route Reconstruction",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Processing GPS observations and reconstructing route..."
    ):

        try:

            result = client.predict(selected_trip)

            st.session_state["prediction_result"] = result

        except requests.RequestException as error:

            st.error(
                f"Prediction request failed: {error}"
            )


# --------------------------------------------------
# Display results
# --------------------------------------------------

result = st.session_state.get("prediction_result")

if result and result["trip_id"] == selected_trip:

    st.success(
        result["message"]
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Road Segments",
            len(result["predicted_arc_ids"])
        )

    with col2:

        st.metric(
            "Processing Time",
            f"{result['processing_time_seconds']:.3f} sec"
        )

    display_metrics(result.get("metrics"))

    display_route(
        result["predicted_arc_ids"]
    )

    display_map(
        gps_points=result.get("gps_points"),
        predicted_route=result.get("predicted_route_segments"),
    )
