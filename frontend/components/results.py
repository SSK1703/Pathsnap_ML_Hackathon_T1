import streamlit as st


def display_metrics(metrics):

    if not metrics:
        st.info(
            "Evaluation metrics are not available for this response."
        )
        return

    columns = st.columns(4)

    values = [
        ("Top-1 Accuracy", metrics.get("top1_accuracy")),
        ("Macro F1", metrics.get("macro_f1")),
        ("Log Loss", metrics.get("log_loss")),
        ("Route Continuity", metrics.get("route_continuity"))
    ]

    for column, (label, value) in zip(columns, values):

        if value is None:
            column.metric(label, "N/A")
        else:
            column.metric(label, f"{value:.4f}")


def display_route(arc_ids):

    st.subheader("Predicted Road Sequence")

    if not arc_ids:
        st.warning("No predicted route was returned.")
        return

    st.write("Number of predicted segments:", len(arc_ids))

    st.code(
        " → ".join(map(str, arc_ids))
    )
