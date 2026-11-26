"""Streamlit investigator view (M9). Read-only: it shows the queues the pipeline wrote."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fraud.config import load_config


def main() -> None:
    config = load_config()
    reports = config.paths.reports
    st.set_page_config(page_title="Fraud alert queue", layout="wide")
    st.title("Fraud alert queue")
    if not (reports / "queue_summary.csv").is_file():
        st.error("No queues found. Run the pipeline first.")
        return
    summary = pd.read_csv(reports / "queue_summary.csv")
    day = st.selectbox("Day", summary["day"].tolist())
    reveal = st.toggle("Reveal true labels", value=False)
    row = summary[summary["day"] == day].iloc[0]
    queue = pd.read_csv(reports / "queues" / f"day_{int(day):03d}.csv")

    columns = st.columns(4)
    columns[0].metric("Alerts", int(row["alerts"]))
    # Outcome figures give the labels away, so they stay hidden until revealed.
    if reveal:
        columns[1].metric("Frauds caught", int(row["frauds_caught"]))
        columns[2].metric("Fraud value caught", f"{row['value_caught']:,.0f}")
        columns[3].metric("Net benefit (₹, assumed costs)", f"{row['net_benefit']:,.0f}")

    st.subheader("Ranked queue")
    st.dataframe(
        queue if reveal else queue.drop(columns="true_label"), hide_index=True, width="stretch"
    )

    st.subheader("Last 14 days")
    recent = summary[summary["day"] <= day].tail(14)
    total = recent["value_caught"] + recent["value_missed"]
    chart = pd.DataFrame(
        {
            "day": recent["day"],
            "precision@K": recent["frauds_caught"] / recent["alerts"].replace(0, float("nan")),
            "value detection rate": recent["value_caught"] / total.replace(0, float("nan")),
        }
    )
    st.line_chart(chart, x="day")


main()
