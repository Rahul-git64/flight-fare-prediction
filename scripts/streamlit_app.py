"""Streamlit interactive client for flight fare prediction and drift inspection."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("FLIGHT_FARE_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Flight Fare Estimator & Drift Monitor",
    layout="wide",
)

st.title("Flight Fare Prediction & Supervision Console")
st.caption("Backed by FastAPI, Pydantic v2 validation, and KS/PSI distribution monitoring.")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Itinerary Fare Estimator")
    with st.form("flight_prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            airline = st.selectbox(
                "Airline",
                ["Vistara", "Indigo", "Air_India", "SpiceJet", "AirAsia", "GO_FIRST"],
            )
            source_city = st.selectbox(
                "Origin City",
                ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"],
            )
            departure_time = st.selectbox(
                "Departure Window",
                ["Morning", "Early_Morning", "Afternoon", "Evening", "Night", "Late_Night"],
            )
            stops = st.selectbox("Stops", ["zero", "one", "two_or_more"])
            travel_class = st.selectbox("Cabin Class", ["Economy", "Business"])
        with c2:
            destination_city = st.selectbox(
                "Destination City",
                ["Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai", "Delhi"],
            )
            arrival_time = st.selectbox(
                "Arrival Window",
                ["Afternoon", "Morning", "Evening", "Night", "Early_Morning", "Late_Night"],
            )
            duration = st.number_input(
                "Flight Duration (Hours)",
                min_value=0.5,
                max_value=36.0,
                value=2.25,
                step=0.25,
            )
            days_left = st.slider("Days Until Departure", min_value=1, max_value=49, value=15)

        submitted = st.form_submit_button("Predict Flight Fare")

    if submitted:
        if source_city == destination_city:
            st.error("Origin City and Destination City must be distinct.")
        else:
            payload = {
                "airline": airline,
                "source_city": source_city,
                "departure_time": departure_time,
                "stops": stops,
                "arrival_time": arrival_time,
                "destination_city": destination_city,
                "travel_class": travel_class,
                "duration": duration,
                "days_left": days_left,
            }
            try:
                resp = httpx.post(f"{API_BASE_URL}/predict", json=payload, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Estimated Fare: INR {data['predicted_price_inr']:,.2f}")
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "90% Prediction Band",
                    f"INR {data['prediction_interval_inr']['lower']:,.0f} - {data['prediction_interval_inr']['upper']:,.0f}",
                )
                m2.metric("Confidence Score", f"{data['confidence_score'] * 100:.1f}%")
                m3.metric("Inference Latency", f"{data['latency_ms']:.2f} ms")
            except Exception as exc:
                st.error(f"API request failed: {exc}")

with col_right:
    st.subheader("Service Health & Drift Status")
    if st.button("Refresh Telemetry"):
        try:
            health = httpx.get(f"{API_BASE_URL}/health", timeout=5.0).json()
            drift = httpx.get(f"{API_BASE_URL}/monitor/drift", timeout=5.0).json()
            st.json({"health": health, "drift_report": drift})
        except Exception as exc:
            st.warning(f"Could not connect to API at {API_BASE_URL}: {exc}")
