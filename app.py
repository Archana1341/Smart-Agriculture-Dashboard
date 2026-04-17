"""Streamlit app for Smart Agriculture IoT Dashboard."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from serial.tools import list_ports
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from data_store import (
    append_sensor_data,
    get_data_log_bytes,
    load_recent_data,
    reset_data_log,
)
from health_score import calculate_health_score, get_health_status, predict_yield
from insight_engine import (
    generate_insights,
    get_smart_alerts,
    irrigation_status,
    predict_crop_risks,
    soil_to_moisture_percent,
    weather_adjusted_irrigation_recommendation,
)
from serial_reader import read_sensor_data


@st.cache_resource
def _get_reader_registry() -> tuple[dict[str, threading.Thread], dict[str, queue.Queue[dict[str, Any]]]]:
    """Create one shared reader registry for all reruns/sessions in this process."""
    return {}, {}


_reader_threads, _reader_queues = _get_reader_registry()


def _start_reader_thread(port: str) -> queue.Queue[dict[str, Any]]:
    """Start serial reader thread for a port if not already running."""
    existing = _reader_threads.get(port)
    existing_queue = _reader_queues.get(port)

    if existing is not None and existing.is_alive() and existing_queue is not None:
        return existing_queue

    message_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=512)
    _reader_queues[port] = message_queue

    def worker() -> None:
        yielded_any = False
        try:
            for reading in read_sensor_data(port):
                yielded_any = True
                event = {"type": "data", "payload": reading}
                try:
                    message_queue.put_nowait(event)
                except queue.Full:
                    try:
                        message_queue.get_nowait()
                    except queue.Empty:
                        pass
                    message_queue.put_nowait(event)
        except Exception as exc:  # pragma: no cover
            message_queue.put({"type": "error", "payload": f"Reader crashed: {exc}"})
            return

        if not yielded_any:
            message_queue.put(
                {
                    "type": "error",
                    "payload": (
                        f"Could not read data from {port}. "
                        "Check COM port and Arduino connection."
                    ),
                }
            )
        else:
            message_queue.put(
                {"type": "error", "payload": f"Serial stream on {port} has stopped."}
            )

    thread = threading.Thread(target=worker, daemon=True, name=f"serial-reader-{port}")
    _reader_threads[port] = thread
    thread.start()
    return message_queue


def _initialize_session() -> None:
    """Initialize Streamlit session state defaults."""
    if "monitoring" not in st.session_state:
        st.session_state.monitoring = False
    if "selected_port" not in st.session_state:
        st.session_state.selected_port = None
    if "latest_data" not in st.session_state:
        st.session_state.latest_data = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = ""
    if "connection_status" not in st.session_state:
        st.session_state.connection_status = "Disconnected"
    if "connection_message" not in st.session_state:
        st.session_state.connection_message = ""
    if "refresh_rate" not in st.session_state:
        st.session_state.refresh_rate = 2
    if "show_history" not in st.session_state:
        st.session_state.show_history = True
    if "city" not in st.session_state:
        st.session_state.city = "Hyderabad"
    if "weather_api_key" not in st.session_state:
        try:
            st.session_state.weather_api_key = st.secrets.get("OPENWEATHER_API_KEY")
        except StreamlitSecretNotFoundError:
            st.session_state.weather_api_key = None
    if "last_data_timestamp" not in st.session_state:
        st.session_state.last_data_timestamp = None
    if "using_demo_data" not in st.session_state:
        st.session_state.using_demo_data = False
    if "selected_crop" not in st.session_state:
        st.session_state.selected_crop = "Tomato"
    if "reading_history" not in st.session_state:
        st.session_state.reading_history = []


def _is_arduino_port(port: list_ports.ListPortInfo) -> bool:
    """Heuristic detection for Arduino-compatible serial devices."""
    searchable = " ".join(
        [
            str(port.device or ""),
            str(port.description or ""),
            str(port.manufacturer or ""),
            str(port.hwid or ""),
            str(port.product or ""),
        ]
    ).lower()
    signatures = [
        "arduino",
        "uno r4",
        "uno",
        "usb serial",
        "ch340",
        "cp210",
    ]
    return any(signature in searchable for signature in signatures)


def _detect_arduino_port() -> str | None:
    """Scan serial ports and return most likely Arduino port."""
    ports = list(list_ports.comports())
    if not ports:
        return None

    # Prefer explicit Arduino matches first.
    for port in ports:
        if _is_arduino_port(port):
            return str(port.device)

    # Fallback: use first available COM device when Arduino signature is not visible.
    return str(ports[0].device)

    return None


def _handle_disconnect(message: str) -> None:
    """Update session state after serial disconnection/error."""
    st.session_state.connection_status = "Disconnected"
    st.session_state.monitoring = False
    st.session_state.selected_port = None
    st.session_state.last_error = message
    st.session_state.connection_message = ""


def _auto_connect_serial() -> None:
    """Auto-detect and auto-connect to Arduino serial device."""
    selected_port = st.session_state.selected_port
    detected_port = _detect_arduino_port()

    # Auto-switch to detected USB port if needed (plug-and-play behavior).
    if detected_port and selected_port != detected_port:
        st.session_state.selected_port = detected_port
        st.session_state.monitoring = True
        st.session_state.connection_status = "Connected"
        st.session_state.last_error = ""
        st.session_state.connection_message = f"Connected to Arduino on {detected_port}"
        _start_reader_thread(detected_port)
        return

    if selected_port:
        # Keep existing connection stable and only revive reader if needed.
        running_thread = _reader_threads.get(selected_port)
        if running_thread is None or not running_thread.is_alive():
            _start_reader_thread(selected_port)

        st.session_state.connection_status = "Connected"
        st.session_state.monitoring = True
        st.session_state.last_error = ""
        return

    if not detected_port:
        st.session_state.connection_status = "Disconnected"
        st.session_state.monitoring = False
        st.session_state.selected_port = None
        if not st.session_state.last_error:
            st.session_state.last_error = "No Arduino detected. Please connect Arduino UNO R4 WiFi via USB."
        return

    # Keep connected status while thread remains alive.
    thread = _reader_threads.get(detected_port)
    if thread is not None and thread.is_alive():
        st.session_state.connection_status = "Connected"
        st.session_state.monitoring = True


def _drain_queue(port: str) -> None:
    """Drain latest serial events into session state and persistent storage."""
    message_queue = _start_reader_thread(port)

    while True:
        try:
            event = message_queue.get_nowait()
        except queue.Empty:
            break

        print(f"QUEUE EVENT: {event}")

        event_type = event.get("type")
        payload = event.get("payload")
        print(f"PAYLOAD: {payload}")

        if event_type == "data" and isinstance(payload, dict):
            print(f"DATA RECEIVED: {payload}")
            st.session_state.latest_data = payload
            st.session_state.connection_status = "Connected"
            current_time = time.time()
            st.session_state.last_data_timestamp = current_time
            st.session_state.using_demo_data = False
            try:
                signature = (
                    int(payload.get("soil", -1)),
                    float(payload.get("temperature", 0.0)),
                    float(payload.get("humidity", 0.0)),
                )
                if signature != st.session_state.get("last_logged_signature"):
                    append_sensor_data(payload)
                    st.session_state.last_logged_signature = signature
            except Exception as exc:
                st.session_state.last_error = f"Failed to write data log: {exc}"
        elif event_type == "error":
            _handle_disconnect(str(payload))


@st.cache_data(ttl=600)
def _fetch_weather(city: str, api_key: str | None) -> tuple[dict[str, Any] | None, str]:
    """Fetch current weather from OpenWeatherMap API."""
    if not city:
        return None, "Enter city to load weather data."
    if not api_key:
        return None, "Weather integration disabled – no API key configured."

    endpoint = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(endpoint, params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return None, f"Weather API request failed: {exc}"

    main = payload.get("main", {})
    weather_items = payload.get("weather", [{}])
    weather_item = weather_items[0] if weather_items else {}
    rain_raw = payload.get("rain", {})
    rain_mm = float(rain_raw.get("1h", rain_raw.get("3h", 0.0))) if isinstance(rain_raw, dict) else 0.0

    weather = {
        "temp": float(main.get("temp", 0.0)),
        "humidity": float(main.get("humidity", 0.0)),
        "condition": str(weather_item.get("description", "Unknown")).title(),
        "weather_main": str(weather_item.get("main", "")),
        "icon": str(weather_item.get("icon", "")),
        "rain_mm": rain_mm,
    }
    return weather, ""


def _gauge_chart(title: str, value: float, axis_max: float, steps: list[dict[str, Any]]) -> go.Figure:
    """Build one Plotly gauge meter."""
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [0, axis_max]},
                "bar": {"color": "#1f2937"},
                "steps": steps,
            },
        )
    )
    figure.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=20))
    return figure


def _prepare_history(data: pd.DataFrame) -> pd.DataFrame:
    """Clean, type-cast, and index data for plotting."""
    if data.empty:
        return data

    history = data.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
    history = history.dropna(subset=["timestamp"]).sort_values("timestamp")

    for field in ["soil", "temperature", "humidity"]:
        history[field] = pd.to_numeric(history[field], errors="coerce")

    history = history.dropna(subset=["soil", "temperature", "humidity"])
    if history.empty:
        return history

    # Use inverted soil moisture calculation (high raw value = dry soil)
    history["soil_moisture_percent"] = (
        (1023.0 - history["soil"].astype(float).clip(lower=0, upper=1023)) / 1023.0 * 100.0
    )
    return history.set_index("timestamp")


def _zone_color(moisture_percent: float) -> str:
    """Map moisture to zone color."""
    if moisture_percent < 30:
        return "#d73027"
    if moisture_percent <= 60:
        return "#f4a300"
    return "#1a9850"


def _prediction_frame(series: pd.Series, forecast_steps: int = 5) -> pd.Series:
    """Build simple linear forecast extension from current time series."""
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return pd.Series(dtype=float)

    x = pd.Series(range(len(clean_series)), index=clean_series.index, dtype=float)
    slope, intercept = np.polyfit(x.values, clean_series.values, 1)

    if len(clean_series.index) >= 2:
        step_delta = clean_series.index[-1] - clean_series.index[-2]
    else:
        step_delta = pd.Timedelta(seconds=st.session_state.refresh_rate)

    start = len(clean_series)
    future_index = [clean_series.index[-1] + step_delta * (i + 1) for i in range(forecast_steps)]
    future_values = [slope * (start + i) + intercept for i in range(forecast_steps)]
    return pd.Series(future_values, index=future_index, dtype=float)


def _line_with_trend_and_forecast(
    history: pd.DataFrame, column: str, title: str, rolling_window: int = 5
) -> go.Figure:
    """Build a chart with actual values, moving average, and forecast line."""
    fig = go.Figure()
    series = history[column]
    moving_avg = series.rolling(window=rolling_window, min_periods=1).mean()
    forecast = _prediction_frame(series)

    # Color mapping based on metric type
    color_map = {
        "soil_moisture_percent": {"actual": "#1976d2", "avg": "#64b5f6", "forecast": "#0d47a1"},
        "temperature": {"actual": "#f57c00", "avg": "#ffb74d", "forecast": "#e65100"},
        "humidity": {"actual": "#7b1fa2", "avg": "#ba68c8", "forecast": "#4a148c"},
    }
    
    colors = color_map.get(column, {"actual": "#2e7d32", "avg": "#81c784", "forecast": "#1b5e20"})

    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            name="Actual",
            line=dict(color=colors["actual"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=moving_avg.index,
            y=moving_avg,
            mode="lines",
            name="Moving Avg",
            line=dict(color=colors["avg"], dash="dash", width=2),
        )
    )
    if not forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast,
                mode="lines",
                name="Forecast",
                line=dict(color=colors["forecast"], dash="dot", width=2),
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f6fbf6",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        transition={"duration": 400},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#e6efe7")
    fig.update_yaxes(gridcolor="#e6efe7")
    return fig


def _estimate_irrigation_minutes(history: pd.DataFrame) -> tuple[str, float | None]:
    """Estimate time until soil moisture drops below 40% using recent trend."""
    if history.empty or "soil_moisture_percent" not in history.columns:
        return "Not enough data for irrigation timing yet.", None

    recent = history.tail(20)
    if len(recent) < 3:
        return "Collect a few more readings to estimate irrigation timing.", None

    time_delta_minutes = (
        recent.index.to_series().astype("int64") - recent.index[0].value
    ) / 60_000_000_000
    slope, intercept = np.polyfit(time_delta_minutes.values, recent["soil_moisture_percent"].values, 1)
    current_moisture = float(recent["soil_moisture_percent"].iloc[-1])
    threshold = 40.0

    if slope >= -0.01:
        return "Soil moisture is stable or increasing. No immediate irrigation predicted.", None

    if current_moisture <= threshold:
        return "Soil moisture is already below 40%. Irrigate now.", 0.0

    minutes_remaining = max(0.0, (current_moisture - threshold) / abs(float(slope)))
    return f"Estimated irrigation required in {minutes_remaining:.0f} minutes", minutes_remaining


def _build_realtime_environment_chart(history: pd.DataFrame) -> go.Figure:
    """Create a live multi-metric chart for soil moisture, temperature, and humidity."""
    chart = go.Figure()
    if history.empty:
        chart.update_layout(
            title="Live Environment Signals",
            height=360,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        return chart

    recent = history.tail(60)
    chart.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["soil_moisture_percent"],
            mode="lines+markers",
            name="Soil Moisture (%)",
            line=dict(color="#1976d2", width=3),
        )
    )
    chart.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["temperature"],
            mode="lines+markers",
            name="Temperature (C)",
            line=dict(color="#f57c00", width=2),
            yaxis="y2",
        )
    )
    chart.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["humidity"],
            mode="lines+markers",
            name="Humidity (%)",
            line=dict(color="#7b1fa2", width=2),
            yaxis="y2",
        )
    )
    chart.update_layout(
        title="Live Environment Signals (Auto-updating)",
        template="plotly_white",
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f6fbf6",
        margin=dict(l=20, r=20, t=45, b=20),
        hovermode="x unified",
        transition={"duration": 500},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Soil Moisture (%)", range=[0, 100]),
        yaxis2=dict(title="Temp / Humidity", overlaying="y", side="right"),
    )
    chart.update_xaxes(gridcolor="#e6efe7")
    chart.update_yaxes(gridcolor="#e6efe7")
    return chart


def _build_zone_heatmap(soil_percent: float) -> go.Figure:
    """Render compact three-zone farm heatmap."""
    zone_values = [
        max(0.0, min(100.0, soil_percent)),
        max(0.0, min(100.0, soil_percent - 12.0)),
        max(0.0, min(100.0, soil_percent + 8.0)),
    ]
    fig = go.Figure(
        go.Heatmap(
            z=[zone_values],
            x=["Zone A", "Zone B", "Zone C"],
            y=["Farm"],
            colorscale=[
                [0.0, "#e8f5e9"],
                [0.4, "#c8e6c9"],
                [0.7, "#66bb6a"],
                [1.0, "#2e7d32"],
            ],
            zmin=0,
            zmax=100,
            text=[[f"{v:.0f}%" for v in zone_values]],
            texttemplate="%{x}<br>%{text}",
            hovertemplate="%{x}: %{z:.1f}%<extra></extra>",
            showscale=False,
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f6fbf6",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(side="top"),
        yaxis=dict(visible=False),
    )
    return fig


def _format_last_updated(last_data_timestamp: float | None) -> str:
    """Convert epoch timestamp to readable hh:mm:ss format."""
    if not last_data_timestamp:
        return "No sensor data yet"
    return time.strftime("%H:%M:%S", time.localtime(last_data_timestamp))


def _soil_condition_label(soil_moisture_percent: float) -> str:
    """Return farmer-friendly soil condition label."""
    if soil_moisture_percent < 20:
        return "Very Dry"
    if soil_moisture_percent < 35:
        return "Dry"
    if soil_moisture_percent <= 70:
        return "Good"
    return "Too Wet"


def _sensor_status_snapshot(
    latest_data: dict[str, Any] | None,
    last_data_timestamp: float | None,
    refresh_rate_seconds: int,
) -> tuple[dict[str, str], bool]:
    """Return per-sensor status labels and stale-data warning state."""
    stale_threshold = max(12, int(refresh_rate_seconds) * 3)
    is_stale = (
        last_data_timestamp is None
        or (time.time() - float(last_data_timestamp)) > stale_threshold
    )

    if latest_data is None:
        return {
            "soil": "Not responding",
            "temperature": "Error",
            "humidity": "Error",
        }, True

    soil_ok = 0 <= int(latest_data.get("soil", -1)) <= 1023
    temp_ok = -20.0 <= float(latest_data.get("temperature", -999.0)) <= 85.0
    humidity_ok = 0.0 <= float(latest_data.get("humidity", -1.0)) <= 100.0

    return {
        "soil": "Active" if soil_ok else "Not responding",
        "temperature": "Active" if temp_ok else "Error",
        "humidity": "Active" if humidity_ok else "Error",
    }, is_stale


def _ai_farm_decision_engine(
    soil_moisture: float,
    temperature: float,
    humidity: float,
    irrigation_eta_minutes: float | None,
) -> tuple[str, str, str]:
    """Generate dynamic irrigation decision, reasoning, and priority label."""
    if soil_moisture < 20:
        return (
            "Irrigate immediately",
            "Soil moisture is critically low and can rapidly stress crops.",
            "critical",
        )

    if soil_moisture < 30:
        wait_minutes = 5 if temperature > 34 else 10
        return (
            f"Irrigate in {wait_minutes} minutes",
            "Soil is very dry. Schedule irrigation now to prevent moisture collapse.",
            "warning",
        )

    if soil_moisture < 45:
        if irrigation_eta_minutes is not None:
            wait_minutes = max(10, min(45, int(round(irrigation_eta_minutes))))
        else:
            wait_minutes = 20 if temperature > 33 else 30
        return (
            f"Wait {wait_minutes} minutes before irrigation",
            "Moisture is moderate but trending toward dry conditions.",
            "warning",
        )

    if 45 <= soil_moisture <= 75 and 18 <= temperature <= 32 and 40 <= humidity <= 75:
        return (
            "Conditions are optimal",
            "Current soil, temperature, and humidity support healthy crop growth.",
            "optimal",
        )

    if temperature > 36:
        return (
            "Irrigate lightly during early morning",
            "Heat is high; prefer cooler-hour irrigation to reduce evaporation.",
            "warning",
        )

    return (
        "Monitor conditions and re-evaluate in 20 minutes",
        "Environment is acceptable but not fully optimal.",
        "info",
    )


def _build_notification_cards(
    soil_moisture: float,
    temperature: float,
    humidity: float,
    stale_data: bool,
) -> list[tuple[str, str, str]]:
    """Create smart notifications as (level, title, message)."""
    notifications: list[tuple[str, str, str]] = []

    if soil_moisture < 30:
        notifications.append(("critical", "⚠ Soil moisture critically low", "Irrigation is required to avoid drought stress."))
    if temperature > 36:
        notifications.append(("warning", "🔥 Heat stress risk", "High temperature can reduce yield and increase water demand."))
    if humidity > 82:
        notifications.append(("warning", "🦠 Disease risk", "Humidity is high enough to increase fungal disease probability."))
    if stale_data:
        notifications.append(("info", "Sensor data delayed", "Recent sensor updates are missing. Check device connectivity."))

    if not notifications:
        notifications.append(("success", "All systems stable", "No critical environmental alerts at this time."))

    return notifications


def _smart_suggestions(
    soil_moisture: float,
    temperature: float,
    humidity: float,
    weather_data: dict[str, Any] | None,
    recommended_crop: str,
) -> list[str]:
    """Generate contextual smart suggestions for farm operations."""
    suggestions: list[str] = []

    if temperature > 33:
        suggestions.append("Increase irrigation frequency during afternoon heat windows.")
    if soil_moisture < 35:
        suggestions.append("Consider switching to drought-resistant crops in this plot.")
    if humidity > 80:
        suggestions.append("Improve airflow around plants to reduce fungal pressure.")
    if weather_data and float(weather_data.get("rain_mm", 0.0)) > 0:
        suggestions.append("Rain is expected, reduce your next irrigation cycle.")

    suggestions.append(f"Current conditions favor {recommended_crop} cultivation.")
    suggestions.append("Use early-morning watering to improve water-use efficiency.")

    deduplicated: list[str] = []
    seen = set()
    for suggestion in suggestions:
        if suggestion not in seen:
            deduplicated.append(suggestion)
            seen.add(suggestion)
    return deduplicated


def _crop_recommendation(avg_soil: float, avg_temp: float, avg_hum: float) -> tuple[str, int]:
    """Recommend crop based on environment averages with confidence score."""
    crop_profiles = {
        "Rice": {"soil": 75.0, "temp": 29.0, "hum": 78.0},
        "Tomato": {"soil": 58.0, "temp": 25.0, "hum": 62.0},
        "Wheat": {"soil": 45.0, "temp": 20.0, "hum": 52.0},
        "Corn": {"soil": 55.0, "temp": 26.0, "hum": 60.0},
    }

    best_crop = "Tomato"
    best_score = -1.0

    for crop, profile in crop_profiles.items():
        soil_score = max(0.0, 100.0 - abs(avg_soil - profile["soil"]) * 2.0)
        temp_score = max(0.0, 100.0 - abs(avg_temp - profile["temp"]) * 4.0)
        hum_score = max(0.0, 100.0 - abs(avg_hum - profile["hum"]) * 2.5)
        combined = 0.4 * soil_score + 0.35 * temp_score + 0.25 * hum_score
        if combined > best_score:
            best_score = combined
            best_crop = crop

    confidence = int(max(35.0, min(98.0, best_score)))
    return best_crop, confidence


def _weather_impact_alert(weather: dict[str, Any] | None) -> tuple[str, str]:
    """Return severity and weather impact message for advisor panel."""
    if not weather:
        return "info", "Weather data unavailable. Recommendations are based on sensors only."

    rain_mm = float(weather.get("rain_mm", 0.0))
    temp = float(weather.get("temp", 0.0))
    humidity = float(weather.get("humidity", 0.0))

    if rain_mm > 1.0:
        return "warning", "Rain expected. Reduce irrigation volume for the next cycle."
    if temp >= 34.0:
        return "warning", "High outdoor temperature detected. Prefer early-morning irrigation."
    if humidity >= 85.0:
        return "warning", "High ambient humidity may increase fungal risk."
    return "success", "Weather impact is low. Standard irrigation schedule is suitable."


def get_crop_thresholds(crop_name: str) -> dict[str, tuple[float, float]]:
    """Return crop-specific ideal ranges used only for advanced insights."""
    thresholds = {
        "Rice": {
            "soil_moisture": (55.0, 85.0),
            "temperature": (20.0, 32.0),
            "humidity": (60.0, 85.0),
        },
        "Wheat": {
            "soil_moisture": (35.0, 60.0),
            "temperature": (15.0, 28.0),
            "humidity": (40.0, 65.0),
        },
        "Tomato": {
            "soil_moisture": (45.0, 70.0),
            "temperature": (18.0, 30.0),
            "humidity": (45.0, 70.0),
        },
    }
    return thresholds.get(crop_name, thresholds["Tomato"])


def predict_moisture_trend(data_history: list[dict[str, float]]) -> str:
    """Predict near-term moisture risk from last 5 readings."""
    if len(data_history) < 5:
        return "Collecting trend data (need at least 5 readings)."

    last_five = data_history[-5:]
    moisture = [float(item.get("soil_moisture", 0.0)) for item in last_five]
    strictly_decreasing = all(moisture[i] > moisture[i + 1] for i in range(len(moisture) - 1))
    total_drop = moisture[0] - moisture[-1]

    if strictly_decreasing and (moisture[-1] < 35.0 or total_drop >= 8.0):
        return "Soil moisture likely to become critical soon."
    if strictly_decreasing:
        return "Soil moisture is decreasing gradually. Monitor irrigation timing."
    return "Soil moisture trend is stable."


def generate_explanation(soil: float, temp: float, humidity: float) -> str:
    """Generate human-readable decision explanation from current conditions."""
    reasons: list[str] = []

    if soil < 30:
        reasons.append("soil moisture is low")
    elif soil > 80:
        reasons.append("soil moisture is high")

    if temp > 34:
        reasons.append("temperature is high, increasing evaporation")
    elif temp < 15:
        reasons.append("temperature is low for active crop growth")

    if humidity < 35:
        reasons.append("humidity is low, so plants lose water faster")
    elif humidity > 80:
        reasons.append("humidity is high, increasing disease pressure")

    if reasons:
        return "Irrigation attention needed because " + ", and ".join(reasons) + "."
    return "Conditions are balanced, so standard irrigation scheduling is appropriate."


def calculate_water_efficiency(soil_value: float) -> int:
    """Calculate water optimization score (0-100) from soil moisture percentage."""
    optimal_low, optimal_high = 45.0, 70.0
    if optimal_low <= soil_value <= optimal_high:
        return 95

    if soil_value < optimal_low:
        penalty = min(80.0, (optimal_low - soil_value) * 1.8)
    else:
        penalty = min(80.0, (soil_value - optimal_high) * 1.5)

    return int(max(10.0, 95.0 - penalty))


def detect_disease_risk(temp: float, humidity: float) -> tuple[str, str]:
    """Detect simple fungal disease risk from humidity and temperature."""
    if humidity > 70 and 20 <= temp <= 30:
        return "warning", "High risk of fungal disease."
    return "success", "Disease risk is currently low."


def calculate_decision_confidence(soil: float, temp: float, humidity: float) -> tuple[str, int, str, float, float, float]:
    """Compute explainable AI confidence and reason breakdown."""
    soil_score = max(0.0, min(100.0, 100.0 - float(soil)))
    temp_score = max(0.0, min(100.0, abs(float(temp) - 25.0) * 4.0))
    humidity_score = max(0.0, min(100.0, abs(float(humidity) - 60.0) * 2.0))

    confidence = (
        soil_score * 0.6
        + temp_score * 0.25
        + humidity_score * 0.15
    )
    confidence = int(min(100.0, confidence))

    if soil < 30:
        decision = "Irrigate Now"
    elif soil < 45:
        decision = "Consider Irrigation"
    else:
        decision = "Optimal"

    if confidence > 80:
        trust = "High"
    elif confidence > 50:
        trust = "Medium"
    else:
        trust = "Low"

    return decision, confidence, trust, soil_score, temp_score, humidity_score


def get_fake_weather(temperature: float, humidity: float) -> dict[str, Any]:
    """Infer weather-like insights from live sensor values without API dependency."""
    if temperature > 30:
        condition = "Hot"
    elif temperature < 20:
        condition = "Cool"
    else:
        condition = "Moderate"

    if humidity > 70:
        rain = "High"
    elif humidity > 50:
        rain = "Medium"
    else:
        rain = "Low"

    return {
        "temp": float(temperature),
        "humidity": float(humidity),
        "condition": condition,
        "rain": rain,
    }


st.set_page_config(page_title="Smart Agriculture IoT Dashboard", layout="wide")
_initialize_session()

st.markdown(
    """
    <style>
    :root {
        --primary: #2e7d32;
        --primary-strong: #1b5e20;
        --bg-soft: #f4faf4;
        --bg-card: #ffffff;
        --border-soft: #dbe8dc;
        --text-main: #16311a;
        --text-muted: #4f6b53;
    }
    .main {
        background: linear-gradient(180deg, #f8fcf8 0%, #eff7ef 100%);
    }
    div.block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.4rem;
    }
    .section-title {
        color: var(--primary-strong);
        font-size: 1.06rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }
    .dashboard-card {
        background: var(--bg-card);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 10px 24px rgba(28, 62, 35, 0.08);
        height: 100%;
    }
    .metric-title {
        color: var(--text-muted);
        font-size: 0.88rem;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        color: var(--text-main);
        font-size: 1.45rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .status-pill {
        display: inline-block;
        background: #e8f5e9;
        color: var(--primary-strong);
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.8rem;
        font-weight: 700;
        margin-top: 0.45rem;
    }
    .alert-card {
        border-radius: 14px;
        padding: 11px 13px;
        margin-bottom: 0.5rem;
        border: 1px solid;
        font-size: 0.92rem;
    }
    .alert-critical {
        background: #fff3f3;
        border-color: #ffcccb;
        color: #8d1f1f;
    }
    .alert-warning {
        background: #fff8ec;
        border-color: #ffe0b2;
        color: #8a5a12;
    }
    .alert-success {
        background: #edf9ef;
        border-color: #cbe7cf;
        color: #1f6b2a;
    }
    .alert-info {
        background: #f1f7f1;
        border-color: #d3e5d5;
        color: #2f5b36;
    }
    .panel-muted {
        color: var(--text-muted);
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌾 Smart Agriculture IoT Dashboard")
st.info("Smart Weather Inference is active. Scroll down to the Smart Weather Insights section.")

with st.sidebar:
    st.header("⚙ Connection & Controls")
    if st.session_state.connection_status == "Connected" and st.session_state.selected_port:
        st.success(f"Connection Status: Connected ({st.session_state.selected_port})")
    else:
        st.warning("Connection Status: Disconnected")

    st.caption("Arduino port is auto-detected (plug-and-play mode).")
    available_ports = [str(port.device) for port in list_ports.comports()]
    if available_ports:
        selected_port = st.session_state.selected_port
        default_index = 0
        if selected_port in available_ports:
            default_index = available_ports.index(selected_port)

        manual_port = st.selectbox(
            "Serial Port (manual)",
            options=available_ports,
            index=default_index,
            help="If auto-detect fails, select the Arduino COM port manually.",
        )
        if st.button("Connect Selected Port", use_container_width=True):
            st.session_state.selected_port = manual_port
            st.session_state.monitoring = True
            st.session_state.connection_status = "Connected"
            st.session_state.last_error = ""
            st.session_state.connection_message = f"Connected to Arduino on {manual_port}"
            _start_reader_thread(manual_port)

        if st.button("Hard Connect COM10", use_container_width=True):
            st.session_state.selected_port = "COM10"
            st.session_state.monitoring = True
            st.session_state.connection_status = "Connected"
            st.session_state.last_error = ""
            st.session_state.connection_message = "Connected to Arduino on COM10 (forced)"
            _start_reader_thread("COM10")
    else:
        st.caption("No COM ports detected right now.")
        if st.button("Hard Connect COM10", use_container_width=True):
            st.session_state.selected_port = "COM10"
            st.session_state.monitoring = True
            st.session_state.connection_status = "Connected"
            st.session_state.last_error = ""
            st.session_state.connection_message = "Connected to Arduino on COM10 (forced)"
            _start_reader_thread("COM10")

    city_input = st.text_input("Farm City", value=st.session_state.city)
    weather_api_key = st.text_input(
        "OpenWeather API Key",
        value=st.session_state.weather_api_key or "",
        type="password",
        help="You can also set OPENWEATHER_API_KEY in Streamlit secrets.",
    )
    refresh_rate = st.slider("Sensor Refresh Rate (sec)", min_value=1, max_value=10, value=st.session_state.refresh_rate)
    show_history = st.toggle("Show Historical Graphs", value=st.session_state.show_history)
    crop_options = ["Rice", "Wheat", "Tomato"]
    selected_crop = st.session_state.selected_crop if st.session_state.selected_crop in crop_options else "Tomato"
    selected_crop = st.selectbox(
        "Target Crop",
        options=crop_options,
        index=crop_options.index(selected_crop),
        help="Used only for Advanced AI Insights.",
    )
    st.session_state.selected_crop = selected_crop
    st.session_state.refresh_rate = refresh_rate
    st.session_state.show_history = show_history
    st.session_state.city = city_input.strip()
    st.session_state.weather_api_key = weather_api_key.strip() or None

    if st.button("Reconnect Device", use_container_width=True):
        st.session_state.selected_port = None
        st.session_state.monitoring = False
        st.session_state.last_error = ""
        _auto_connect_serial()

    csv_data = get_data_log_bytes()
    st.download_button(
        "Download CSV Data",
        data=csv_data,
        file_name="data_log.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if st.button("Reset Historical Data", type="secondary", use_container_width=True):
        reset_data_log()
        st.success("Historical data reset successfully.")

    if st.button("Simulate Irrigation", use_container_width=True):
        if st.session_state.latest_data:
            current = dict(st.session_state.latest_data)
            current_percent = soil_to_moisture_percent(int(current["soil"]))
            simulated_percent = min(100.0, current_percent + 20.0)
            # Convert moisture percent back to inverted raw sensor value.
            current["soil"] = int(1023 - (simulated_percent / 100.0) * 1023)
            st.session_state.latest_data = current
            append_sensor_data(current)
            st.success("Irrigation simulated. Soil moisture increased.")
        else:
            st.info("No live reading available yet for simulation.")

_auto_connect_serial()

if st.session_state.monitoring and st.session_state.selected_port:
    _drain_queue(st.session_state.selected_port)

if st.session_state.connection_message:
    st.success(st.session_state.connection_message)
    st.session_state.connection_message = ""

if st.session_state.last_error:
    if st.session_state.connection_status == "Disconnected":
        st.warning(st.session_state.last_error)
    else:
        st.error(st.session_state.last_error)

latest = st.session_state.latest_data
weather_data, weather_error = _fetch_weather(
    st.session_state.city,
    st.session_state.weather_api_key
)

data_source_label = "Live Serial"
if latest is None:
    # Do not use historical values for current status panels.
    latest = {
        "soil": 1023,
        "temperature": 0.0,
        "humidity": 0.0,
        "distance": None,
        "tank": "Unknown",
    }
    st.session_state.using_demo_data = False
    data_source_label = "Waiting Live Data"

# Waiting state depends only on whether latest_data exists.
waiting_live_data = st.session_state.latest_data is None
if waiting_live_data:
    data_source_label = "Waiting Live Data"
else:
    data_source_label = "Live Serial"

soil_percent = soil_to_moisture_percent(int(latest["soil"]))
temperature_value = float(latest["temperature"])
humidity_value = float(latest["humidity"])
health_score = calculate_health_score(
    {
        "soil_moisture": soil_percent,
        "temperature": temperature_value,
        "humidity": humidity_value,
    }
)

inferred_weather: dict[str, Any] | None = None
try:
    if st.session_state.latest_data is not None:
        inferred_weather = get_fake_weather(temperature_value, humidity_value)
except Exception:
    inferred_weather = None

try:
    if st.session_state.latest_data is not None:
        st.session_state.reading_history.append(
            {
                "soil_moisture": float(soil_percent),
                "temperature": float(temperature_value),
                "humidity": float(humidity_value),
                "timestamp": time.time(),
            }
        )
        st.session_state.reading_history = st.session_state.reading_history[-30:]
except Exception:
    pass

try:
    history_raw = load_recent_data(limit=500)
except Exception as exc:
    history_raw = pd.DataFrame()
    st.error(f"Failed to load historical data: {exc}")

history = _prepare_history(history_raw)

if not history.empty:
    avg_soil_for_crop = float(history["soil_moisture_percent"].tail(20).mean())
    avg_temp_for_crop = float(history["temperature"].tail(20).mean())
    avg_hum_for_crop = float(history["humidity"].tail(20).mean())
else:
    avg_soil_for_crop = soil_percent
    avg_temp_for_crop = temperature_value
    avg_hum_for_crop = humidity_value

crop_name, crop_confidence = _crop_recommendation(
    avg_soil_for_crop,
    avg_temp_for_crop,
    avg_hum_for_crop,
)

irrigation_timing_msg, irrigation_minutes = _estimate_irrigation_minutes(history)
weather_alert_level, weather_alert_message = _weather_impact_alert(weather_data)

health_label, _ = get_health_status(health_score)
sensor_states, stale_data = _sensor_status_snapshot(
    st.session_state.latest_data,
    st.session_state.last_data_timestamp,
    st.session_state.refresh_rate,
)

if waiting_live_data:
    sensor_states = {
        "soil": "Not responding",
        "temperature": "Error",
        "humidity": "Error",
    }
    stale_data = True

decision_title, decision_reason, decision_level = _ai_farm_decision_engine(
    soil_percent,
    temperature_value,
    humidity_value,
    irrigation_minutes,
)
smart_notifications = _build_notification_cards(
    soil_percent,
    temperature_value,
    humidity_value,
    stale_data,
)
suggestions = _smart_suggestions(
    soil_percent,
    temperature_value,
    humidity_value,
    weather_data,
    st.session_state.selected_crop,
)

if waiting_live_data:
    smart_notifications = [
        ("info", "Waiting for live sensor data", "Plug Arduino USB and close Serial Monitor to start live updates.")
    ]

st.markdown("<div class='section-title'>Live Sensor Overview</div>", unsafe_allow_html=True)
o1, o2, o3, o4 = st.columns(4)
with o1:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-title'>Soil Moisture</div>", unsafe_allow_html=True)
    if waiting_live_data:
        st.markdown("<div class='metric-value'>--</div>", unsafe_allow_html=True)
        st.progress(0.0)
        st.caption("Raw sensor value: Waiting for live packet")
        st.caption("Soil condition: Waiting")
    else:
        st.markdown(f"<div class='metric-value'>{soil_percent:.1f}%</div>", unsafe_allow_html=True)
        st.progress(max(0.0, min(1.0, soil_percent / 100.0)))
        st.caption(f"Raw sensor value: {int(latest['soil'])}")
        st.caption(f"Soil condition: {_soil_condition_label(soil_percent)}")
    st.markdown("</div>", unsafe_allow_html=True)

with o2:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-title'>Temperature</div>", unsafe_allow_html=True)
    if waiting_live_data:
        st.markdown("<div class='metric-value'>--</div>", unsafe_allow_html=True)
        st.progress(0.0)
        st.caption("Waiting for first live temperature value")
    else:
        temp_progress = max(0.0, min(1.0, temperature_value / 50.0))
        st.markdown(f"<div class='metric-value'>{temperature_value:.1f} °C</div>", unsafe_allow_html=True)
        st.progress(temp_progress)
        st.caption("Optimal band: 20°C to 32°C")
    st.markdown("</div>", unsafe_allow_html=True)

with o3:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-title'>Humidity</div>", unsafe_allow_html=True)
    if waiting_live_data:
        st.markdown("<div class='metric-value'>--</div>", unsafe_allow_html=True)
        st.progress(0.0)
        st.caption("Waiting for first live humidity value")
    else:
        st.markdown(f"<div class='metric-value'>{humidity_value:.1f}%</div>", unsafe_allow_html=True)
        st.progress(max(0.0, min(1.0, humidity_value / 100.0)))
        st.caption("Target band: 40% to 75%")
    st.markdown("</div>", unsafe_allow_html=True)

with o4:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-title'>Crop Health Score</div>", unsafe_allow_html=True)
    if waiting_live_data:
        st.markdown("<div class='metric-value'>--/100</div>", unsafe_allow_html=True)
        st.progress(0.0)
        st.markdown("<span class='status-pill'>Waiting</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='metric-value'>{health_score}/100</div>", unsafe_allow_html=True)
        st.progress(max(0.0, min(1.0, health_score / 100.0)))
        st.markdown(f"<span class='status-pill'>{health_label}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>AI Farm Decision Engine</div>", unsafe_allow_html=True)
decision_col, status_col = st.columns([1.45, 1.0])

with decision_col:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    if waiting_live_data:
        st.markdown("### Waiting for live sensor data")
        st.info("Connect Arduino and wait for first packet to get decision recommendations.")
    else:
        st.markdown(f"### {decision_title}")
        st.markdown(f"<div class='panel-muted'>{decision_reason}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='panel-muted' style='margin-top:0.55rem;'><strong>Irrigation State:</strong> {irrigation_status(latest)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='panel-muted'><strong>Weather-adjusted recommendation:</strong> {weather_adjusted_irrigation_recommendation(latest, weather_data)}</div>",
            unsafe_allow_html=True,
        )
        if irrigation_minutes is not None:
            st.markdown(
                f"<div class='panel-muted'><strong>Trend estimate:</strong> {irrigation_timing_msg}</div>",
                unsafe_allow_html=True,
            )
        if decision_level == "critical":
            st.error("Priority: Immediate action required")
        elif decision_level == "warning":
            st.warning("Priority: High")
        elif decision_level == "optimal":
            st.success("Priority: Optimal")
        else:
            st.info("Priority: Monitoring")
    st.markdown("</div>", unsafe_allow_html=True)

with status_col:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("### Sensor Status Monitor")
    st.markdown(f"- Data source: **{data_source_label}**")
    st.markdown(f"- Soil sensor: **{sensor_states['soil']}**")
    st.markdown(f"- Temperature sensor: **{sensor_states['temperature']}**")
    st.markdown(f"- Humidity sensor: **{sensor_states['humidity']}**")
    st.markdown(
        f"<div class='panel-muted'>Last updated: {_format_last_updated(st.session_state.last_data_timestamp)}</div>",
        unsafe_allow_html=True,
    )
    if stale_data:
        st.warning("No recent sensor payload received. Check Arduino connection.")
    if data_source_label == "Waiting Live Data":
        st.info("No Arduino packet received yet. Please check COM port and cable.")
        st.info("If using Arduino IDE, close Serial Monitor first. Only one app can read COM10 at a time.")
    st.write("DEBUG latest_data:", st.session_state.latest_data)
    st.write("DEBUG last_data_timestamp:", st.session_state.last_data_timestamp)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Smart Notifications</div>", unsafe_allow_html=True)
for level, title, message in smart_notifications:
    css_level = "critical" if level == "critical" else level
    st.markdown(
        f"<div class='alert-card alert-{css_level}'><strong>{title}</strong><br>{message}</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-title'>Smart Suggestions</div>", unsafe_allow_html=True)
sug_col, insight_col = st.columns([1.1, 1.1])
rotation_start = int(time.time() / max(2, st.session_state.refresh_rate))
display_count = min(3, len(suggestions))

with sug_col:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("### Rotating Intelligent Suggestions")
    for idx in range(display_count):
        suggestion_index = (rotation_start + idx) % len(suggestions)
        st.markdown(f"- {suggestions[suggestion_index]}")
    st.markdown("</div>", unsafe_allow_html=True)

with insight_col:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("### Existing Insights")
    for insight in generate_insights(latest):
        st.markdown(f"- {insight}")
    for severity, message in get_smart_alerts(latest):
        if severity == "error":
            st.error(message)
        else:
            st.warning(message)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Advanced AI Insights</div>", unsafe_allow_html=True)
try:
    with st.expander("View Advanced AI Insights", expanded=True):
        st.write("DEBUG selected_crop:", st.session_state.selected_crop)
        crop_thresholds = get_crop_thresholds(st.session_state.selected_crop)
        st.info(
            f"Crop: {st.session_state.selected_crop} | "
            f"Soil {crop_thresholds['soil_moisture'][0]:.0f}-{crop_thresholds['soil_moisture'][1]:.0f}% | "
            f"Temp {crop_thresholds['temperature'][0]:.0f}-{crop_thresholds['temperature'][1]:.0f} C | "
            f"Humidity {crop_thresholds['humidity'][0]:.0f}-{crop_thresholds['humidity'][1]:.0f}%"
        )

        trend_msg = predict_moisture_trend(st.session_state.reading_history)
        if "critical soon" in trend_msg.lower():
            st.warning(trend_msg)
        else:
            st.success(trend_msg)

        explanation = generate_explanation(soil_percent, temperature_value, humidity_value)
        st.info(explanation)

        water_score = calculate_water_efficiency(soil_percent)
        st.metric("Water Optimization Score", f"{water_score}/100")
        st.progress(max(0.0, min(1.0, water_score / 100.0)))

        risk_level, risk_message = detect_disease_risk(temperature_value, humidity_value)
        if risk_level == "warning":
            st.warning(risk_message)
        else:
            st.success(risk_message)
except Exception as exc:
    st.info(f"Advanced AI insights unavailable right now: {exc}")

st.markdown("<div class='section-title'>AI Farm Advisor</div>", unsafe_allow_html=True)
card1, card2, card3 = st.columns(3)
with card1:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("### Crop Recommendation")
    st.markdown(f"Recommended crop: **{st.session_state.selected_crop}**")
    st.markdown(f"Confidence: **{crop_confidence}%**")
    st.markdown("</div>", unsafe_allow_html=True)

with card2:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("### Weather Impact")
    st.markdown(weather_alert_message)
    if weather_alert_level == "warning":
        st.warning("Weather impact alert")
    elif weather_alert_level == "success":
        st.success("Weather conditions favorable")
    else:
        st.info("Limited weather data")
    st.markdown("</div>", unsafe_allow_html=True)

with card3:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("### Irrigation Timing")
    st.markdown(irrigation_timing_msg)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Real-Time Environment Chart</div>", unsafe_allow_html=True)
st.plotly_chart(_build_realtime_environment_chart(history), use_container_width=True)

st.markdown("<div class='section-title'>Weather and Crop Disease Panel</div>", unsafe_allow_html=True)
w_col, d_col = st.columns(2)

with w_col:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("### Live Weather")
    if weather_data:
        icon = weather_data.get("icon", "")
        if icon:
            st.image(f"https://openweathermap.org/img/wn/{icon}@2x.png", width=72)
        st.metric("Weather Temperature", f"{weather_data['temp']:.1f} °C")
        st.metric("Weather Humidity", f"{weather_data['humidity']:.0f} %")
        st.metric("Rain Forecast", f"{weather_data['rain_mm']:.1f} mm")
        st.write(f"Condition: {weather_data['condition']}")
    else:
        st.info(weather_error or "Weather data unavailable.")
    st.markdown("</div>", unsafe_allow_html=True)

with d_col:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.markdown("### Crop Disease Prediction")
    for severity, message in predict_crop_risks(latest):
        if severity == "error":
            st.error(message)
        elif severity == "warning":
            st.warning(message)
        else:
            st.success(message)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Smart Weather Insights</div>", unsafe_allow_html=True)
try:
    with st.container():
        if inferred_weather is None:
            st.info("Waiting for sensor data...")
        else:
            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.write(f"Temperature: {inferred_weather['temp']:.1f} C")
            st.write(f"Humidity: {inferred_weather['humidity']:.1f} %")
            st.write(f"Condition: {inferred_weather['condition']}")
            st.write(f"Rain Probability: {inferred_weather['rain']}")

            rain_level = inferred_weather["rain"]
            if rain_level == "High":
                weather_sentence = "If current conditions persist, there is a High probability of rainfall, which may reduce irrigation needs."
            elif rain_level == "Medium":
                weather_sentence = "If current conditions persist, there is a Medium probability of rainfall, which may slightly reduce irrigation needs."
            else:
                weather_sentence = "If current conditions persist, there is a Low probability of rainfall, so irrigation planning should continue as normal."

            st.info(weather_sentence)
            st.markdown("</div>", unsafe_allow_html=True)
except Exception as exc:
    st.warning("Smart Weather Inference is temporarily unavailable.")
    st.text(str(exc))

st.markdown("<div class='section-title'>Farm Monitoring Map</div>", unsafe_allow_html=True)
st.plotly_chart(_build_zone_heatmap(soil_percent), use_container_width=True)
st.caption("Zone shading uses a monochrome green scale from dry to optimal moisture.")

st.markdown("<div class='section-title'>Farm Analytics and Yield Prediction</div>", unsafe_allow_html=True)
if not history.empty:
    today = pd.Timestamp.now().date()
    today_count = int((history.index.date == today).sum())
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Avg Soil Moisture (%)", f"{history['soil_moisture_percent'].mean():.1f}")
    a2.metric(
        "Highest Temp Today (°C)",
        f"{history[history.index.date == today]['temperature'].max():.1f}" if today_count else "--",
    )
    a3.metric(
        "Lowest Humidity Today (%)",
        f"{history[history.index.date == today]['humidity'].min():.1f}" if today_count else "--",
    )
    a4.metric("Readings Collected", f"{len(history)}")

    avg_soil = float(history["soil_moisture_percent"].mean())
    avg_temp = float(history["temperature"].mean())
    avg_hum = float(history["humidity"].mean())
    yield_label, yield_score = predict_yield(avg_soil, avg_temp, avg_hum)
    st.metric("Predicted Crop Yield", yield_label)
    st.progress(yield_score / 100.0)
else:
    st.info("No analytics available yet. Collect some readings first.")

if st.session_state.show_history:
    st.markdown("---")
    st.subheader("Interactive Charts")

    if not history.empty:
        recent = history.tail(24)

        combined = go.Figure()
        combined.add_trace(
            go.Scatter(
                x=recent.index,
                y=recent["soil_moisture_percent"],
                mode="lines+markers",
                name="Soil Moisture (%)",
                line=dict(color="#1976d2", width=3),
            )
        )
        combined.add_trace(
            go.Scatter(
                x=recent.index,
                y=recent["temperature"],
                mode="lines+markers",
                name="Temperature (°C)",
                line=dict(color="#f57c00", width=2),
            )
        )
        combined.add_trace(
            go.Scatter(
                x=recent.index,
                y=recent["humidity"],
                mode="lines+markers",
                name="Humidity (%)",
                line=dict(color="#7b1fa2", width=2),
            )
        )
        combined.update_layout(
            title="Combined Environmental Graph (Last 24 Readings)",
            template="plotly_white",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f6fbf6",
            margin=dict(l=20, r=20, t=45, b=20),
            hovermode="x unified",
            transition={"duration": 400},
        )
        combined.update_xaxes(gridcolor="#e6efe7")
        combined.update_yaxes(gridcolor="#e6efe7")
        st.plotly_chart(combined, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                _line_with_trend_and_forecast(recent, "soil_moisture_percent", "Soil Moisture Trend + Forecast"),
                use_container_width=True,
            )
            st.plotly_chart(
                _line_with_trend_and_forecast(recent, "humidity", "Humidity Trend + Forecast"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                _line_with_trend_and_forecast(recent, "temperature", "Temperature Trend + Forecast"),
                use_container_width=True,
            )
    else:
        st.info("No historical data yet. Start monitoring to build interactive charts.")

try:
    with st.expander("🧠 AI Decision Confidence (Explainable AI)", expanded=True):
        soil_val = locals().get("soil_moisture", 50)
        temp_val = locals().get("temperature", 25)
        humidity_val = locals().get("humidity", 60)

        if "soil_percent" in locals():
            soil_val = soil_percent
        if "temperature_value" in locals():
            temp_val = temperature_value
        if "humidity_value" in locals():
            humidity_val = humidity_value

        decision, confidence, trust, s_score, t_score, h_score = calculate_decision_confidence(
            soil_val, temp_val, humidity_val
        )

        st.subheader(f"Decision: {decision}")
        st.metric("Confidence", f"{confidence}%")
        st.write(f"Trust Level: {trust}")

        st.write("### Explanation")

        explanation_text = ""

        if s_score > 60:
            explanation_text += "Soil moisture is low, which strongly increases the need for irrigation. "
        elif s_score > 30:
            explanation_text += "Soil moisture is moderate and should be monitored. "
        else:
            explanation_text += "Soil moisture is at a healthy level. "

        if t_score > 50:
            explanation_text += "Temperature is high and may stress the crops. "
        elif t_score > 20:
            explanation_text += "Temperature is slightly affecting crop conditions. "
        else:
            explanation_text += "Temperature is within the optimal range. "

        if h_score > 50:
            explanation_text += "Humidity levels are high, increasing disease risk. "
        elif h_score > 20:
            explanation_text += "Humidity has a mild effect on crop conditions. "
        else:
            explanation_text += "Humidity is at a healthy level. "

        explanation_text += f"Overall, the system suggests: {decision} with a confidence of {confidence}%."

        st.info(explanation_text)

except Exception as e:
    st.warning("AI Confidence module error")
    st.text(str(e))

st.markdown("---")
auto_refresh_placeholder = st.empty()
with auto_refresh_placeholder.container():
    col_refresh_info, col_refresh_icon = st.columns([0.9, 0.1])
    with col_refresh_info:
        st.caption(f"🔄 Auto-refreshing every {st.session_state.refresh_rate} second(s)...")
    with col_refresh_icon:
        st.caption("⚡")

time.sleep(st.session_state.refresh_rate)
st.rerun()
