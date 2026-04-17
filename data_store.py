"""Data persistence utilities for Smart Agriculture IoT Dashboard."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_LOG_PATH = Path(__file__).resolve().parent / "data_log.csv"
DEFAULT_COLUMNS = ["timestamp", "soil", "temperature", "humidity"]


def append_sensor_data(data: dict) -> None:
    """Append one sensor reading to CSV log, creating it when needed."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "soil": int(data["soil"]),
        "temperature": float(data["temperature"]),
        "humidity": float(data["humidity"]),
    }

    frame = pd.DataFrame([row], columns=DEFAULT_COLUMNS)
    write_header = not DATA_LOG_PATH.exists() or DATA_LOG_PATH.stat().st_size == 0
    frame.to_csv(DATA_LOG_PATH, mode="a", header=write_header, index=False)


def load_data() -> pd.DataFrame:
    """Load historical sensor data from CSV for trend visualization."""
    if not DATA_LOG_PATH.exists() or DATA_LOG_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    data = pd.read_csv(DATA_LOG_PATH)
    for column in DEFAULT_COLUMNS:
        if column not in data.columns:
            data[column] = pd.Series(dtype="float64")

    return data[DEFAULT_COLUMNS]


def reset_data_log() -> None:
    """Reset historical data file and keep the CSV header."""
    pd.DataFrame(columns=DEFAULT_COLUMNS).to_csv(DATA_LOG_PATH, index=False)


def get_data_log_bytes() -> bytes:
    """Return raw CSV bytes for download button."""
    if not DATA_LOG_PATH.exists():
        pd.DataFrame(columns=DEFAULT_COLUMNS).to_csv(DATA_LOG_PATH, index=False)
    return DATA_LOG_PATH.read_bytes()


def load_recent_data(limit: int = 240) -> pd.DataFrame:
    """Load recent rows for responsive live charts."""
    data = load_data()
    if data.empty:
        return data
    return data.tail(max(1, int(limit))).reset_index(drop=True)
