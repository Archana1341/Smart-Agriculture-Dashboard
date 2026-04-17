"""Crop health score logic for Smart Agriculture IoT Dashboard."""

from __future__ import annotations

from typing import Any


def calculate_health_score(data: dict[str, Any]) -> int:
    """Calculate crop health score in range 0 to 100."""
    soil_moisture = float(data.get("soil_moisture", 0.0))
    temperature = float(data.get("temperature", 0.0))
    humidity = float(data.get("humidity", 0.0))

    # Soil moisture is the dominant signal (max 60) because dry soil quickly harms crops.
    if 60 <= soil_moisture <= 80:
        soil_score = 60
    elif 45 <= soil_moisture < 60:
        soil_score = 50
    elif 30 <= soil_moisture < 45:
        soil_score = 35
    elif 20 <= soil_moisture < 30:
        soil_score = 18
    else:
        soil_score = 8

    # Temperature contribution (max 20)
    if 20 <= temperature <= 32:
        temp_score = 20
    elif 15 <= temperature < 20 or 32 < temperature <= 36:
        temp_score = 14
    else:
        temp_score = 8

    # Humidity contribution (max 20)
    if 45 <= humidity <= 75:
        humidity_score = 20
    elif 30 <= humidity < 45 or 75 < humidity <= 85:
        humidity_score = 14
    else:
        humidity_score = 8

    score = soil_score + temp_score + humidity_score

    # Strong real-world dry-soil penalties.
    if soil_moisture < 30:
        score -= 25
    if soil_moisture < 20:
        # Enforce a severe drop for critically dry soil.
        score = min(score, 35)

    return int(max(0, min(100, round(score))))


def get_health_status(score: int) -> tuple[str, str]:
    """Map numeric health score to (label, color)."""
    if score >= 75:
        return "Healthy", "#2e7d32"
    if score >= 45:
        return "Warning", "#ef6c00"
    return "Critical", "#c62828"


def predict_yield(avg_soil_moisture: float, avg_temperature: float, avg_humidity: float) -> tuple[str, int]:
    """Predict yield category from average environment conditions."""
    points = 0

    if avg_soil_moisture >= 60:
        points += 40
    elif avg_soil_moisture >= 40:
        points += 25
    else:
        points += 10

    if 18 <= avg_temperature <= 32:
        points += 35
    elif 15 <= avg_temperature <= 36:
        points += 22
    else:
        points += 10

    if 45 <= avg_humidity <= 75:
        points += 25
    elif 35 <= avg_humidity <= 85:
        points += 15
    else:
        points += 8

    score = max(0, min(100, points))
    if score >= 75:
        return "High Yield", score
    if score >= 45:
        return "Medium Yield", score
    return "Low Yield", score
