"""Insight generation logic for Smart Agriculture IoT Dashboard."""

from __future__ import annotations

from typing import Any


def soil_to_moisture_percent(soil_raw: int, max_raw: int = 1023) -> float:
    """Convert raw soil reading to moisture percentage in range 0-100.
    
    NOTE: Soil sensors are typically inverted - high raw values indicate DRY soil
    and low raw values indicate WET soil. This calculation reverses that.
    """
    if max_raw <= 0:
        return 0.0

    # Invert the raw value: dry soil (high reading) → low moisture %, wet soil (low reading) → high moisture %
    inverted_raw = max_raw - soil_raw
    ratio = float(inverted_raw) / float(max_raw)
    return max(0.0, min(100.0, ratio * 100.0))


def generate_insights(data: dict[str, Any]) -> list[str]:
    """Generate human-readable agronomy insights from sensor values."""
    insights: list[str] = []

    soil = int(data.get("soil", 0))
    temperature = float(data.get("temperature", 0.0))
    humidity = float(data.get("humidity", 0.0))

    if soil < 400:
        insights.append("⚠ Soil moisture critically low. Immediate irrigation required.")
    elif soil <= 600:
        insights.append("Soil moisture slightly low. Irrigation recommended.")
    else:
        insights.append("Soil moisture healthy.")

    if temperature > 35:
        insights.append("Temperature too high. Crop stress possible.")
    elif temperature < 15:
        insights.append("Temperature too low for optimal growth.")
    else:
        insights.append("Temperature within optimal range.")

    if humidity < 40:
        insights.append("Humidity low. Plants may lose water quickly.")
    elif humidity > 80:
        insights.append("Humidity high. Risk of fungal diseases.")
    else:
        insights.append("Humidity level healthy.")

    return insights


def get_irrigation_recommendation(data: dict[str, Any]) -> str:
    """Generate rule-based irrigation recommendation from soil moisture."""
    soil = int(data.get("soil", 0))
    moisture_percent = soil_to_moisture_percent(soil)

    if moisture_percent < 30:
        return "🚨 Soil is too dry. Irrigation required immediately."
    if moisture_percent <= 60:
        return "⚠ Soil moisture moderate. Monitor irrigation."
    return "✅ Soil moisture optimal."


def get_smart_alerts(data: dict[str, Any]) -> list[tuple[str, str]]:
    """Return alert tuples as (severity, message) for abnormal conditions."""
    alerts: list[tuple[str, str]] = []

    soil = int(data.get("soil", 0))
    temperature = float(data.get("temperature", 0.0))
    humidity = float(data.get("humidity", 0.0))
    moisture_percent = soil_to_moisture_percent(soil)

    if temperature > 40:
        alerts.append(("warning", "High temperature stress detected"))
    if humidity < 20:
        alerts.append(("warning", "Low humidity affecting crops"))
    if moisture_percent < 20:
        alerts.append(("error", "Critical soil moisture level detected"))

    return alerts


def predict_crop_risks(data: dict[str, Any]) -> list[tuple[str, str]]:
    """Predict crop stress/disease risks as (severity, message)."""
    risks: list[tuple[str, str]] = []

    soil = int(data.get("soil", 0))
    temperature = float(data.get("temperature", 0.0))
    humidity = float(data.get("humidity", 0.0))
    moisture_percent = soil_to_moisture_percent(soil)

    if humidity > 80 and 25 <= temperature <= 30:
        risks.append(("warning", "Fungal disease risk is elevated in current conditions."))
    if moisture_percent < 25:
        risks.append(("error", "Drought stress detected due to very low soil moisture."))
    if temperature > 40:
        risks.append(("warning", "Heat stress risk detected for sensitive crops."))

    if not risks:
        risks.append(("success", "No immediate crop disease/stress risks detected."))

    return risks


def irrigation_status(data: dict[str, Any]) -> str:
    """Return irrigation status label from soil moisture percentage."""
    moisture_percent = soil_to_moisture_percent(int(data.get("soil", 0)))
    if moisture_percent < 30:
        return "Water Needed"
    if moisture_percent <= 60:
        return "Optimal Moisture"
    return "Overwatered"


def weather_adjusted_irrigation_recommendation(
    data: dict[str, Any], weather: dict[str, Any] | None
) -> str:
    """Improve irrigation recommendation using weather signals."""
    base_message = get_irrigation_recommendation(data)
    if not weather:
        return f"{base_message} (Weather unavailable)"

    rain_mm = float(weather.get("rain_mm", 0.0))
    weather_main = str(weather.get("weather_main", "")).lower()

    if rain_mm > 0 or "rain" in weather_main:
        return f"{base_message} Rain expected/ongoing, reduce irrigation volume."

    if float(weather.get("temp", 0.0)) > 34:
        return f"{base_message} Hot weather forecast, schedule irrigation in early morning."

    return base_message
