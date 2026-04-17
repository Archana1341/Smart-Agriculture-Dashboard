"""Serial reader module for Smart Agriculture IoT Dashboard.

Reads live sensor messages from Arduino over USB serial.

Example Arduino line:
soil:999,temp:28.70,humidity:60.20,uv:1.00
"""

from __future__ import annotations
from typing import Dict, Optional
import time
import serial
from serial import SerialException


def _parse_sensor_line(line: str) -> Optional[Dict]:
    """Parse one serial line safely.

    Expected format:
    soil:999,temp:28.70,humidity:60.20,uv:1.00
    """

    if not line:
        return None

    parts = line.split(",")
    kv_map = {}

    for part in parts:
        if ":" not in part:
            continue

        key, value = part.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

        kv_map[key] = value

    try:
        soil_raw = kv_map.get("soil")
        temp_raw = kv_map.get("temp", kv_map.get("temperature"))
        humidity_raw = kv_map.get("humidity")

        if soil_raw is None or temp_raw is None or humidity_raw is None:
            return None

        soil = int(float(soil_raw))
        temperature = float(temp_raw)
        humidity = float(humidity_raw)
    except (TypeError, ValueError):
        return None

    return {
        "soil": soil,
        "temperature": temperature,
        "humidity": humidity,
    }


def read_sensor_data(port: str):
    """Continuously stream parsed sensor packets from one COM port."""
    while True:
        try:
            with serial.Serial(port=port, baudrate=9600, timeout=2) as ser:
                # Arduino can reset after serial open; wait for stable lines.
                time.sleep(2)
                ser.reset_input_buffer()

                while True:
                    raw_line = ser.readline()
                    if not raw_line:
                        continue

                    decoded_line = raw_line.decode("utf-8", errors="ignore").strip()
                    print(f"RAW: {decoded_line}")

                    parsed = _parse_sensor_line(decoded_line)
                    if parsed is None:
                        print(f"Skipping malformed line: {decoded_line}")
                        continue

                    yield parsed

        except SerialException as exc:
            error_text = str(exc)
            if "Access is denied" in error_text:
                print(
                    f"Serial connection error on {port}: {exc} | "
                    "Close Arduino IDE Serial Monitor, then wait 2-3 seconds."
                )
            else:
                print(f"Serial connection error on {port}: {exc}")
            time.sleep(2)
        except Exception as exc:  # pragma: no cover
            print(f"Unexpected serial reader error on {port}: {exc}")
            time.sleep(1)


if __name__ == "__main__":

    port = input("Enter Arduino COM port (Example COM10): ")

    print(f"Listening to {port}... Press CTRL+C to stop.")

    try:
        for data in read_sensor_data(port):
            print(data)
    except KeyboardInterrupt:
        print("Stopped.")