import paho.mqtt.client as mqtt
import json
import time
import random
import threading
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(message)s"
)
logger = logging.getLogger(__name__)

BROKER = "localhost"
PORT   = 1883

# Each sensor has its own publish interval — this is realistic.
# A temperature sensor might update every 2s, pH every 10s.
SENSORS = [
    {
        "sensor_id": "TMP_BATIDORA_01",
        "location":  "batidora_1",
        "variable":  "temperature",
        "unit":      "celsius",
        "base":      27.0,
        "noise":     0.5,
        "interval":  0.3,
    },
    {
        "sensor_id": "TMP_BATIDORA_02",
        "location":  "batidora_2",
        "variable":  "temperature",
        "unit":      "celsius",
        "base":      28.5,
        "noise":     0.4,
        "interval":  0.5,
    },
    {
        "sensor_id": "HUM_PASTA_01",
        "location":  "molino_entrada",
        "variable":  "humidity",
        "unit":      "percent",
        "base":      35.0,
        "noise":     1.0,
        "interval":  0.4,
    },
    {
        "sensor_id": "PH_DECANTER_01",
        "location":  "decanter_entrada",
        "variable":  "ph",
        "unit":      "pH",
        "base":      5.2,
        "noise":     0.05,
        "interval":  0.01,
    },
]


def sensor_thread(client: mqtt.Client, sensor: dict):
    """Each sensor runs in its own thread, publishing at its own interval."""
    topic = f"almazara/sensor/{sensor['variable']}/{sensor['sensor_id']}"
    logger.info(f"Starting — publishing every {sensor['interval']}s on {topic}")

    while True:
        value = round(sensor["base"] + random.gauss(0, sensor["noise"]), 4)
        payload = json.dumps({
            "sensor_id": sensor["sensor_id"],
            "location":  sensor["location"],
            "variable":  sensor["variable"],
            "value":     value,
            "unit":      sensor["unit"],
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        })
        client.publish(topic, payload, qos=1)
        logger.info(f"{sensor['sensor_id']} → {sensor['variable']}: {value} {sensor['unit']}")
        time.sleep(sensor["interval"])


# One shared MQTT client — thread-safe with loop_start()
client = mqtt.Client(client_id="multi-sensor-simulator")
client.connect(BROKER, PORT)
client.loop_start()

# Spawn one thread per sensor
threads = []
for sensor in SENSORS:
    t = threading.Thread(
        target=sensor_thread,
        args=(client, sensor),
        name=sensor["sensor_id"],
        daemon=True
    )
    threads.append(t)
    t.start()

logger.info(f"{len(SENSORS)} sensor threads running. Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logger.info("Stopping simulator.")
finally:
    client.loop_stop()
    client.disconnect()