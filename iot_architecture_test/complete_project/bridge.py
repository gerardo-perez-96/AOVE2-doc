import paho.mqtt.client as mqtt
import psycopg2
import json
import time
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BRIDGE] %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# --- Config ---
MQTT_BROKER = "localhost"
MQTT_PORT   = 1883
TOPIC       = "almazara/sensor/#"

DB_HOST     = "localhost"
DB_PORT     = 5432
DB_NAME     = "sensors"
DB_USER     = "postgres"
DB_PASSWORD = "password"

# --- Database ---
def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER,
        password=DB_PASSWORD
    )

conn = get_connection()
conn.autocommit = True
logger.info("Connected to TimescaleDB")

INSERT_SQL = """
    INSERT INTO sensor_readings (time, sensor_id, location, variable, value, unit)
    VALUES (%s, %s, %s, %s, %s, %s)
"""

def insert_reading(payload: dict):
    timestamp = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
    with conn.cursor() as cur:
        cur.execute(INSERT_SQL, (
            timestamp,
            payload["sensor_id"],
            payload.get("location", "unknown"),
            payload["variable"],
            payload["value"],
            payload.get("unit")
        ))

# --- MQTT callbacks ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(TOPIC, qos=1)
        logger.info(f"Subscribed to {TOPIC}")
    else:
        logger.error(f"Connection failed, rc={rc}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        insert_reading(payload)
        logger.info(
            f"{payload['sensor_id']} | "
            f"{payload['variable']} = {payload['value']} {payload.get('unit','')}"
        )
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON: {msg.payload}")
    except KeyError as e:
        logger.error(f"Missing field {e} in: {payload}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

# --- Main ---
client = mqtt.Client(client_id="timescale-bridge")
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT)

logger.info("Bridge running — waiting for messages")
client.loop_forever()