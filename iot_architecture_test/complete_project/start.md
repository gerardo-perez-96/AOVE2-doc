# MDI Lab — Local Architecture Setup
## Windows 11 · TimescaleDB · MQTT · Python Bridge

---

## What this architecture is

A local replica of the data ingestion layer of the MDI project.
Three components working in sequence:

```
[Python Simulator] → publishes → [MQTT Broker] → consumed by → [Python Bridge] → inserts → [TimescaleDB]
                                                                                                    ↓
                                                                                            [Grafana]
```

| Component | Technology | Role |
|---|---|---|
| Simulator | Python + paho-mqtt | Generates fake sensor data and publishes it |
| Broker | Mosquitto (Docker) | Receives and routes messages by topic |
| Bridge | Python + psycopg2 | Subscribes to broker, writes rows to database |
| Database | TimescaleDB (Docker) | Stores timestamped sensor readings in a hypertable |

**Why this matters for the project:**
In production, the simulator is replaced by real sensors (temperature, humidity, pH) communicating over MQTT or OPC-UA. The broker, bridge, and database remain the same. Building this locally means you understand every layer before the real hardware arrives.

---

## Prerequisites

### 1. Virtualization check
Open Task Manager → Performance → CPU → confirm **Virtualization: Enabled**.
If disabled, enable it in your BIOS before proceeding.

### 2. WSL2
Open PowerShell as Administrator:
```powershell
wsl --install
```
Restart when prompted. After restart, complete the Ubuntu setup (create username and password).
Verify:
```powershell
wsl --status
```
Expected output: `Default Version: 2`

### 3. Docker Desktop
Download: https://www.docker.com/products/docker-desktop/

During install:
- ✅ Use WSL2 backend (default)
- ✅ Add to PATH

After install, open Docker Desktop and wait for the whale icon in the system tray to go solid.
Verify in PowerShell:
```powershell
docker --version
docker compose version
```

### 4. Python 3.11+
Download: https://www.python.org/downloads/

During install:
- ✅ Add Python to PATH

Verify:
```powershell
python --version
```

### 5. VS Code
Download: https://code.visualstudio.com/

During install:
- ✅ Add to PATH
- ✅ Register as editor for supported file types

### 6. Python libraries
```powershell
pip install paho-mqtt psycopg2-binary
```

---

## Project structure

Create this folder structure manually or follow the setup steps below:

```
mdi-lab/
├── docker-compose.yml
├── mosquitto/
│   └── config/
│       └── mosquitto.conf
├── timescaledb/
│   └── init.sql
├── simulator/
│   └── publish.py
└── bridge/
    └── bridge.py
```

Create the root folder and open it in VS Code:
```powershell
mkdir mdi-lab
cd mdi-lab
code .
```

---

## Component 1 — TimescaleDB

### What it is
PostgreSQL with the TimescaleDB extension. Sensor readings are stored in a **hypertable** — a table partitioned automatically by time into chunks. A query for "last hour" only touches the chunk covering that period, not all data ever written.

Why TimescaleDB over generic NoSQL:
- Standard SQL — you can JOIN sensor data against ERP relational tables in one query
- `time_bucket()` — a native function for time-series aggregation used in Grafana panels
- Hypertable chunks make time-range queries fast regardless of total data volume

### Configuration

**`docker-compose.yml`** — add the timescaledb service:
```yaml
version: '3.8'

services:

  timescaledb:
    image: timescale/timescaledb:latest-pg15
    container_name: timescaledb
    environment:
      POSTGRES_DB: sensors
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - timescale-data:/var/lib/postgresql/data
      - ./timescaledb/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d sensors"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  timescale-data:
```

**`timescaledb/init.sql`** — schema created automatically on first start:
```sql
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS sensor_readings (
    time        TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    sensor_id   TEXT             NOT NULL,
    location    TEXT             NOT NULL,
    variable    TEXT             NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    unit        TEXT
);

-- Converts the table into a hypertable partitioned by time
SELECT create_hypertable('sensor_readings', 'time');

-- Index for querying one specific sensor's history
CREATE INDEX ON sensor_readings (sensor_id, time DESC);

-- Index for querying all sensors of one variable type (e.g. all temperature sensors)
CREATE INDEX ON sensor_readings (variable, time DESC);
```

### Key decisions explained

`TIMESTAMPTZ` — stores timestamps with timezone in UTC. Always use this for sensor data.
Plain `TIMESTAMP` has no timezone information and causes silent errors with international data.

Named volume `timescale-data` — persists data across container restarts.
`docker compose down` keeps the volume. `docker compose down -v` deletes it.

`init.sql` mounted at `/docker-entrypoint-initdb.d/` — PostgreSQL runs every `.sql` file
in that directory automatically on first startup, only if the data directory is empty.

---

## Component 2 — MQTT Broker (Mosquitto)

### What it is
A message broker that receives published messages and routes them to subscribers by topic.
The broker **decouples** producers from consumers — the sensor doesn't know who is listening,
and the bridge doesn't know where the data comes from.

Why this matters for the project:
- Adding a new consumer (ML model, alert system, dashboard) requires zero changes to the sensors
- If the bridge crashes, the sensor keeps publishing without errors
- Multiple consumers can receive the same data simultaneously

### Topic structure
Topics are hierarchical paths: `almazara/sensor/temperature/BATIDORA_01`

| Subscription | Receives |
|---|---|
| `almazara/sensor/#` | Every sensor, every variable |
| `almazara/sensor/temperature/#` | All temperature sensors only |
| `almazara/sensor/temperature/BATIDORA_01` | One specific sensor only |

`#` is a wildcard matching everything below that level.

### QoS levels

| Level | Name | Behaviour | Use when |
|---|---|---|---|
| 0 | Fire and forget | No confirmation | High frequency, losing one reading is acceptable |
| 1 | At least once | Broker confirms delivery, duplicates possible | Sensor readings every few seconds |
| 2 | Exactly once | Guaranteed, no duplicates, slowest | Commands, critical events |

For sensor readings at 3-second intervals, **QoS 1** is the right choice.
The ACK overhead on a 3-second interval is negligible, so you get delivery guarantees for free.

### Configuration

Add to `docker-compose.yml` under `services:`:
```yaml
  mosquitto:
    image: eclipse-mosquitto:2
    container_name: mosquitto
    ports:
      - "1883:1883"
    volumes:
      - ./mosquitto/config:/mosquitto/config
```

**`mosquitto/config/mosquitto.conf`:**
```
listener 1883
allow_anonymous true
persistence false
log_type all
```

`allow_anonymous true` — no authentication required. Local development only.
`persistence false` — messages are not saved to disk. Undelivered messages lost on broker restart.

---

## Component 3 — Python Simulator

### What it is
A Python script that generates fake sensor readings and publishes them to the broker.
In production this is replaced by real hardware. Until then, it lets you test the full pipeline.

**`simulator/publish.py`:**
```python
import paho.mqtt.client as mqtt
import json
import time
import random
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SIMULATOR] %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

BROKER           = "localhost"
PORT             = 1883
PUBLISH_INTERVAL = 3  # seconds

SENSORS = [
    {
        "sensor_id": "TMP_BATIDORA_01",
        "location":  "batidora_1",
        "variable":  "temperature",
        "unit":      "celsius",
        "base":      27.0,
        "noise":     0.5
    },
    {
        "sensor_id": "HUM_PASTA_01",
        "location":  "molino_entrada",
        "variable":  "humidity",
        "unit":      "percent",
        "base":      35.0,
        "noise":     1.0
    },
    {
        "sensor_id": "PH_DECANTER_01",
        "location":  "decanter_entrada",
        "variable":  "ph",
        "unit":      "pH",
        "base":      5.2,
        "noise":     0.05
    },
]

client = mqtt.Client(client_id="olive-mill-simulator")
client.connect(BROKER, PORT)
client.loop_start()

logger.info(f"Simulator running — publishing {len(SENSORS)} sensors every {PUBLISH_INTERVAL}s")

try:
    while True:
        for sensor in SENSORS:
            value = round(sensor["base"] + random.gauss(0, sensor["noise"]), 4)
            payload = json.dumps({
                "sensor_id": sensor["sensor_id"],
                "location":  sensor["location"],
                "variable":  sensor["variable"],
                "value":     value,
                "unit":      sensor["unit"],
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            })
            topic = f"almazara/sensor/{sensor['variable']}/{sensor['sensor_id']}"
            client.publish(topic, payload, qos=1)
            logger.info(f"{sensor['sensor_id']} → {sensor['variable']}: {value}")
        time.sleep(PUBLISH_INTERVAL)

except KeyboardInterrupt:
    logger.info("Simulator stopped.")
finally:
    client.loop_stop()
    client.disconnect()
```

### Key decisions explained

`client.loop_start()` — starts a background thread managing the network connection.
Without it, published messages queue in memory and are never actually sent until you
call `loop()` manually. Always pair it with `loop_stop()` on exit.

`json.dumps()` — MQTT messages are raw bytes. The broker doesn't validate or parse them.
Both publisher and subscriber must agree on the format. JSON is the standard choice
because it's human-readable, self-describing, and parseable in any language.

`TIMESTAMPTZ` in the payload — always attach a timestamp at the sensor side, not the bridge side.
If the bridge is delayed or batching, a bridge-side timestamp would be wrong.

---

## Component 4 — Python Bridge

### What it is
Subscribes to all sensor topics on the broker and writes each message as a row in TimescaleDB.
This is the only component that knows about both the broker and the database.

**`bridge/bridge.py`:**
```python
import paho.mqtt.client as mqtt
import psycopg2
import json
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BRIDGE] %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

BROKER      = "localhost"
PORT        = 1883
TOPIC       = "almazara/sensor/#"

DB_HOST     = "localhost"
DB_PORT     = 5432
DB_NAME     = "sensors"
DB_USER     = "postgres"
DB_PASSWORD = "password"

INSERT_SQL = """
    INSERT INTO sensor_readings (time, sensor_id, location, variable, value, unit)
    VALUES (%s, %s, %s, %s, %s, %s)
"""

conn = psycopg2.connect(
    host=DB_HOST, port=DB_PORT,
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
)
conn.autocommit = True
logger.info("Connected to TimescaleDB")


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
            f"{payload['variable']} = {payload['value']} {payload.get('unit', '')}"
        )
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON on topic {msg.topic}: {msg.payload}")
    except KeyError as e:
        logger.error(f"Missing required field {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


client = mqtt.Client(client_id="timescale-bridge")
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT)

logger.info("Bridge running — waiting for messages")
client.loop_forever()
```

### Key decisions explained

`on_connect` subscribes inside the callback, not before `connect()`.
Subscribing before the connection is confirmed causes a silent failure.
The callback only fires when the connection is live.

`conn.autocommit = True` — without this, every INSERT is inside an uncommitted transaction.
If the script crashes, the transaction rolls back and you lose all pending data.
Autocommit writes each row immediately and permanently.

Three separate `except` blocks — each catches a different failure class.
A bad JSON message from one misbehaving sensor must not crash the bridge
and stop ingestion from every other sensor.

`loop_forever()` — blocks the main thread and processes messages indefinitely.
Unlike `loop_start()`, this doesn't need a separate `while True` — it is the loop.
Use `loop_forever()` when the bridge has nothing else to do.

---

## Running the full stack

### Start the containers

```powershell
cd mdi-lab
docker compose up -d
```

Wait until both containers are healthy:
```powershell
docker ps
```

Expected output: both `timescaledb` and `mosquitto` showing `Up` or `healthy`.

### Start the bridge (Terminal 1)

```powershell
cd mdi-lab
python bridge/bridge.py
```

Expected output:
```
Connected to TimescaleDB
Subscribed to almazara/sensor/#
Bridge running — waiting for messages
```

### Start the simulator (Terminal 2)

```powershell
cd mdi-lab
python simulator/publish.py
```

Expected output:
```
Simulator running — publishing 3 sensors every 3s
TMP_BATIDORA_01 → temperature: 27.3
HUM_PASTA_01 → humidity: 34.8
PH_DECANTER_01 → ph: 5.19
```

The bridge terminal should start printing each received message simultaneously.

### Verify data in the database (Terminal 3)

```powershell
docker exec -it timescaledb psql -U postgres -d sensors
```

```sql
-- Latest 10 readings
SELECT time, sensor_id, variable, value, unit
FROM sensor_readings
ORDER BY time DESC
LIMIT 10;

-- Count per sensor
SELECT sensor_id, COUNT(*) AS total_readings
FROM sensor_readings
GROUP BY sensor_id
ORDER BY sensor_id;

-- Average per sensor per minute
SELECT
    time_bucket('1 minute', time) AS minute,
    sensor_id,
    ROUND(AVG(value)::numeric, 4) AS avg_value
FROM sensor_readings
WHERE time > NOW() - INTERVAL '10 minutes'
GROUP BY minute, sensor_id
ORDER BY minute DESC, sensor_id;
```

---

## Stopping everything

```powershell
# Stop the bridge and simulator: Ctrl+C in their respective terminals

# Stop containers, keep database data
docker compose down

# Stop containers AND delete database (all rows gone)
docker compose down -v
```

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `Connection refused` on port 5432 | TimescaleDB not running | `docker compose up -d` |
| `Connection refused` on port 1883 | Mosquitto not running | `docker compose up -d` |
| Port 5432 already in use | Local PostgreSQL installed | Change host port to 5433 in docker-compose.yml |
| `sensors-#` prompt in psql | Inside a block comment | Type `*/;` and press Enter |
| `sensors->` prompt in psql | Missing semicolon | Type `;` and press Enter |
| Bridge logs `Missing required field` | Simulator payload missing a field | Check simulator payload matches INSERT_SQL fields |