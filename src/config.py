import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

for directory in [DATA_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Streaming & Storage Endpoints
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "turbine-server-telemetry")

POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    f"sqlite:///{BASE_DIR / 'data' / 'anomaly_timeseries.db'}"
)

SENSOR_FEATURES = [
    "vibration_mm_s",
    "bearing_temp_c",
    "acoustic_db",
    "rotational_rpm",
    "cpu_load_pct",
    "exhaust_pressure_psi"
]

DEFAULT_AE_THRESHOLD = 0.42
TARGET_LATENCY_MS = 20.0
