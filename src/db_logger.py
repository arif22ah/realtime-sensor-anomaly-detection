import sqlite3
import pandas as pd
from src.config import POSTGRES_URI, BASE_DIR

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except Exception:
    PSYCOPG2_AVAILABLE = False

class TimeSeriesAnomalyLogger:
    """Persists streaming sensor telemetry & anomaly detections to PostgreSQL (with automatic built-in SQLite fallback)."""
    def __init__(self, db_uri: str = POSTGRES_URI):
        self.db_uri = db_uri
        self.sqlite_path = BASE_DIR / "data" / "anomaly_timeseries.db"
        self.use_postgres = db_uri.startswith("postgresql://") and PSYCOPG2_AVAILABLE
        if self.use_postgres:
            try:
                self._init_postgres_schema()
            except Exception:
                self.use_postgres = False
                self._init_sqlite_schema()
        else:
            self._init_sqlite_schema()

    def _init_postgres_schema(self) -> None:
        create_sql = """
        CREATE TABLE IF NOT EXISTS sensor_anomaly_logs (
            timestamp TEXT,
            sensor_id TEXT,
            fault_mode TEXT,
            vibration_mm_s DOUBLE PRECISION,
            bearing_temp_c DOUBLE PRECISION,
            acoustic_db DOUBLE PRECISION,
            rotational_rpm DOUBLE PRECISION,
            cpu_load_pct DOUBLE PRECISION,
            exhaust_pressure_psi DOUBLE PRECISION,
            ae_reconstruction_mse DOUBLE PRECISION,
            ae_threshold DOUBLE PRECISION,
            ae_is_anomaly INTEGER,
            if_anomaly_score DOUBLE PRECISION,
            if_is_anomaly INTEGER,
            root_cause_sensor TEXT,
            inference_latency_ms DOUBLE PRECISION
        );
        """
        with psycopg2.connect(self.db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(create_sql)
            conn.commit()

    def _init_sqlite_schema(self) -> None:
        create_sql = """
        CREATE TABLE IF NOT EXISTS sensor_anomaly_logs (
            timestamp TEXT,
            sensor_id TEXT,
            fault_mode TEXT,
            vibration_mm_s REAL,
            bearing_temp_c REAL,
            acoustic_db REAL,
            rotational_rpm REAL,
            cpu_load_pct REAL,
            exhaust_pressure_psi REAL,
            ae_reconstruction_mse REAL,
            ae_threshold REAL,
            ae_is_anomaly INTEGER,
            if_anomaly_score REAL,
            if_is_anomaly INTEGER,
            root_cause_sensor TEXT,
            inference_latency_ms REAL
        );
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(create_sql)
            conn.commit()

    def log_event(self, scored_packet: dict) -> None:
        cols = [
            "timestamp", "sensor_id", "fault_mode",
            "vibration_mm_s", "bearing_temp_c", "acoustic_db",
            "rotational_rpm", "cpu_load_pct", "exhaust_pressure_psi",
            "ae_reconstruction_mse", "ae_threshold", "ae_is_anomaly",
            "if_anomaly_score", "if_is_anomaly", "root_cause_sensor",
            "inference_latency_ms"
        ]
        vals = tuple(scored_packet[c] for c in cols)
        if self.use_postgres:
            placeholders = ", ".join(["%s"] * len(cols))
            sql = f"INSERT INTO sensor_anomaly_logs ({', '.join(cols)}) VALUES ({placeholders})"
            with psycopg2.connect(self.db_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, vals)
                conn.commit()
        else:
            placeholders = ", ".join(["?"] * len(cols))
            sql = f"INSERT INTO sensor_anomaly_logs ({', '.join(cols)}) VALUES ({placeholders})"
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(sql, vals)
                conn.commit()

    def fetch_recent_logs(self, limit: int = 120) -> pd.DataFrame:
        query = f"SELECT * FROM sensor_anomaly_logs ORDER BY timestamp DESC LIMIT {int(limit)}"
        if self.use_postgres:
            with psycopg2.connect(self.db_uri) as conn:
                return pd.read_sql_query(query, conn)
        else:
            with sqlite3.connect(self.sqlite_path) as conn:
                return pd.read_sql_query(query, conn)
