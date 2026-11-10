# Real-Time Anomaly Detection on Streaming Sensor Data

**Category:** Deep Learning + Data Engineering • Apache Kafka  
**Author:** Omar Faruk Arif ([GitHub: arif22ah](https://github.com/arif22ah))  
**Repository:** `realtime-sensor-anomaly-detection`

---

## 1. Executive Summary & Problem Statement

### The Industrial Problem
Conventional batch diagnostics evaluate mechanical turbine and datacenter server telemetry hours after ingestion. By the time batch jobs flag an excursion, subtle multi-sensor degradation—such as bearing micro-seizures, harmonic cavitation surges, or cooling thermal runaway—has already escalated into catastrophic physical hardware failure.

### The Architectural Solution
This platform implements a **sub-20ms streaming anomaly detection pipeline** over high-frequency 6-channel sensor telemetry (`vibration_mm_s`, `bearing_temp_c`, `acoustic_db`, `rotational_rpm`, `cpu_load_pct`, `exhaust_pressure_psi`). Incoming events are ingested via **Apache Kafka**, evaluated concurrently by a **PyTorch Reconstruction Autoencoder** and **Scikit-learn's Isolation Forest**, and persisted into a **PostgreSQL** time-series anomaly log with an interactive **Streamlit Fault-Injection Sandbox**.

### Key Highlights & Verified Benchmarks
- **Superior Multi-Sensor Fault Precision & Recall:** The **PyTorch Reconstruction Autoencoder** captures nonlinear cross-sensor couplings, achieving **96.4%+ precision** and **94.8%+ recall** on subtle multi-sensor correlation breakdowns where univariate thresholds and tree-based partitioning struggle.
- **Sub-20ms Real-Time Inference Latency:** Achieves **18.0 ms** end-to-end streaming inference latency (and `< 1.0 ms` raw PyTorch tensor forward-pass compute on CPU), meeting strict industrial protection SLAs.
- **Persistent Time-Series Telemetry Sink:** Logs every scored packet, reconstruction MSE, model agreement flag, and root-cause culprit sensor channel to **PostgreSQL** (with automatic zero-config **SQLite** fallback for standalone local execution).
- **Interactive Operator Fault-Injection Sandbox:** Enables operators and SREs to inject synthetic mechanical/thermal faults at variable severity multipliers and stress-test alert MSE thresholds in real time.

---

## 2. System Architecture & Data Flow

```text
+-------------------------------------------------------------------------+
|                   High-Frequency Sensor Telemetry Stream                |
|  6 Channels: Vibration | Temp | Acoustic | RPM | CPU Load | Pressure    |
+------------------------------------+------------------------------------+
                                     |
                                     v  (JSON Packets @ 20Hz - 100Hz)
+-------------------------------------------------------------------------+
|                Apache Kafka Broker (Topic: turbine-server-telemetry)    |
+------------------------------------+------------------------------------+
                                     |
                                     v  (Low-Latency Consumer Group)
+-------------------------------------------------------------------------+
|               Dual Streaming Inference Engine (< 20ms SLA)              |
|                                                                         |
|  +---------------------------------+  +------------------------------+  |
|  | PyTorch Reconstruction AE       |  | Scikit-learn Isolation Forest|  |
|  | 6 -> 24 -> 12 -> 2 -> 12 -> 24  |  | 150 Trees, Contamination 6%  |  |
|  | Scores Nonlinear Coupling MSE   |  | Scores Marginal Partitioning |  |
|  +----------------+----------------+  +--------------+---------------+  |
|                   |                                  |                  |
|                   +----------------+-----------------+                  |
|                                    |                                    |
|                    Per-Feature Squared Error Ranking                    |
|                    (Root-Cause Sensor Attribution)                      |
+------------------------------------+------------------------------------+
                                     |
                   +-----------------+-----------------+
                   |                                   |
                   v                                   v
+------------------------------------+  +---------------------------------+
|   PostgreSQL Time-Series Database  |  |   Streamlit Operator Sandbox    |
|   Table: sensor_anomaly_logs       |  |   - Live MSE Threshold Tuning   |
|   (Auto-Fallback to Local SQLite)  |  |   - Multi-Fault Injection UI    |
+------------------------------------+  +---------------------------------+
```

---

## 3. Repository Structure

```text
realtime-sensor-anomaly-detection/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline running Pytest suite
├── data/
│   ├── reference_telemetry.csv     # 4,500 healthy baseline samples for calibration
│   ├── benchmark_evaluation.csv    # 2,500 labeled multi-sensor fault evaluation rows
│   └── anomaly_timeseries.db       # Local SQLite time-series sink (auto-created)
├── models/
│   ├── pytorch_autoencoder.pt      # Serialized PyTorch Autoencoder state_dict
│   ├── isolation_forest.pkl        # Serialized Scikit-learn Isolation Forest model
│   ├── standard_scaler.pkl         # Fitted z-score StandardScaler artifact
│   └── benchmark_metrics.json      # Precision, Recall, F1, ROC-AUC, and Latency metrics
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central configuration, environment vars, sensor schema
│   ├── telemetry_simulator.py      # Coupled nonlinear telemetry & fault-injection generator
│   ├── models.py                   # PyTorch Autoencoder + Isolation Forest training & scoring
│   ├── db_logger.py                # SQLAlchemy PostgreSQL / SQLite time-series persistence
│   ├── kafka_producer.py           # High-frequency Apache Kafka sensor stream producer
│   └── kafka_consumer.py           # Real-time Kafka consumer & inference orchestrator
├── tests/
│   └── test_anomaly_engine.py      # Automated tests for <20ms SLA & fault detection
├── app.py                          # Interactive Streamlit dashboard & fault sandbox
├── docker-compose.yml              # Multi-container Kafka, Zookeeper, Postgres, Streamlit
├── Dockerfile                      # Production container definition
├── requirements.txt                # Locked Python package dependencies
├── run_local.bat                   # One-click Windows batch launcher
└── README.md                       # Project documentation
```

---

## 4. Deep Learning vs. Tree-Based Anomaly Detection

Industrial turbines and high-density servers exhibit tight physical coupling across sensors (for example, rotational RPM, bearing temperature, and exhaust pressure co-vary along a smooth nonlinear manifold).

1. **Why Isolation Forest Struggles on Subtle Multi-Sensor Faults:**
   Isolation Forest isolates points via random axis-parallel splits. When a bearing micro-seizure causes *bearing temperature* to rise slightly while *rotational RPM* drops slightly, each individual sensor reading remains inside its normal univariate min/max range. Axis-parallel cuts fail to isolate these subtle correlation inversions early.
2. **Why the PyTorch Reconstruction Autoencoder Excels:**
   The Autoencoder compresses the 6-dimensional sensor vector through a 2-dimensional bottleneck (`6 -> 24 -> 12 -> 2 -> 12 -> 24 -> 6` with `GELU` activations). Because the bottleneck learns the healthy physical manifold, any breakdown in cross-sensor coupling produces a sharp spike in **Reconstruction Mean Squared Error (MSE)**—and the largest per-channel squared error immediately pinpoints the **root-cause sensor**.

| Model Architecture | Precision | Recall | F1-Score | ROC-AUC | Stream Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PyTorch Reconstruction Autoencoder** | **96.4%+** | **94.8%+** | **95.6%+** | **0.992+** | **18.0 ms** |
| **Scikit-learn Isolation Forest** | ~68.5% | ~36.0% | ~47.2% | ~0.815 | 19.4 ms |

---

## 5. Quickstart & Execution Guide

### Option A: One-Click Windows Local Execution (`run_local.bat`)
No external Kafka or PostgreSQL installation is required for local evaluation—the platform automatically initializes a local SQLite time-series store and interactive streaming simulator:
```cmd
run_local.bat
```

### Option B: Manual CLI Setup
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.models
pytest tests/ -v
streamlit run app.py
```

### Option C: Full Distributed Stack (`docker-compose`)
To spin up **Apache Kafka**, **Zookeeper**, **PostgreSQL 16**, and the **Streamlit** application in containers:
```cmd
docker-compose up --build
```
Then open `http://localhost:8501` in your browser. To stream live packets over Kafka from a separate terminal:
```cmd
python -m src.kafka_consumer
python -m src.kafka_producer
```
