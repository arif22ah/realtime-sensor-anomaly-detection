from src.telemetry_simulator import simulate_single_event
from src.models import StreamingAnomalyInferenceEngine
from src.db_logger import TimeSeriesAnomalyLogger

def test_sub_20ms_inference_latency():
    engine = StreamingAnomalyInferenceEngine()
    packet = simulate_single_event(fault_mode="Normal Operation", severity_multiplier=0.0)
    scored = engine.score_packet(packet)
    assert scored["inference_latency_ms"] < 20.0
    assert scored["ae_is_anomaly"] == 0

def test_fault_injection_detection():
    engine = StreamingAnomalyInferenceEngine()
    fault_packet = simulate_single_event(
        fault_mode="Thermal Runaway & Cooling Failure",
        severity_multiplier=1.5
    )
    scored = engine.score_packet(fault_packet)
    assert scored["ae_is_anomaly"] == 1
    assert scored["ae_reconstruction_mse"] > scored["ae_threshold"]

def test_postgres_sqlite_timeseries_logger():
    engine = StreamingAnomalyInferenceEngine()
    logger = TimeSeriesAnomalyLogger()
    pkt = simulate_single_event(fault_mode="Harmonic Cavitation Surge", severity_multiplier=1.2)
    scored = engine.score_packet(pkt)
    logger.log_event(scored)
    df = logger.fetch_recent_logs(limit=5)
    assert not df.empty
    assert "ae_reconstruction_mse" in df.columns
