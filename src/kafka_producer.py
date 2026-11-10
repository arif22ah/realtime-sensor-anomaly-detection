import json
import time
from src.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC
from src.telemetry_simulator import simulate_single_event

def run_kafka_producer(interval_sec: float = 0.05, max_events: int = 500) -> None:
    """Streams high-frequency turbine & server telemetry packets to Apache Kafka."""
    from kafka import KafkaProducer
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    for idx in range(max_events):
        fault = "Bearing Micro-Seizure (Multi-Sensor)" if idx % 12 == 0 else "Normal Operation"
        packet = simulate_single_event(fault_mode=fault, severity_multiplier=1.0)
        producer.send(KAFKA_TOPIC, value=packet)
        time.sleep(interval_sec)
    producer.flush()

if __name__ == "__main__":
    run_kafka_producer()
