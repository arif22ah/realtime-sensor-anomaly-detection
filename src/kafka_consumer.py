import json
from src.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC
from src.models import StreamingAnomalyInferenceEngine
from src.db_logger import TimeSeriesAnomalyLogger

def run_kafka_consumer(max_messages: int = 500) -> None:
    """Consumes Kafka telemetry stream, runs sub-20ms PyTorch + IF inference, and logs to PostgreSQL."""
    from kafka import KafkaConsumer
    engine = StreamingAnomalyInferenceEngine()
    db_logger = TimeSeriesAnomalyLogger()

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="latest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    )

    for idx, msg in enumerate(consumer):
        scored = engine.score_packet(msg.value)
        db_logger.log_event(scored)
        if idx + 1 >= max_messages:
            break

if __name__ == "__main__":
    run_kafka_consumer()
