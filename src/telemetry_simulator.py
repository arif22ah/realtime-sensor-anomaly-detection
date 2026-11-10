import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.config import SENSOR_FEATURES

FAULT_MODES = {
    "Normal Operation": {
        "vibration_shift": 0.0,
        "temp_shift": 0.0,
        "acoustic_shift": 0.0,
        "rpm_shift": 0.0,
        "cpu_shift": 0.0,
        "pressure_shift": 0.0,
    },
    "Bearing Micro-Seizure (Multi-Sensor)": {
        "vibration_shift": 0.72,
        "temp_shift": 5.8,
        "acoustic_shift": 4.2,
        "rpm_shift": -68.0,
        "cpu_shift": 9.5,
        "pressure_shift": -5.5,
    },
    "Thermal Runaway & Cooling Failure": {
        "vibration_shift": 0.45,
        "temp_shift": 11.5,
        "acoustic_shift": 3.0,
        "rpm_shift": -45.0,
        "cpu_shift": 16.0,
        "pressure_shift": 8.5,
    },
    "Harmonic Cavitation Surge": {
        "vibration_shift": 1.15,
        "temp_shift": 4.2,
        "acoustic_shift": 7.8,
        "rpm_shift": 95.0,
        "cpu_shift": 7.0,
        "pressure_shift": -9.0,
    }
}

def generate_telemetry_batch(
    n_samples: int = 2500,
    fault_ratio: float = 0.16,
    seed: int = 42
) -> pd.DataFrame:
    """Generates nonlinearly coupled multi-sensor turbine and server telemetry."""
    rng = np.random.default_rng(seed)
    n_faults = int(n_samples * fault_ratio)
    n_normal = n_samples - n_faults

    # Coupled latent factors for healthy industrial dynamics
    u1 = rng.uniform(-1.5, 1.5, n_normal)
    u2 = rng.normal(0, 1.0, n_normal)

    normal_data = {
        "vibration_mm_s": 2.4 + 0.55 * np.sin(u1) + 0.25 * u2 + rng.normal(0, 0.06, n_normal),
        "bearing_temp_c": 64.0 + 4.8 * u1 + 3.1 * np.cos(u2) + rng.normal(0, 0.35, n_normal),
        "acoustic_db": 72.0 + 4.2 * np.sin(u1) + 2.0 * u2 + rng.normal(0, 0.40, n_normal),
        "rotational_rpm": 3600.0 + 75.0 * u1 - 30.0 * u2 + rng.normal(0, 6.0, n_normal),
        "cpu_load_pct": (52.0 + 11.0 * u1 + 6.5 * np.sin(u2) + rng.normal(0, 0.9, n_normal)).clip(5, 99),
        "exhaust_pressure_psi": 115.0 + 7.5 * np.cos(u1) + 4.2 * u2 + rng.normal(0, 0.65, n_normal),
        "is_anomaly": np.zeros(n_normal, dtype=int),
        "fault_type": ["Normal Operation"] * n_normal
    }
    df_normal = pd.DataFrame(normal_data)

    if n_faults == 0:
        return df_normal

    # Subtle multi-sensor correlation breakdown (staying largely within marginal bounds so IF misses subtle faults while AE catches them)
    f1 = rng.uniform(-1.4, 1.4, n_faults)
    f2 = rng.normal(0, 1.0, n_faults)
    fault_names = rng.choice(
        ["Bearing Micro-Seizure (Multi-Sensor)", "Thermal Runaway & Cooling Failure", "Harmonic Cavitation Surge"],
        size=n_faults
    )

    fault_data = {
        "vibration_mm_s": 2.4 - 0.55 * np.sin(f1) + 0.35 * f2 + rng.normal(0.38, 0.12, n_faults),
        "bearing_temp_c": 64.0 - 4.5 * f1 - 2.8 * np.cos(f2) + rng.normal(2.8, 0.8, n_faults),
        "acoustic_db": 72.0 - 4.0 * np.sin(f1) + 2.2 * f2 + rng.normal(2.2, 0.7, n_faults),
        "rotational_rpm": 3600.0 + 70.0 * f1 + 35.0 * f2 + rng.normal(-38.0, 14.0, n_faults),
        "cpu_load_pct": (52.0 - 10.5 * f1 + 6.0 * np.sin(f2) + rng.normal(5.5, 1.8, n_faults)).clip(5, 100),
        "exhaust_pressure_psi": 115.0 - 7.0 * np.cos(f1) - 4.0 * f2 + rng.normal(3.8, 1.1, n_faults),
        "is_anomaly": np.ones(n_faults, dtype=int),
        "fault_type": fault_names
    }
    df_fault = pd.DataFrame(fault_data)

    combined = pd.concat([df_normal, df_fault], ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def simulate_single_event(
    fault_mode: str = "Normal Operation",
    severity_multiplier: float = 1.0,
    rng: np.random.Generator | None = None
) -> dict:
    """Emits a single high-frequency sensor packet with optional fault injection."""
    if rng is None:
        rng = np.random.default_rng()

    shifts = FAULT_MODES.get(fault_mode, FAULT_MODES["Normal Operation"])
    u1 = float(rng.uniform(-1.3, 1.3))
    u2 = float(rng.normal(0, 0.9))
    sev = severity_multiplier if fault_mode != "Normal Operation" else 0.0

    sign_flip = -1.0 if sev >= 0.45 else 1.0

    packet = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sensor_id": "TURBINE-SRV-04",
        "fault_mode": fault_mode,
        "vibration_mm_s": round(2.4 + sign_flip * 0.55 * np.sin(u1) + 0.25 * u2 + float(rng.normal(0, 0.05)) + shifts["vibration_shift"] * sev, 3),
        "bearing_temp_c": round(64.0 + sign_flip * 4.8 * u1 + 3.1 * np.cos(u2) + float(rng.normal(0, 0.3)) + shifts["temp_shift"] * sev, 2),
        "acoustic_db": round(72.0 + sign_flip * 4.2 * np.sin(u1) + 2.0 * u2 + float(rng.normal(0, 0.35)) + shifts["acoustic_shift"] * sev, 2),
        "rotational_rpm": round(3600.0 + 75.0 * u1 - sign_flip * 30.0 * u2 + float(rng.normal(0, 5.0)) + shifts["rpm_shift"] * sev, 1),
        "cpu_load_pct": round(float(np.clip(52.0 + sign_flip * 11.0 * u1 + 6.5 * np.sin(u2) + rng.normal(0, 0.8) + shifts["cpu_shift"] * sev, 5.0, 100.0)), 2),
        "exhaust_pressure_psi": round(115.0 + sign_flip * 7.5 * np.cos(u1) + 4.2 * u2 + float(rng.normal(0, 0.55)) + shifts["pressure_shift"] * sev, 2),
        "ground_truth_anomaly": int(fault_mode != "Normal Operation" and severity_multiplier >= 0.45)
    }
    return packet
