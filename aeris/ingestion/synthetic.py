"""
Synthetic Data Stream Generator for AERIS Module 1 Testing.

Generates reproducible streaming time series for the 4 hackathon validation scenarios:
1. Stable demand with normal Gaussian noise.
2. Single isolated extreme spike (transient anomaly).
3. Persistent baseline step change (structural regime shift).
4. Gradual demand drift/ramp.
"""

from datetime import datetime, timedelta, timezone
from typing import List
import numpy as np
from aeris.ingestion.schemas import DemandObservation


class SyntheticStreamGenerator:
    """Utility class to construct synthetic demand observation streams."""

    @staticmethod
    def _create_timestamps(n_steps: int, start_iso: str = "2026-01-01T00:00:00Z") -> List[str]:
        base_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        return [(base_dt + timedelta(minutes=15 * i)).isoformat() for i in range(n_steps)]

    @classmethod
    def generate_scenario_1_stable(
        cls,
        n_steps: int = 100,
        baseline_mw: float = 500.0,
        noise_std: float = 10.0,
        seed: int = 42
    ) -> List[DemandObservation]:
        """TEST 1: Stable electricity demand with normal random noise."""
        rng = np.random.default_rng(seed)
        timestamps = cls._create_timestamps(n_steps)
        demands = rng.normal(loc=baseline_mw, scale=noise_std, size=n_steps)
        
        return [
            DemandObservation(
                timestamp=ts,
                demand_mw=float(d),
                sequence_idx=i,
                metadata={"scenario": "stable_noise"}
            )
            for i, (ts, d) in enumerate(zip(timestamps, demands))
        ]

    @classmethod
    def generate_scenario_2_isolated_spike(
        cls,
        n_steps: int = 100,
        baseline_mw: float = 500.0,
        noise_std: float = 10.0,
        spike_idx: int = 45,
        spike_magnitude_mw: float = 900.0,
        seed: int = 42
    ) -> List[DemandObservation]:
        """TEST 2: Stable demand with one isolated extreme spike at spike_idx."""
        rng = np.random.default_rng(seed)
        timestamps = cls._create_timestamps(n_steps)
        demands = rng.normal(loc=baseline_mw, scale=noise_std, size=n_steps)
        demands[spike_idx] = spike_magnitude_mw  # Single isolated spike
        
        return [
            DemandObservation(
                timestamp=ts,
                demand_mw=float(d),
                sequence_idx=i,
                metadata={"scenario": "isolated_spike", "is_spike": i == spike_idx}
            )
            for i, (ts, d) in enumerate(zip(timestamps, demands))
        ]

    @classmethod
    def generate_scenario_3_persistent_shift(
        cls,
        n_steps: int = 120,
        baseline_1_mw: float = 500.0,
        baseline_2_mw: float = 600.0,
        change_idx: int = 50,
        noise_std: float = 10.0,
        seed: int = 42
    ) -> List[DemandObservation]:
        """TEST 3: Persistent baseline step shift (500 MW -> 600 MW)."""
        rng = np.random.default_rng(seed)
        timestamps = cls._create_timestamps(n_steps)
        
        demands = np.zeros(n_steps)
        demands[:change_idx] = rng.normal(loc=baseline_1_mw, scale=noise_std, size=change_idx)
        demands[change_idx:] = rng.normal(loc=baseline_2_mw, scale=noise_std, size=n_steps - change_idx)
        
        return [
            DemandObservation(
                timestamp=ts,
                demand_mw=float(d),
                sequence_idx=i,
                metadata={"scenario": "persistent_shift", "true_change_idx": change_idx}
            )
            for i, (ts, d) in enumerate(zip(timestamps, demands))
        ]

    @classmethod
    def generate_scenario_4_gradual_change(
        cls,
        n_steps: int = 120,
        start_mw: float = 500.0,
        end_mw: float = 600.0,
        ramp_start_idx: int = 40,
        ramp_end_idx: int = 80,
        noise_std: float = 10.0,
        seed: int = 42
    ) -> List[DemandObservation]:
        """TEST 4: Gradual demand ramp from 500 MW to 600 MW across 40 steps."""
        rng = np.random.default_rng(seed)
        timestamps = cls._create_timestamps(n_steps)
        
        base_signal = np.full(n_steps, start_mw, dtype=float)
        ramp_length = ramp_end_idx - ramp_start_idx
        base_signal[ramp_start_idx:ramp_end_idx] = np.linspace(start_mw, end_mw, ramp_length)
        base_signal[ramp_end_idx:] = end_mw
        
        demands = base_signal + rng.normal(loc=0.0, scale=noise_std, size=n_steps)
        
        return [
            DemandObservation(
                timestamp=ts,
                demand_mw=float(d),
                sequence_idx=i,
                metadata={"scenario": "gradual_change", "ramp_start": ramp_start_idx, "ramp_end": ramp_end_idx}
            )
            for i, (ts, d) in enumerate(zip(timestamps, demands))
        ]
