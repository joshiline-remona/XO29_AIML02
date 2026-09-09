"""
Data models and Enums for Bayesian Change-Point Detection Results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class DetectorStatus(str, Enum):
    """Status enum for the change-point detector at time t."""
    WARMUP = "WARMUP"
    STABLE = "STABLE"
    ANOMALY_SUSPECTED = "ANOMALY_SUSPECTED"
    CHANGE_CONFIRMED = "CHANGE_CONFIRMED"


@dataclass(frozen=True)
class DetectionResult:
    """
    Result returned by Bayesian Change-Point Detector for observation at time t.
    
    Attributes:
        timestamp: Timestamp of the observation.
        observed_demand: Demand value in MW.
        change_point_prob: Posterior probability that a new regime/change started at time t P(r_t=0|x_{1:t}).
        map_run_length: Maximum A Posteriori (MAP) run length r_t (steps since last change point).
        change_detected: Boolean flag signaling a confirmed persistent structural change.
        detector_status: High-level status enum (WARMUP, STABLE, ANOMALY_SUSPECTED, CHANGE_CONFIRMED).
        posterior_mean: Estimated mean of the current regime.
        posterior_std: Estimated standard deviation of the current regime.
        sequence_idx: Monotonic sequence index t.
        metadata: Additional metadata dictionary.
    """
    timestamp: str
    observed_demand: float
    change_point_prob: float
    map_run_length: int
    change_detected: bool
    detector_status: DetectorStatus
    posterior_mean: float
    posterior_std: float
    sequence_idx: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "observed_demand": self.observed_demand,
            "change_point_prob": self.change_point_prob,
            "map_run_length": self.map_run_length,
            "change_detected": self.change_detected,
            "detector_status": self.detector_status.value,
            "posterior_mean": self.posterior_mean,
            "posterior_std": self.posterior_std,
            "sequence_idx": self.sequence_idx,
            "metadata": self.metadata,
        }
