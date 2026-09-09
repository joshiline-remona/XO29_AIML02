"""
Data models and Enums for AERIS Module 3 Dynamic Forecast Model Routing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aeris.config import RoutingConfig
from aeris.regime.models import RegimeType


@dataclass(frozen=True)
class RoutingResult:
    """
    Structured result emitted by Module 3 Dynamic Model Router at timestamp t.
    
    Attributes:
        timestamp: ISO 8601 string or timestamp at forecast origin t.
        selected_model: Unique identifier of selected forecasting model.
        regime: Classified RegimeType string.
        confidence: Routing decision confidence score (0.0 to 1.0).
        reason: Human-readable, transparent explanation for routing choice.
        change_probability: Bayesian change-point probability from Module 1.
        fallback_used: True if a fallback model was selected due to preference failure/unavailability.
        sequence_idx: Monotonic sequence index t.
        historical_mae: Walk-forward Mean Absolute Error of selected model up to t.
        metadata: Additional diagnostic contextual details.
    """
    timestamp: str
    selected_model: str
    regime: str
    confidence: float
    reason: str
    change_probability: float
    fallback_used: bool
    sequence_idx: int = 0
    historical_mae: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "selected_model": self.selected_model,
            "regime": self.regime,
            "confidence": self.confidence,
            "reason": self.reason,
            "change_probability": self.change_probability,
            "fallback_used": self.fallback_used,
            "sequence_idx": self.sequence_idx,
            "historical_mae": self.historical_mae,
            "metadata": self.metadata,
        }


@dataclass
class RoutingPolicyConfig:
    """Configurable threshold parameters for rule-based routing policy."""
    normal_demand_preferred: str = "arima"
    peak_shock_preferred: str = "xgboost"
    structural_shift_preferred: str = "xgboost"
    festival_event_preferred: str = "event_aware"
    warming_up_preferred: str = "baseline"
    min_samples_for_statistical: int = 5
    min_samples_for_supervised: int = 6
