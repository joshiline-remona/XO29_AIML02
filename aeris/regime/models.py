"""
Data models and Enums for AERIS Module 2 Regime Detection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RegimeType(str, Enum):
    """Supported electricity demand regimes."""
    NORMAL_DEMAND = "NORMAL_DEMAND"
    PEAK_SHOCK = "PEAK_SHOCK"
    STRUCTURAL_SHIFT = "STRUCTURAL_SHIFT"
    FESTIVAL_EVENT = "FESTIVAL_EVENT"
    WARMING_UP = "WARMING_UP"


@dataclass(frozen=True)
class RegimeClassificationResult:
    """
    Result emitted by Module 2 Regime Classifier at timestamp t.
    
    Attributes:
        timestamp: ISO 8601 string or timestamp.
        regime: Classified RegimeType enum.
        confidence: Classification confidence score (0.0 to 1.0).
        reason: Human-readable, transparent explanation for the classification.
        change_probability: Change-point probability passed from Module 1.
        supporting_features: Dictionary of engineered features used for classification.
        sequence_idx: Monotonic sequence index t.
    """
    timestamp: str
    regime: RegimeType
    confidence: float
    reason: str
    change_probability: float
    supporting_features: Dict[str, Any] = field(default_factory=dict)
    sequence_idx: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "regime": self.regime.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "change_probability": self.change_probability,
            "supporting_features": self.supporting_features,
            "sequence_idx": self.sequence_idx,
        }
