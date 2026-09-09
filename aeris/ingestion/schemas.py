"""
Data schemas for AERIS Data Ingestion.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DemandObservation:
    """
    Chronological electricity-demand observation.
    
    Attributes:
        timestamp: ISO 8601 string or valid datetime string.
        demand_mw: Observed demand value in MW.
        sequence_idx: Monotonic sequence index t (0-indexed).
        metadata: Optional dictionary for contextual stream metadata.
    """
    timestamp: str
    demand_mw: float
    sequence_idx: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "demand_mw": self.demand_mw,
            "sequence_idx": self.sequence_idx,
            "metadata": self.metadata,
        }
