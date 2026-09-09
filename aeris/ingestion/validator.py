"""
Data Validation Layer for AERIS Streaming Ingestion.
Enforces numerical sanity and strict chronological non-leakage ordering.
"""

import math
from datetime import datetime
from typing import Optional
from aeris.config import ValidationConfig
from aeris.ingestion.schemas import DemandObservation


class ValidationError(ValueError):
    """Raised when an observation fails numerical or formatting validation."""
    pass


class ChronologyError(ValueError):
    """Raised when observations arrive out of chronological order (anti-leakage guard)."""
    pass


class StreamValidator:
    """
    Validates streaming electricity-demand observations in real time.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self._last_parsed_dt: Optional[datetime] = None
        self._last_seq_idx: Optional[int] = None

    def parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string into datetime object."""
        if not isinstance(timestamp_str, str) or not timestamp_str.strip():
            raise ValidationError(f"Invalid timestamp string format: {timestamp_str}")
        
        cleaned = timestamp_str.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            # Fallback for common datetime formats
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
                try:
                    return datetime.strptime(cleaned, fmt)
                except ValueError:
                    pass
            raise ValidationError(f"Could not parse timestamp ISO string: {timestamp_str}")

    def validate_observation(self, observation: DemandObservation) -> None:
        """
        Validates demand value sanity and strict chronological ordering.
        
        Raises:
            ValidationError: If demand_mw is NaN, infinite, or out of bounds.
            ChronologyError: If timestamp is not strictly greater than previous observation.
        """
        val = observation.demand_mw
        
        # 1. Numerical sanity check
        if not isinstance(val, (int, float)):
            raise ValidationError(f"Demand value must be numeric, got {type(val)}: {val}")
        
        if math.isnan(val) or math.isinf(val):
            raise ValidationError(f"Demand value must be finite, got: {val}")
            
        if val < self.config.min_demand_mw or val > self.config.max_demand_mw:
            raise ValidationError(
                f"Demand value {val} MW outside permitted bounds "
                f"[{self.config.min_demand_mw}, {self.config.max_demand_mw}]"
            )

        # 2. Chronological sequence check
        current_dt = self.parse_timestamp(observation.timestamp)
        
        if self.config.strict_chronological and self._last_parsed_dt is not None:
            if current_dt <= self._last_parsed_dt:
                raise ChronologyError(
                    f"Chronological violation detected: current timestamp '{observation.timestamp}' "
                    f"({current_dt}) is not strictly after previous timestamp ({self._last_parsed_dt}). "
                    f"Online processing requires strict time advancement to prevent data leakage."
                )

        if self._last_seq_idx is not None and observation.sequence_idx <= self._last_seq_idx:
            raise ChronologyError(
                f"Sequence index violation: sequence_idx {observation.sequence_idx} <= {self._last_seq_idx}"
            )

        # Update state after successful validation
        self._last_parsed_dt = current_dt
        self._last_seq_idx = observation.sequence_idx

    def reset(self) -> None:
        """Reset validator memory state."""
        self._last_parsed_dt = None
        self._last_seq_idx = None
