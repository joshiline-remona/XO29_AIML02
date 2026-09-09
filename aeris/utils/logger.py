"""
Structured Logger for AERIS Module 1.

Emits structured events (JSON or formatted text) for:
- OBSERVATION_RECEIVED
- CHANGE_PROBABILITY_UPDATED
- ANOMALY_DETECTED
- CHANGE_DETECTED
- STREAM_ERROR / VALIDATION_ERROR
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from aeris.config import LoggerConfig


class AERISLogger:
    """Structured Logger for AERIS online stream processing and detection events."""
    
    def __init__(self, config: Optional[LoggerConfig] = None):
        self.config = config or LoggerConfig()
        self.logger = logging.getLogger("AERIS")
        self.logger.setLevel(getattr(logging, self.config.level.upper(), logging.INFO))
        self.logger.handlers.clear()
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def _log_event(self, event_type: str, details: Dict[str, Any], level: str = "INFO") -> None:
        """Internal log formatter."""
        log_payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "data": details
        }
        
        if self.config.json_format:
            msg = json.dumps(log_payload, default=str)
        else:
            msg = f"[{log_payload['timestamp_utc']}] [{event_type}] {details}"
            
        log_fn = getattr(self.logger, level.lower(), self.logger.info)
        log_fn(msg)

    def log_observation_received(self, timestamp: str, demand_mw: float, step: int) -> None:
        self._log_event(
            "OBSERVATION_RECEIVED",
            {"timestamp": timestamp, "demand_mw": demand_mw, "step": step}
        )

    def log_change_probability_updated(
        self,
        timestamp: str,
        demand_mw: float,
        change_prob: float,
        map_run_length: int,
        status: str
    ) -> None:
        self._log_event(
            "CHANGE_PROBABILITY_UPDATED",
            {
                "timestamp": timestamp,
                "demand_mw": demand_mw,
                "change_probability": round(change_prob, 6),
                "map_run_length": map_run_length,
                "status": status
            }
        )

    def log_anomaly_detected(self, timestamp: str, demand_mw: float, change_prob: float) -> None:
        self._log_event(
            "ANOMALY_DETECTED",
            {
                "timestamp": timestamp,
                "demand_mw": demand_mw,
                "change_probability": round(change_prob, 6),
                "note": "Isolated temporary anomaly suspected; not triggering structural change"
            },
            level="WARNING"
        )

    def log_change_detected(
        self,
        timestamp: str,
        demand_mw: float,
        change_prob: float,
        posterior_mean: float,
        posterior_std: float
    ) -> None:
        self._log_event(
            "CHANGE_DETECTED",
            {
                "timestamp": timestamp,
                "demand_mw": demand_mw,
                "change_probability": round(change_prob, 6),
                "new_regime_mean": round(posterior_mean, 2),
                "new_regime_std": round(posterior_std, 2)
            },
            level="WARNING"
        )

    def log_validation_error(self, timestamp: Any, demand_mw: Any, reason: str) -> None:
        self._log_event(
            "VALIDATION_ERROR",
            {"timestamp": timestamp, "demand_mw": demand_mw, "reason": reason},
            level="ERROR"
        )


# Global default logger instance
default_logger = AERISLogger()
