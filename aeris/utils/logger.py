"""
Structured Logger for AERIS Module 1 & Module 2.

Emits structured events (JSON or formatted text) for:
- OBSERVATION_RECEIVED
- CHANGE_PROBABILITY_UPDATED
- ANOMALY_DETECTED
- CHANGE_DETECTED
- REGIME_FEATURES_UPDATED
- REGIME_CLASSIFIED
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

    def log_regime_features_updated(self, timestamp: str, z_score: float, rolling_mean: float, rolling_std: float) -> None:
        self._log_event(
            "REGIME_FEATURES_UPDATED",
            {
                "timestamp": timestamp,
                "z_score": round(z_score, 4),
                "rolling_mean": round(rolling_mean, 2),
                "rolling_std": round(rolling_std, 2)
            },
            level="DEBUG"
        )

    def log_regime_classified(
        self,
        timestamp: str,
        regime: str,
        confidence: float,
        reason: str,
        change_prob: float
    ) -> None:
        self._log_event(
            "REGIME_CLASSIFIED",
            {
                "timestamp": timestamp,
                "regime": regime,
                "confidence": round(confidence, 4),
                "reason": reason,
                "change_probability": round(change_prob, 6)
            }
        )

    def log_validation_error(self, timestamp: Any, demand_mw: Any, reason: str) -> None:
        self._log_event(
            "VALIDATION_ERROR",
            {"timestamp": timestamp, "demand_mw": demand_mw, "reason": reason},
            level="ERROR"
        )

    def log_routing_decision(
        self,
        timestamp: str,
        selected_model: str,
        regime: str,
        confidence: float,
        reason: str,
        change_prob: float,
        fallback_used: bool
    ) -> None:
        self._log_event(
            "ROUTING_DECISION",
            {
                "timestamp": timestamp,
                "selected_model": selected_model,
                "regime": regime,
                "confidence": round(confidence, 4),
                "reason": reason,
                "change_probability": round(change_prob, 6),
                "fallback_used": fallback_used
            }
        )

    def log_model_selected(self, timestamp: str, selected_model: str, regime: str) -> None:
        self._log_event(
            "MODEL_SELECTED",
            {
                "timestamp": timestamp,
                "selected_model": selected_model,
                "regime": regime
            }
        )

    def log_model_fallback(self, timestamp: str, failed_model: str, fallback_model: str, reason: str) -> None:
        self._log_event(
            "MODEL_FALLBACK",
            {
                "timestamp": timestamp,
                "failed_model": failed_model,
                "fallback_model": fallback_model,
                "reason": reason
            },
            level="WARNING"
        )

    def log_model_training_started(self, model_name: str) -> None:
        self._log_event(
            "MODEL_TRAINING_STARTED",
            {"model_name": model_name}
        )

    def log_model_training_completed(self, model_name: str, duration_ms: float) -> None:
        self._log_event(
            "MODEL_TRAINING_COMPLETED",
            {"model_name": model_name, "duration_ms": round(duration_ms, 2)}
        )

    def log_model_prediction_failed(self, model_name: str, error_msg: str) -> None:
        self._log_event(
            "MODEL_PREDICTION_FAILED",
            {"model_name": model_name, "error": error_msg},
            level="ERROR"
        )


# Global default logger instance
default_logger = AERISLogger()

