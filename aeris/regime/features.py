"""
Online Feature Engineering Layer for AERIS Module 2.

Computes rolling demand statistics and contextual time/metadata features strictly online (up to timestamp t)
with zero lookahead or future data leakage.
"""

import math
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional
import numpy as np

from aeris.config import RegimeConfig
from aeris.detection.models import DetectionResult
from aeris.ingestion.schemas import DemandObservation
from aeris.utils.logger import default_logger


class OnlineFeatureExtractor:
    """
    Computes online streaming features for regime classification.
    
    Maintains a rolling historical window buffer of demand observations x_{<=t}.
    Integrates Module 1 Bayesian signals with engineered statistical and calendar features.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.history_buffer: deque = deque(maxlen=self.config.rolling_window)
        self.step_count: int = 0

    def parse_calendar_features(self, timestamp_str: str) -> Dict[str, Any]:
        """Extract hour of day, day of week, and weekend indicator from timestamp."""
        cleaned = timestamp_str.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(cleaned)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(cleaned, fmt)
                    break
                except ValueError:
                    dt = None
            if dt is None:
                return {"hour_of_day": 0, "day_of_week": 0, "is_weekend": False}

        hour = dt.hour
        weekday = dt.weekday()  # Monday=0, Sunday=6
        is_weekend = weekday >= 5
        return {
            "hour_of_day": hour,
            "day_of_week": weekday,
            "is_weekend": is_weekend,
        }

    def compute_features(
        self,
        observation: DemandObservation,
        detection_result: DetectionResult,
        context_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute online feature set for observation at timestamp t.
        
        Uses only data up to timestamp t (strict non-leakage).
        """
        self.step_count += 1
        x_t = float(observation.demand_mw)
        
        # 1. Historical rolling window statistics BEFORE adding x_t
        if len(self.history_buffer) > 0:
            hist_array = np.array(self.history_buffer, dtype=np.float64)
            prev_rolling_mean = float(np.mean(hist_array))
            prev_rolling_std = float(np.std(hist_array))
            prev_val = float(self.history_buffer[-1])
        else:
            prev_rolling_mean = x_t
            prev_rolling_std = 1.0
            prev_val = x_t
            
        # Update history buffer with current observation x_t
        self.history_buffer.append(x_t)
        
        # 2. Statistical features
        curr_hist = np.array(self.history_buffer, dtype=np.float64)
        curr_rolling_mean = float(np.mean(curr_hist))
        curr_rolling_std = float(np.std(curr_hist))
        
        deviation = x_t - prev_rolling_mean
        safe_std = max(prev_rolling_std, 1.0)
        z_score = deviation / safe_std
        
        pct_change = (x_t - prev_val) / max(abs(prev_val), 1.0)
        volatility = curr_rolling_std / max(abs(curr_rolling_mean), 1.0)
        
        # Trend / slope over recent window
        if len(curr_hist) >= 3:
            indices = np.arange(len(curr_hist))
            trend_slope = float(np.polyfit(indices, curr_hist, 1)[0])
        else:
            trend_slope = 0.0

        # 3. Time and Calendar features
        calendar_feats = self.parse_calendar_features(observation.timestamp)
        
        # 4. Contextual Metadata features (optional)
        metadata = observation.metadata.copy()
        if context_override:
            metadata.update(context_override)
            
        is_event = bool(metadata.get("is_event", False) or metadata.get("is_holiday", False) or metadata.get("festival", False))
        event_name = str(metadata.get("event_name", metadata.get("festival_name", "")))
        temperature = metadata.get("temperature", None)

        # 5. Combined feature dictionary
        features = {
            "timestamp": observation.timestamp,
            "demand_mw": x_t,
            "sequence_idx": observation.sequence_idx,
            "step_count": self.step_count,
            "rolling_mean": curr_rolling_mean,
            "rolling_std": curr_rolling_std,
            "prev_rolling_mean": prev_rolling_mean,
            "prev_rolling_std": prev_rolling_std,
            "deviation_from_baseline": deviation,
            "z_score": z_score,
            "pct_change": pct_change,
            "volatility": volatility,
            "trend_slope": trend_slope,
            # Module 1 Signals
            "change_point_prob": detection_result.change_point_prob,
            "map_run_length": detection_result.map_run_length,
            "change_detected": detection_result.change_detected,
            "detector_status": detection_result.detector_status.value,
            "posterior_mean": detection_result.posterior_mean,
            "posterior_std": detection_result.posterior_std,
            # Calendar & Context
            "hour_of_day": calendar_feats["hour_of_day"],
            "day_of_week": calendar_feats["day_of_week"],
            "is_weekend": calendar_feats["is_weekend"],
            "is_event": is_event,
            "event_name": event_name,
            "temperature": temperature,
        }
        
        default_logger.log_regime_features_updated(
            observation.timestamp,
            z_score,
            curr_rolling_mean,
            curr_rolling_std
        )
        
        return features

    def reset(self) -> None:
        """Reset online feature extractor memory."""
        self.history_buffer.clear()
        self.step_count = 0
