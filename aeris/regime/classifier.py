"""
Regime Classifiers for AERIS Module 2.

Implements transparent, explainable rule-based classification into:
- NORMAL_DEMAND
- PEAK_SHOCK
- STRUCTURAL_SHIFT
- FESTIVAL_EVENT
- WARMING_UP
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from aeris.config import RegimeConfig
from aeris.detection.models import DetectionResult
from aeris.ingestion.schemas import DemandObservation
from aeris.regime.features import OnlineFeatureExtractor
from aeris.regime.models import RegimeClassificationResult, RegimeType
from aeris.utils.logger import default_logger


class BaseRegimeClassifier(ABC):
    """Abstract Base Class for AERIS Regime Classifiers."""
    
    @abstractmethod
    def classify_regime(
        self,
        observation: DemandObservation,
        detection_result: DetectionResult,
        context_override: Optional[Dict[str, Any]] = None
    ) -> RegimeClassificationResult:
        """
        Classify demand regime for observation at time t.
        """
        pass


class RuleBasedRegimeClassifier(BaseRegimeClassifier):
    """
    Transparent, explainable statistical rule-based regime classifier.
    
    Combines Module 1 Bayesian signals with online engineered demand features
    and optional calendar/metadata context.
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.feature_extractor = OnlineFeatureExtractor(config=self.config)

    def classify_regime(
        self,
        observation: DemandObservation,
        detection_result: DetectionResult,
        context_override: Optional[Dict[str, Any]] = None
    ) -> RegimeClassificationResult:
        """
        Classify current demand observation into a regime strictly online.
        """
        features = self.feature_extractor.compute_features(
            observation, detection_result, context_override=context_override
        )
        
        step_count = features["step_count"]
        z_score = features["z_score"]
        deviation = features["deviation_from_baseline"]
        change_prob = features["change_point_prob"]
        change_detected = features["change_detected"]
        is_event = features["is_event"]
        event_name = features["event_name"]
        
        # 1. Warmup Check (Insufficient historical data)
        if step_count <= self.config.warmup_steps:
            regime = RegimeType.WARMING_UP
            confidence = 1.0
            reason = "Insufficient historical observations for regime classification (warming up window)"
            
        # 2. Calendar / Festival Event Check
        elif is_event:
            regime = RegimeType.FESTIVAL_EVENT
            confidence = 0.95
            if event_name:
                reason = f"Festival/calendar event active ({event_name})"
            else:
                reason = "Festival/calendar event feature active"
                
        # 3. Persistent Structural Shift Check
        elif change_detected:
            regime = RegimeType.STRUCTURAL_SHIFT
            confidence = min(0.99, max(0.75, change_prob if change_prob > 0 else 0.85))
            reason = "Persistent demand baseline shift confirmed by Bayesian change detector"
            
        # 4. Sudden Peak Demand Shock Check (Extreme short-term deviation / spike)
        elif abs(z_score) >= self.config.peak_shock_z_threshold or detection_result.detector_status.value == "ANOMALY_SUSPECTED":
            regime = RegimeType.PEAK_SHOCK
            confidence = min(0.95, max(0.70, abs(z_score) / 4.0))
            reason = f"Peak shock detected from demand behaviour (Z-score = {z_score:.2f}, deviation = {deviation:.1f} MW)"
            
        # 5. Normal Demand Check
        else:
            regime = RegimeType.NORMAL_DEMAND
            confidence = max(0.60, 1.0 - change_prob)
            reason = "Normal demand pattern within expected operational variance"

        default_logger.log_regime_classified(
            observation.timestamp,
            regime.value,
            confidence,
            reason,
            change_prob
        )

        return RegimeClassificationResult(
            timestamp=observation.timestamp,
            regime=regime,
            confidence=round(confidence, 4),
            reason=reason,
            change_probability=change_prob,
            supporting_features=features,
            sequence_idx=observation.sequence_idx,
        )

    def reset(self) -> None:
        """Reset classifier state."""
        self.feature_extractor.reset()
