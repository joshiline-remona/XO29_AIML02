"""
Transparent and explainable routing policies for AERIS Module 3.

Evaluates regime signals, change probability, demand history, model readiness,
and contextual event availability to select the target forecasting model.
"""

from typing import Any, Dict, List, Optional

from aeris.config import RoutingConfig
from aeris.forecasting.base import ModelStatus
from aeris.regime.models import RegimeClassificationResult, RegimeType
from aeris.routing.models import RoutingPolicyConfig
from aeris.routing.registry import ModelRegistry


class RuleBasedRoutingPolicy:
    """
    Data-aware, transparent rule-based routing policy.
    Separates routing rules from policy configuration.
    """

    def __init__(
        self,
        config: Optional[RoutingConfig] = None,
        policy_config: Optional[RoutingPolicyConfig] = None
    ):
        self.config = config or RoutingConfig()
        self.policy_config = policy_config or RoutingPolicyConfig()

    def select_model(
        self,
        regime_result: RegimeClassificationResult,
        history_length: int,
        registry: ModelRegistry,
        context_override: Optional[Dict[str, Any]] = None
    ) -> tuple[str, float, str, bool]:
        """
        Evaluate inputs and determine selected model name, confidence, reason, and fallback flag.
        
        Returns:
            (selected_model_name, confidence, reason_text, fallback_used_flag)
        """
        regime = regime_result.regime
        regime_str = regime.value if isinstance(regime, RegimeType) else str(regime)
        features = regime_result.supporting_features or {}
        if context_override:
            features = {**features, **context_override}

        change_prob = regime_result.change_probability
        reg_conf = regime_result.confidence

        # 1. Warmup Check: insufficient historical observations
        if history_length < self.config.min_history_steps or regime_str == "WARMING_UP":
            return (
                "baseline",
                1.0,
                "Fallback baseline selected due to warming up window / insufficient historical observations",
                False if regime_str == "WARMING_UP" else True
            )

        # 2. FESTIVAL_EVENT Regime Evaluation
        if regime_str == "FESTIVAL_EVENT":
            has_event_feature = bool(features.get("is_event", False)) or bool(features.get("event_multiplier"))
            if has_event_feature and registry.is_registered("event_aware"):
                return (
                    "event_aware",
                    0.95,
                    "Event-aware strategy selected because an explicit event indicator feature is available.",
                    False
                )
            else:
                # Event feature absent -> safe fallback
                fallback_target = "xgboost" if registry.is_registered("xgboost") else "arima"
                return (
                    fallback_target,
                    round(reg_conf * 0.85, 4),
                    "FESTIVAL_EVENT regime detected but explicit event feature is missing; safely falling back to responsive model.",
                    True
                )

        # 3. PEAK_SHOCK Regime Evaluation
        if regime_str == "PEAK_SHOCK":
            if registry.is_registered("xgboost"):
                return (
                    "xgboost",
                    min(0.98, max(0.75, reg_conf)),
                    "XGBoost selected because the current regime indicates PEAK_SHOCK demand surge and recent feature history is available.",
                    False
                )
            elif registry.is_registered("arima"):
                return (
                    "arima",
                    0.70,
                    "ARIMA fallback selected for PEAK_SHOCK regime because XGBoost is unavailable.",
                    True
                )

        # 4. STRUCTURAL_SHIFT Regime Evaluation
        if regime_str == "STRUCTURAL_SHIFT":
            if registry.is_registered("xgboost"):
                return (
                    "xgboost",
                    min(0.99, max(0.80, change_prob)),
                    "XGBoost selected for STRUCTURAL_SHIFT to adapt and retrain using recent post-shift observations.",
                    False
                )
            elif registry.is_registered("arima"):
                return (
                    "arima",
                    0.75,
                    "ARIMA selected for STRUCTURAL_SHIFT regime.",
                    False
                )

        # 5. NORMAL_DEMAND Regime Evaluation
        if regime_str == "NORMAL_DEMAND":
            if registry.is_registered("arima") and history_length >= 5:
                return (
                    "arima",
                    min(0.95, max(0.70, reg_conf)),
                    "Statistical model (ARIMA) selected because regime is NORMAL_DEMAND and sufficient history exists.",
                    False
                )
            elif registry.is_registered("xgboost"):
                return (
                    "xgboost",
                    0.80,
                    "XGBoost selected as baseline for NORMAL_DEMAND regime.",
                    False
                )

        # Default fallback to baseline
        return (
            "baseline",
            0.60,
            "Default persistence baseline selected by policy.",
            True
        )
