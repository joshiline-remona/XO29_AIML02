"""
Dynamic Forecast Model Router for AERIS Module 3 (ROUTE Stage).

Coordinates regime output consumption, policy evaluation, model registry lookup,
robust fallback chains, walk-forward performance tracking, and forecast result generation.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


from aeris.config import ForecastingConfig, RoutingConfig
from aeris.detection.models import DetectionResult
from aeris.forecasting.arima import ARIMAForecastModel
from aeris.forecasting.base import BaseForecastModel, ForecastResult, ModelStatus
from aeris.forecasting.baseline import BaselinePersistenceModel
from aeris.forecasting.event_aware import EventAwareForecastModel
from aeris.forecasting.lstm_stub import LSTMForecastModelStub
from aeris.forecasting.xgboost_model import XGBoostForecastModel
from aeris.ingestion.schemas import DemandObservation
from aeris.regime.models import RegimeClassificationResult
from aeris.routing.models import RoutingResult
from aeris.routing.policies import RuleBasedRoutingPolicy
from aeris.routing.registry import ModelRegistry
from aeris.utils.logger import default_logger


class DynamicModelRouter:
    """
    Main Module 3 Router coordinating dynamic forecast model selection and execution.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        policy: Optional[RuleBasedRoutingPolicy] = None,
        routing_config: Optional[RoutingConfig] = None,
        forecasting_config: Optional[ForecastingConfig] = None,
    ):
        self.routing_config = routing_config or RoutingConfig()
        self.forecasting_config = forecasting_config or ForecastingConfig()
        self.registry = registry or ModelRegistry()
        self.policy = policy or RuleBasedRoutingPolicy(config=self.routing_config)

        self._history: List[DemandObservation] = []
        self._walkforward_errors: Dict[str, List[float]] = {}
        self._pending_forecasts: Dict[str, float] = {}

        # Register default models if registry is empty
        if not self.registry.available_models():
            self._register_default_models()

    def _register_default_models(self) -> None:
        """Register default model plugins into registry."""
        self.registry.register(BaselinePersistenceModel(self.forecasting_config))
        self.registry.register(ARIMAForecastModel(self.forecasting_config))
        self.registry.register(XGBoostForecastModel(self.forecasting_config))
        self.registry.register(EventAwareForecastModel(self.forecasting_config))
        self.registry.register(LSTMForecastModelStub())

    def update_walkforward_actual(self, actual_observation: DemandObservation) -> None:
        """
        Walk-forward validation update: compare actual observation at t against prior forecast.
        Maintains chronological evaluation order without future lookahead.
        """
        ts = actual_observation.timestamp
        if ts in self._pending_forecasts:
            pred_val, model_name = self._pending_forecasts.pop(ts)
            err = abs(actual_observation.demand_mw - pred_val)
            if model_name not in self._walkforward_errors:
                self._walkforward_errors[model_name] = []
            self._walkforward_errors[model_name].append(err)

    def get_historical_mae(self, model_name: str) -> Optional[float]:
        """Compute Mean Absolute Error for a model over chronological walk-forward evaluations."""
        errors = self._walkforward_errors.get(model_name, [])
        if not errors:
            return None
        import numpy as np
        return float(np.mean(errors))

    def route_and_predict(
        self,
        observation: DemandObservation,
        detection_result: DetectionResult,
        regime_result: RegimeClassificationResult,
        horizon: Optional[int] = None,
        context_override: Optional[Dict[str, Any]] = None
    ) -> Tuple[RoutingResult, ForecastResult]:
        """
        Execute Module 3 online routing and prediction pipeline for timestamp t.
        
        Args:
            observation: Current demand observation at t.
            detection_result: Module 1 Bayesian detection output at t.
            regime_result: Module 2 contextual regime classification at t.
            horizon: Forecast horizon steps ahead.
            context_override: Additional runtime context features.
            
        Returns:
            Tuple of (RoutingResult, ForecastResult)
        """
        h = horizon or self.forecasting_config.forecast_horizon

        # 1. Update walk-forward accuracy from prior predictions
        self.update_walkforward_actual(observation)

        # 2. Append observation to chronological history
        self._history.append(observation)

        # 3. Evaluate Policy to get preferred model candidate
        pref_model_name, conf, reason, policy_fallback = self.policy.select_model(
            regime_result=regime_result,
            history_length=len(self._history),
            registry=self.registry,
            context_override=context_override
        )

        # Build combined context dictionary
        context_dict = dict(regime_result.supporting_features)
        if context_override:
            context_dict.update(context_override)

        selected_model_name = pref_model_name
        fallback_used = policy_fallback
        predicted_values: List[float] = []

        # 4. Attempt to execute model with controlled fallback chain
        fallback_candidates = [pref_model_name] + [
            m for m in self.routing_config.fallback_order if m != pref_model_name
        ]
        if "baseline" not in fallback_candidates:
            fallback_candidates.append("baseline")

        executed_model: Optional[BaseForecastModel] = None
        execution_error: Optional[str] = None

        for candidate_name in fallback_candidates:
            if not self.registry.is_registered(candidate_name):
                continue

            model = self.registry.get(candidate_name)
            try:
                # Attempt fit and predict
                model.fit(self._history)
                preds = model.predict(self._history, horizon=h, context=context_dict)

                if preds and len(preds) == h and not any(np.isnan(p) for p in preds):
                    selected_model_name = candidate_name
                    predicted_values = preds
                    executed_model = model
                    if candidate_name != pref_model_name:
                        fallback_used = True
                        reason = (
                            f"Fallback to '{candidate_name}' executed because preferred model "
                            f"'{pref_model_name}' was unavailable or failed fit/prediction."
                        )
                        default_logger.log_model_fallback(
                            observation.timestamp,
                            pref_model_name,
                            candidate_name,
                            reason
                        )
                    break
            except Exception as e:
                execution_error = str(e)
                default_logger.log_model_prediction_failed(candidate_name, str(e))
                continue

        # Ultimate safety guarantee: if all fails, fallback to simple last value persistence
        if not executed_model or not predicted_values:
            selected_model_name = "baseline"
            fallback_used = True
            reason = f"Emergency baseline fallback triggered after execution error: {execution_error}"
            last_mw = observation.demand_mw
            predicted_values = [last_mw for _ in range(h)]

        mae = self.get_historical_mae(selected_model_name)

        # 5. Log structured routing decision event
        default_logger.log_routing_decision(
            timestamp=observation.timestamp,
            selected_model=selected_model_name,
            regime=regime_result.regime.value if hasattr(regime_result.regime, "value") else str(regime_result.regime),
            confidence=conf,
            reason=reason,
            change_prob=detection_result.change_point_prob,
            fallback_used=fallback_used
        )

        routing_res = RoutingResult(
            timestamp=observation.timestamp,
            selected_model=selected_model_name,
            regime=regime_result.regime.value if hasattr(regime_result.regime, "value") else str(regime_result.regime),
            confidence=conf,
            reason=reason,
            change_probability=detection_result.change_point_prob,
            fallback_used=fallback_used,
            sequence_idx=observation.sequence_idx,
            historical_mae=mae,
            metadata={"history_length": len(self._history)}
        )

        forecast_res = ForecastResult(
            timestamp=observation.timestamp,
            forecast_horizon=h,
            predicted_values=predicted_values,
            selected_model=selected_model_name,
            regime=routing_res.regime,
            routing_confidence=conf,
            model_metadata={
                "fallback_used": fallback_used,
                "historical_mae": mae,
                "sequence_idx": observation.sequence_idx,
            }
        )

        return routing_res, forecast_res

    def reset(self) -> None:
        """Reset internal history and state."""
        self._history.clear()
        self._walkforward_errors.clear()
        self._pending_forecasts.clear()
