"""
Event-Aware Forecasting Strategy for AERIS Module 3.

Used during FESTIVAL_EVENT regimes strictly when explicit event/contextual
features are available. If no explicit event indicator is provided, it reports
unavailability so the router can trigger a safe fallback.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from aeris.config import ForecastingConfig
from aeris.forecasting.base import BaseForecastModel, ModelStatus
from aeris.ingestion.schemas import DemandObservation
from aeris.utils.logger import default_logger


class EventAwareForecastModel(BaseForecastModel):
    """
    Event-Aware forecasting strategy leveraging explicit context/event indicators.
    """

    def __init__(self, config: Optional[ForecastingConfig] = None):
        self.config = config or ForecastingConfig()
        self._fitted = False
        self._status = ModelStatus.WARMING_UP
        self._base_mean: float = 0.0

    def fit(self, history: List[DemandObservation]) -> None:
        """
        Fit baseline trend/mean on historical demand.
        """
        if not history:
            self._fitted = False
            self._status = ModelStatus.WARMING_UP
            return

        values = [obs.demand_mw for obs in history]
        self._base_mean = float(np.mean(values[-min(len(values), 10):]))
        self._fitted = True
        self._status = ModelStatus.READY

    def predict(
        self,
        history: List[DemandObservation],
        horizon: int = 1,
        context: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Generate event-aware forecast.
        Requires explicit event indicator in context or supporting_features.
        """
        if history:
            self.fit(history)

        if not self._fitted:
            raise RuntimeError(f"{self.model_name} is not fitted.")

        context_dict = context or {}

        # Check for explicit event indicator
        is_event = context_dict.get("is_event", False)
        event_multiplier = context_dict.get("event_multiplier", None)

        if not is_event and event_multiplier is None:
            # Safe refusal if explicit event indicator is missing
            self._status = ModelStatus.FAILED
            default_logger.log_model_prediction_failed(
                self.model_name,
                "No explicit event/context indicator provided in observation."
            )
            raise ValueError("EventAwareForecastModel requires an explicit event/context feature.")

        multiplier = float(event_multiplier) if event_multiplier is not None else 1.15
        last_val = history[-1].demand_mw if history else self._base_mean

        event_adjusted_pred = last_val * multiplier
        return [float(event_adjusted_pred) for _ in range(horizon)]

    def is_fitted(self) -> bool:
        return self._fitted

    def status(self) -> ModelStatus:
        return self._status

    @property
    def model_name(self) -> str:
        return "event_aware"

    def metadata(self) -> Dict[str, Any]:
        meta = super().metadata()
        meta["base_mean"] = self._base_mean
        return meta
