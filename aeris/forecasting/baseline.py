"""
Baseline Persistence and Simple Historical Model for AERIS Module 3.

Acts as a reliable, deterministic fallback model when preferred models fail
or have insufficient training history.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from aeris.config import ForecastingConfig
from aeris.forecasting.base import BaseForecastModel, ModelStatus
from aeris.ingestion.schemas import DemandObservation


class BaselinePersistenceModel(BaseForecastModel):
    """
    Simple, deterministic fallback baseline model.
    Strategies supported: 'last_value', 'mean', 'median'.
    """

    def __init__(self, config: Optional[ForecastingConfig] = None):
        self.config = config or ForecastingConfig()
        self._strategy = self.config.baseline_strategy
        self._last_value: Optional[float] = None
        self._fitted = False

    def fit(self, history: List[DemandObservation]) -> None:
        """
        Extract last value, mean, or median from provided demand history.
        """
        if not history:
            self._fitted = False
            return
        
        values = [obs.demand_mw for obs in history]
        if self._strategy == "last_value":
            self._last_value = float(values[-1])
        elif self._strategy == "mean":
            self._last_value = float(np.mean(values))
        elif self._strategy == "median":
            self._last_value = float(np.median(values))
        else:
            self._last_value = float(values[-1])
            
        self._fitted = True

    def predict(
        self,
        history: List[DemandObservation],
        horizon: int = 1,
        context: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Predict horizon values using baseline persistence strategy.
        """
        if not history and not self._fitted:
            raise ValueError("BaselinePersistenceModel has no history to predict from.")
            
        if history:
            self.fit(history)
            
        val = self._last_value if self._last_value is not None else 0.0
        return [float(val) for _ in range(horizon)]

    def is_fitted(self) -> bool:
        return self._fitted

    def status(self) -> ModelStatus:
        if self._fitted:
            return ModelStatus.READY
        return ModelStatus.WARMING_UP

    @property
    def model_name(self) -> str:
        return "baseline"

    def metadata(self) -> Dict[str, Any]:
        meta = super().metadata()
        meta["strategy"] = self._strategy
        meta["last_value"] = self._last_value
        return meta
