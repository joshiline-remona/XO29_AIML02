"""
Lightweight ARIMA / Statistical Forecasting Model for AERIS Module 3.

Used primarily for smooth, predictable demand regimes (e.g., NORMAL_DEMAND).
Handles fitting errors and small historical samples safely with clean fallbacks.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from aeris.config import ForecastingConfig
from aeris.forecasting.base import BaseForecastModel, ModelStatus
from aeris.ingestion.schemas import DemandObservation
from aeris.utils.logger import default_logger


class ARIMAForecastModel(BaseForecastModel):
    """
    Lightweight statistical ARIMA / Exponential Smoothing forecasting model.
    """

    def __init__(self, config: Optional[ForecastingConfig] = None):
        self.config = config or ForecastingConfig()
        self._fitted_model: Any = None
        self._fitted = False
        self._status = ModelStatus.WARMING_UP
        self.min_history = 5

    def fit(self, history: List[DemandObservation]) -> None:
        """
        Fit ARIMA / ETS model on historical demand values up to timestamp t.
        """
        if len(history) < self.min_history:
            self._fitted = False
            self._status = ModelStatus.WARMING_UP
            return

        values = np.array([obs.demand_mw for obs in history], dtype=float)

        try:
            default_logger.log_model_training_started(self.model_name)
            # Use lightweight Exponential Smoothing or AutoReg/ARIMA from statsmodels
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            # If sample size is small, use simple ExponentialSmoothing without seasonal
            if len(values) < 24:
                model = ExponentialSmoothing(
                    values,
                    trend="add" if len(values) >= 10 else None,
                    seasonal=None,
                    initialization_method="estimated"
                )
            else:
                sp = min(self.config.arima_seasonal_period or 24, len(values) // 2)
                sp = max(sp, 2)
                model = ExponentialSmoothing(
                    values,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=sp,
                    initialization_method="estimated"
                )

            self._fitted_model = model.fit(optimized=True)
            self._fitted = True
            self._status = ModelStatus.READY
            default_logger.log_model_training_completed(self.model_name, duration_ms=1.0)
        except Exception as e:
            # Fallback to simple AutoReg or ARIMA(1,1,0) if Exponential Smoothing fails
            try:
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(values, order=(1, 1, 0))
                self._fitted_model = model.fit()
                self._fitted = True
                self._status = ModelStatus.READY
                default_logger.log_model_training_completed(self.model_name, duration_ms=1.0)
            except Exception as inner_e:
                self._fitted = False
                self._status = ModelStatus.FAILED
                default_logger.log_model_prediction_failed(self.model_name, str(inner_e))

    def predict(
        self,
        history: List[DemandObservation],
        horizon: int = 1,
        context: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Generate point predictions for future horizon [t+1 ... t+horizon].
        """
        if history:
            self.fit(history)

        if not self._fitted or self._fitted_model is None:
            raise RuntimeError(f"{self.model_name} is not fitted or failed fitting.")

        try:
            forecasts = self._fitted_model.forecast(steps=horizon)
            return [float(x) for x in forecasts]
        except Exception as e:
            self._status = ModelStatus.FAILED
            default_logger.log_model_prediction_failed(self.model_name, str(e))
            raise RuntimeError(f"ARIMA prediction failed: {e}") from e

    def is_fitted(self) -> bool:
        return self._fitted

    def status(self) -> ModelStatus:
        return self._status

    @property
    def model_name(self) -> str:
        return "arima"

    def metadata(self) -> Dict[str, Any]:
        meta = super().metadata()
        meta["min_history"] = self.min_history
        meta["has_fitted_weights"] = self._fitted_model is not None
        return meta
