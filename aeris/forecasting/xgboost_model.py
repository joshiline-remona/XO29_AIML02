"""
Supervised XGBoost Forecasting Model for AERIS Module 3.

Used for responsive short-term demand forecasting during PEAK_SHOCK and
STRUCTURAL_SHIFT regimes. Strictly enforces chronological feature engineering
without lookahead / future-data leakage.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from aeris.config import ForecastingConfig
from aeris.forecasting.base import BaseForecastModel, ModelStatus
from aeris.ingestion.schemas import DemandObservation
from aeris.utils.logger import default_logger


class XGBoostForecastModel(BaseForecastModel):
    """
    Supervised XGBoost forecasting model with chronological feature generation.
    """

    def __init__(self, config: Optional[ForecastingConfig] = None):
        self.config = config or ForecastingConfig()
        self._model: Any = None
        self._fitted = False
        self._status = ModelStatus.WARMING_UP
        self.min_history = 6

    def _extract_calendar_features(self, timestamp: str) -> tuple[int, int]:
        """Extract hour and day of week safely from timestamp string."""
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.hour, dt.weekday()
        except Exception:
            return 0, 0

    def _build_feature_vector(
        self,
        history_slice: List[DemandObservation],
        timestamp_str: str
    ) -> List[float]:
        """
        Build feature vector for predicting next point, given observations strictly UP TO t.
        Features: [lag_1, lag_2, lag_3, lag_24, rolling_mean_6, rolling_std_6, hour, day_of_week]
        """
        vals = [obs.demand_mw for obs in history_slice]
        n = len(vals)
        if n == 0:
            return [0.0] * 8

        lag_1 = vals[-1]
        lag_2 = vals[-2] if n >= 2 else lag_1
        lag_3 = vals[-3] if n >= 3 else lag_2
        lag_24 = vals[-24] if n >= 24 else lag_1

        r_slice = vals[-min(n, 6):]
        r_mean = float(np.mean(r_slice))
        r_std = float(np.std(r_slice)) if len(r_slice) > 1 else 0.0

        hour, dow = self._extract_calendar_features(timestamp_str)

        return [lag_1, lag_2, lag_3, lag_24, r_mean, r_std, float(hour), float(dow)]

    def fit(self, history: List[DemandObservation]) -> None:
        """
        Build tabular dataset (X, y) from historical stream and fit XGBoost regressor.
        Sample i uses features from history[0...i-1] to predict history[i].demand_mw.
        """
        if len(history) < self.min_history:
            self._fitted = False
            self._status = ModelStatus.WARMING_UP
            return

        X, y = [], []
        # Build training set strictly using past values
        for i in range(1, len(history)):
            past = history[:i]
            target = history[i].demand_mw
            feat = self._build_feature_vector(past, history[i].timestamp)
            X.append(feat)
            y.append(target)

        if len(X) < 3:
            self._fitted = False
            self._status = ModelStatus.WARMING_UP
            return

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=float)

        try:
            default_logger.log_model_training_started(self.model_name)
            import xgboost as xgb

            self._model = xgb.XGBRegressor(
                n_estimators=self.config.xgboost_n_estimators,
                max_depth=self.config.xgboost_max_depth,
                learning_rate=self.config.xgboost_learning_rate,
                random_state=42,
                verbosity=0
            )
            self._model.fit(X_arr, y_arr)
            self._fitted = True
            self._status = ModelStatus.READY
            default_logger.log_model_training_completed(self.model_name, duration_ms=2.0)
        except Exception as e:
            # Fallback to sklearn GradientBoostingRegressor if xgboost runs into issue
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                self._model = GradientBoostingRegressor(
                    n_estimators=self.config.xgboost_n_estimators,
                    max_depth=self.config.xgboost_max_depth,
                    learning_rate=self.config.xgboost_learning_rate,
                    random_state=42
                )
                self._model.fit(X_arr, y_arr)
                self._fitted = True
                self._status = ModelStatus.READY
                default_logger.log_model_training_completed(self.model_name, duration_ms=2.0)
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
        Predict horizon values using iterative multi-step forecast.
        """
        if history:
            self.fit(history)

        if not self._fitted or self._model is None:
            raise RuntimeError(f"{self.model_name} is not fitted or failed fitting.")

        predictions = []
        simulated_history = list(history)

        for h in range(horizon):
            last_obs = simulated_history[-1]
            feat = self._build_feature_vector(simulated_history, last_obs.timestamp)
            pred_val = float(self._model.predict(np.array([feat]))[0])
            predictions.append(pred_val)

            # Append synthetic observation for auto-regressive multi-step prediction
            synthetic_obs = DemandObservation(
                timestamp=last_obs.timestamp,
                demand_mw=pred_val,
                sequence_idx=last_obs.sequence_idx + h + 1
            )
            simulated_history.append(synthetic_obs)

        return predictions

    def is_fitted(self) -> bool:
        return self._fitted

    def status(self) -> ModelStatus:
        return self._status

    @property
    def model_name(self) -> str:
        return "xgboost"

    def metadata(self) -> Dict[str, Any]:
        meta = super().metadata()
        meta["min_history"] = self.min_history
        meta["n_estimators"] = self.config.xgboost_n_estimators
        return meta
