"""
AERIS Module 3 Forecasting Models Package.
"""

from aeris.forecasting.arima import ARIMAForecastModel
from aeris.forecasting.base import BaseForecastModel, ForecastResult, ModelStatus
from aeris.forecasting.baseline import BaselinePersistenceModel
from aeris.forecasting.event_aware import EventAwareForecastModel
from aeris.forecasting.lstm_stub import LSTMForecastModelStub
from aeris.forecasting.xgboost_model import XGBoostForecastModel

__all__ = [
    "BaseForecastModel",
    "ModelStatus",
    "ForecastResult",
    "BaselinePersistenceModel",
    "ARIMAForecastModel",
    "XGBoostForecastModel",
    "EventAwareForecastModel",
    "LSTMForecastModelStub",
]
