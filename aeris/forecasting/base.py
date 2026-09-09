"""
Base interfaces and dataclasses for AERIS Module 3 Forecasting Models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from aeris.ingestion.schemas import DemandObservation


class ModelStatus(str, Enum):
    """Execution readiness status for forecasting models."""
    READY = "READY"
    WARMING_UP = "WARMING_UP"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ForecastResult:
    """
    Result emitted by a forecasting model or router at timestamp t.
    
    Attributes:
        timestamp: ISO 8601 string or timestamp of forecast origin t.
        forecast_horizon: Number of steps ahead predicted.
        predicted_values: List of point predictions [y_hat_{t+1}, ..., y_hat_{t+h}].
        selected_model: Name of the model that generated this forecast.
        regime: Current regime under which model was executed.
        routing_confidence: Router confidence score.
        model_metadata: Auxiliary information (e.g., training MSE, feature count).
    """
    timestamp: str
    forecast_horizon: int
    predicted_values: List[float]
    selected_model: str
    regime: str
    routing_confidence: float
    model_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "forecast_horizon": self.forecast_horizon,
            "predicted_values": self.predicted_values,
            "selected_model": self.selected_model,
            "regime": self.regime,
            "routing_confidence": self.routing_confidence,
            "model_metadata": self.model_metadata,
        }


class BaseForecastModel(ABC):
    """
    Abstract base class for all AERIS forecasting models.
    """

    @abstractmethod
    def fit(self, history: List[DemandObservation]) -> None:
        """
        Fit model parameters using historical observations up to timestamp t.
        Must execute strictly chronologically with no future-data leakage.
        """
        pass

    @abstractmethod
    def predict(
        self,
        history: List[DemandObservation],
        horizon: int = 1,
        context: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Generate point predictions for future steps [t+1, ..., t+horizon].
        Returns list of predicted values (MW).
        """
        pass

    @abstractmethod
    def is_fitted(self) -> bool:
        """Return True if model is trained and ready for prediction."""
        pass

    @abstractmethod
    def status(self) -> ModelStatus:
        """Return current ModelStatus (READY, WARMING_UP, FAILED)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return unique model identifier string."""
        pass

    def metadata(self) -> Dict[str, Any]:
        """Return model metadata dictionary."""
        return {
            "model_name": self.model_name,
            "is_fitted": self.is_fitted(),
            "status": self.status().value,
        }
