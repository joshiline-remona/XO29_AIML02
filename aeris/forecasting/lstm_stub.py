"""
LSTM Forecasting Extension Stub for AERIS Module 3.

Explicit extension point interface for Deep Learning (LSTM / Neural Network)
demand forecasting models. Serves as a documented placeholder without faking
predictions or pretending to be a trained model.
"""

from typing import Any, Dict, List, Optional

from aeris.forecasting.base import BaseForecastModel, ModelStatus
from aeris.ingestion.schemas import DemandObservation


class LSTMForecastModelStub(BaseForecastModel):
    """
    Documented LSTM Model Extension Stub.
    
    This stub implements the BaseForecastModel interface to allow future
    PyTorch/TensorFlow LSTM implementations to be plugged into the ModelRegistry
    without router code changes.
    """

    def __init__(self):
        self._fitted = False
        self._status = ModelStatus.WARMING_UP

    def fit(self, history: List[DemandObservation]) -> None:
        """
        Stub fit method. Clearly documents extension point status.
        """
        # LSTM extension point: deep learning training would occur here
        self._fitted = False
        self._status = ModelStatus.WARMING_UP

    def predict(
        self,
        history: List[DemandObservation],
        horizon: int = 1,
        context: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Explicitly raises NotImplementedError to prevent silent fake predictions.
        """
        raise NotImplementedError(
            "LSTMForecastModelStub is an extension point stub and is not trained. "
            "Please register a fully trained PyTorch/TensorFlow model or use available baseline models."
        )

    def is_fitted(self) -> bool:
        return False

    def status(self) -> ModelStatus:
        return ModelStatus.WARMING_UP

    @property
    def model_name(self) -> str:
        return "lstm"

    def metadata(self) -> Dict[str, Any]:
        meta = super().metadata()
        meta["is_extension_stub"] = True
        meta["note"] = "Explicit LSTM model extension stub point"
        return meta
