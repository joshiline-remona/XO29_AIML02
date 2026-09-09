"""
Model Registry plugin architecture for AERIS Module 3.

Allows dynamic registration, discovery, status inspection, and management
of forecasting models without modifying core routing logic.
"""

from typing import Dict, List, Optional

from aeris.forecasting.base import BaseForecastModel, ModelStatus


class ModelRegistry:
    """
    Plugin-style central registry for AERIS forecasting models.
    """

    def __init__(self):
        self._models: Dict[str, BaseForecastModel] = {}

    def register(self, model: BaseForecastModel, overwrite: bool = True) -> None:
        """
        Register a new forecasting model instance.
        """
        name = model.model_name
        if name in self._models and not overwrite:
            raise KeyError(f"Model '{name}' is already registered.")
        self._models[name] = model

    def unregister(self, model_name: str) -> None:
        """
        Remove a model from the registry.
        """
        if model_name in self._models:
            del self._models[model_name]

    def get(self, model_name: str) -> BaseForecastModel:
        """
        Retrieve model by name. Raises KeyError if not found.
        """
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not found in registry. Available: {self.available_models()}")
        return self._models[model_name]

    def is_registered(self, model_name: str) -> bool:
        """Return True if model is registered."""
        return model_name in self._models

    def available_models(self) -> List[str]:
        """Return list of names of registered models."""
        return list(self._models.keys())

    def get_model_status(self, model_name: str) -> ModelStatus:
        """Query execution readiness status of model."""
        model = self.get(model_name)
        return model.status()
