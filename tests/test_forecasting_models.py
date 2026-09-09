"""
Unit tests for AERIS Module 3 Forecasting Models.
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

from aeris.config import ForecastingConfig
from aeris.forecasting.arima import ARIMAForecastModel
from aeris.forecasting.base import BaseForecastModel, ModelStatus
from aeris.forecasting.baseline import BaselinePersistenceModel
from aeris.forecasting.event_aware import EventAwareForecastModel
from aeris.forecasting.lstm_stub import LSTMForecastModelStub
from aeris.forecasting.xgboost_model import XGBoostForecastModel
from aeris.ingestion.schemas import DemandObservation


def make_history(n: int = 20, base_demand: float = 500.0) -> list[DemandObservation]:
    """Helper generator for deterministic synthetic demand stream."""
    start_time = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)
    obs_list = []
    for i in range(n):
        ts = (start_time + timedelta(hours=i)).isoformat()
        # Synthetic daily pattern + mild noise
        val = base_demand + 50.0 * np.sin(i * np.pi / 12) + (i % 3)
        obs_list.append(DemandObservation(timestamp=ts, demand_mw=float(val), sequence_idx=i + 1))
    return obs_list


class TestBaselinePersistenceModel:
    def test_fit_and_predict_last_value(self):
        history = make_history(10, base_demand=500.0)
        model = BaselinePersistenceModel()
        model.fit(history)
        assert model.is_fitted()
        assert model.status() == ModelStatus.READY

        preds = model.predict(history, horizon=3)
        assert len(preds) == 3
        assert preds[0] == pytest.approx(history[-1].demand_mw)

    def test_predict_mean_strategy(self):
        config = ForecastingConfig(baseline_strategy="mean")
        history = make_history(5, base_demand=100.0)
        model = BaselinePersistenceModel(config=config)
        model.fit(history)
        mean_val = float(np.mean([o.demand_mw for o in history]))
        preds = model.predict(history, horizon=2)
        assert preds[0] == pytest.approx(mean_val)


class TestARIMAForecastModel:
    def test_insufficient_history_warming_up(self):
        history = make_history(3)
        model = ARIMAForecastModel()
        model.fit(history)
        assert not model.is_fitted()
        assert model.status() == ModelStatus.WARMING_UP

    def test_fit_and_predict(self):
        history = make_history(15, base_demand=400.0)
        model = ARIMAForecastModel()
        model.fit(history)
        assert model.is_fitted()
        assert model.status() == ModelStatus.READY

        preds = model.predict(history, horizon=2)
        assert len(preds) == 2
        assert all(np.isfinite(p) for p in preds)


class TestXGBoostForecastModel:
    def test_insufficient_history(self):
        history = make_history(3)
        model = XGBoostForecastModel()
        model.fit(history)
        assert not model.is_fitted()
        assert model.status() == ModelStatus.WARMING_UP

    def test_fit_and_predict(self):
        history = make_history(15, base_demand=600.0)
        model = XGBoostForecastModel()
        model.fit(history)
        assert model.is_fitted()
        assert model.status() == ModelStatus.READY

        preds = model.predict(history, horizon=2)
        assert len(preds) == 2
        assert all(np.isfinite(p) for p in preds)


class TestEventAwareForecastModel:
    def test_predict_with_explicit_event_feature(self):
        history = make_history(10, base_demand=500.0)
        model = EventAwareForecastModel()
        model.fit(history)
        assert model.is_fitted()

        context = {"is_event": True, "event_multiplier": 1.25}
        preds = model.predict(history, horizon=2, context=context)
        assert len(preds) == 2
        assert preds[0] == pytest.approx(history[-1].demand_mw * 1.25)

    def test_refuses_without_event_feature(self):
        history = make_history(10)
        model = EventAwareForecastModel()
        model.fit(history)
        assert model.is_fitted()

        # Context without explicit event feature
        context = {"is_event": False}
        with pytest.raises(ValueError, match="requires an explicit event"):
            model.predict(history, horizon=1, context=context)


class TestLSTMForecastModelStub:
    def test_stub_properties_and_safety(self):
        stub = LSTMForecastModelStub()
        assert stub.model_name == "lstm"
        assert not stub.is_fitted()
        assert stub.status() == ModelStatus.WARMING_UP
        meta = stub.metadata()
        assert meta["is_extension_stub"] is True

        history = make_history(10)
        stub.fit(history)
        assert not stub.is_fitted()

        with pytest.raises(NotImplementedError, match="extension point stub"):
            stub.predict(history, horizon=1)
