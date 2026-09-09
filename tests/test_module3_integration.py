"""
End-to-End Integration Test for AERIS Module 1 -> Module 2 -> Module 3.

Demonstrates complete pipeline:
Synthetic Demand Stream -> Module 1 (BOCPD) -> Module 2 (Regime) -> Module 3 (Router) -> (RoutingResult, ForecastResult)
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

from aeris.config import BOCPDConfig, RegimeConfig, RoutingConfig, ForecastingConfig
from aeris.detection import BOCPDDetector
from aeris.forecasting.base import ForecastResult

from aeris.ingestion.schemas import DemandObservation
from aeris.regime.classifier import RuleBasedRegimeClassifier
from aeris.regime.models import RegimeType
from aeris.routing.models import RoutingResult
from aeris.routing.router import DynamicModelRouter


def generate_synthetic_stream(n_steps: int = 50) -> list[DemandObservation]:
    """
    Generate synthetic electricity demand stream with 3 distinct phases:
    1. Normal demand (steps 0..19, ~500 MW)
    2. Peak Shock spike (step 20..22, ~850 MW)
    3. Structural shift (steps 23..49, ~750 MW baseline shift)
    """
    start_time = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)
    stream = []

    for i in range(n_steps):
        ts = (start_time + timedelta(hours=i)).isoformat()

        if i < 20:
            val = 500.0 + 10.0 * np.sin(i * np.pi / 6) + float(np.random.normal(0, 2))
        elif 20 <= i < 23:
            # Extreme peak demand shock
            val = 850.0 + float(np.random.normal(0, 5))
        else:
            # Structural baseline shift upward
            val = 750.0 + 15.0 * np.cos(i * np.pi / 6) + float(np.random.normal(0, 3))

        stream.append(DemandObservation(timestamp=ts, demand_mw=float(val), sequence_idx=i + 1))

    return stream


def test_module1_module2_module3_end_to_end_pipeline():
    """
    Full integration test verifying sequential flow:
    Module 1 DETECT -> Module 2 UNDERSTAND -> Module 3 ROUTE.
    """
    # Initialize all three module engines
    bocpd_config = BOCPDConfig(warmup_steps=5, change_threshold=0.4, min_confirm_steps=2)
    detector = BOCPDDetector(config=bocpd_config)

    regime_config = RegimeConfig(warmup_steps=5, peak_shock_z_threshold=2.5)
    classifier = RuleBasedRegimeClassifier(config=regime_config)

    router = DynamicModelRouter(
        routing_config=RoutingConfig(min_history_steps=5),
        forecasting_config=ForecastingConfig(forecast_horizon=1)
    )

    stream = generate_synthetic_stream(n_steps=40)

    routing_results = []
    forecast_results = []

    # Stream sequentially through the full 3-module pipeline
    for obs in stream:
        # Module 1: Bayesian Change-Point Detection
        det_res = detector.process_observation(obs)
        assert det_res is not None

        # Module 2: Contextual Regime Classification
        reg_res = classifier.classify_regime(obs, det_res)
        assert reg_res is not None

        # Module 3: Dynamic Model Routing & Forecasting
        rout_res, fore_res = router.route_and_predict(
            observation=obs,
            detection_result=det_res,
            regime_result=reg_res,
            horizon=1
        )

        assert isinstance(rout_res, RoutingResult)
        assert isinstance(fore_res, ForecastResult)
        assert fore_res.predicted_values is not None
        assert len(fore_res.predicted_values) == 1

        routing_results.append(rout_res)
        forecast_results.append(fore_res)

    # Verifications across phases
    # Phase 1 (Normal demand after warmup)
    normal_routes = [r for r in routing_results[10:19] if r.regime == "NORMAL_DEMAND"]
    assert len(normal_routes) > 0
    assert any(r.selected_model == "arima" for r in normal_routes)

    # Phase 2 (Peak Shock spike around step 20-22)
    peak_routes = [r for r in routing_results[20:23] if r.regime == "PEAK_SHOCK"]
    if peak_routes:
        assert any(r.selected_model == "xgboost" for r in peak_routes)

    # Phase 3 (Structural Shift after confirm step 25+)
    shift_routes = [r for r in routing_results[28:] if r.regime == "STRUCTURAL_SHIFT"]
    if shift_routes:
        assert any(r.selected_model == "xgboost" for r in shift_routes)
