"""
AERIS Module 3 Demonstration: Live Streaming Ingestion, Bayesian Change-Point Detection,
Contextual Regime Classification, and Dynamic Forecast Model Routing.
"""

from datetime import datetime, timezone, timedelta
import random
import time

from aeris.config import BOCPDConfig, ForecastingConfig, RegimeConfig, RoutingConfig
from aeris.detection import BOCPDDetector
from aeris.ingestion.schemas import DemandObservation
from aeris.regime.classifier import RuleBasedRegimeClassifier
from aeris.routing.router import DynamicModelRouter


def run_demo():
    print("=" * 80)
    print(" AERIS: Adaptive Electricity-Demand Forecasting System")
    print(" MODULE 3 DEMO: Live Stream Ingestion -> DETECT -> UNDERSTAND -> ROUTE")
    print("=" * 80)

    detector = BOCPDDetector(config=BOCPDConfig(warmup_steps=5, change_threshold=0.4))
    classifier = RuleBasedRegimeClassifier(config=RegimeConfig(warmup_steps=5))
    router = DynamicModelRouter(
        routing_config=RoutingConfig(min_history_steps=5),
        forecasting_config=ForecastingConfig(forecast_horizon=1)
    )

    start_time = datetime.now(timezone.utc)
    base_demand = 500.0

    print(f"\nSimulating streaming electricity demand over 30 time steps...\n")

    for i in range(1, 31):
        ts = (start_time + timedelta(hours=i)).isoformat()

        # Simulate dynamic demand scenarios
        if i < 10:
            # Phase 1: Normal demand
            val = base_demand + random.gauss(0, 5)
            context = None
        elif 10 <= i < 13:
            # Phase 2: Sudden Peak Shock
            val = base_demand + 300.0 + random.gauss(0, 10)
            context = None
        elif 13 <= i < 22:
            # Phase 3: Persistent Structural Shift
            val = 720.0 + random.gauss(0, 8)
            context = None
        elif 22 <= i < 26:
            # Phase 4: Festival Event with explicit calendar indicator
            val = 850.0 + random.gauss(0, 10)
            context = {"is_event": True, "event_name": "Festival Peak", "event_multiplier": 1.2}
        else:
            # Phase 5: Return to stable normal demand
            val = 520.0 + random.gauss(0, 5)
            context = None

        obs = DemandObservation(timestamp=ts, demand_mw=round(val, 2), sequence_idx=i)

        # Pipeline execution: DETECT -> UNDERSTAND -> ROUTE
        det_res = detector.process_observation(obs)
        reg_res = classifier.classify_regime(obs, det_res, context_override=context)
        rout_res, fore_res = router.route_and_predict(obs, det_res, reg_res, horizon=1, context_override=context)

        pred_val = fore_res.predicted_values[0] if fore_res.predicted_values else 0.0

        print(
            f"Step {i:02d} | Demand: {obs.demand_mw:6.1f} MW | "
            f"BOCPD P(change): {det_res.change_point_prob:.3f} | "
            f"Regime: {reg_res.regime.value:<16} | "
            f"Selected Model: {rout_res.selected_model:<11} | "
            f"Forecast: {pred_val:6.1f} MW"
        )
        print(f"       -> Reason: {rout_res.reason}\n")
        time.sleep(0.05)

    print("=" * 80)
    print(" DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
