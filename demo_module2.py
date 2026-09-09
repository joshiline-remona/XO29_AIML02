"""
AERIS Module 2 Demonstration Script.

Runs end-to-end sequential processing pipeline:
  Demand Observation -> Ingestion -> Module 1 (BOCPD Detector) -> Module 2 (Regime Classifier)

Demonstrates classification across 5 benchmark scenarios:
  1. Stable Electricity Demand -> NORMAL_DEMAND
  2. Isolated Extreme Spike -> PEAK_SHOCK (Not STRUCTURAL_SHIFT)
  3. Persistent Baseline Shift (500 MW -> 600 MW) -> STRUCTURAL_SHIFT
  4. Gradual Demand Ramp -> STABLE / GRADUAL
  5. Festival / Calendar Event -> FESTIVAL_EVENT
"""

from typing import List
from aeris.config import BOCPDConfig, RegimeConfig
from aeris.detection.bayesian_change_point import BOCPDDetector
from aeris.ingestion.stream import ListStreamAdapter
from aeris.ingestion.synthetic import SyntheticStreamGenerator
from aeris.regime.classifier import RuleBasedRegimeClassifier
from aeris.regime.models import RegimeClassificationResult


def run_pipeline_demo(scenario_name: str, observations) -> List[RegimeClassificationResult]:
    print(f"\n==========================================================================================")
    print(f"   RUNNING AERIS MODULE 1 -> MODULE 2 PIPELINE: {scenario_name}")
    print(f"==========================================================================================")
    
    bocpd = BOCPDDetector(config=BOCPDConfig(warmup_steps=10))
    classifier = RuleBasedRegimeClassifier(config=RegimeConfig(warmup_steps=10))
    stream = ListStreamAdapter(observations)
    
    results: List[RegimeClassificationResult] = []
    print(
        f"{'Step':<6} | {'Timestamp':<20} | {'Demand (MW)':<11} | {'P(Change)':<10} | {'Regime':<18} | {'Conf':<6} | {'Reason':<40}"
    )
    print("-" * 125)
    
    for obs in stream.stream():
        det_res = bocpd.process_observation(obs)
        reg_res = classifier.classify_regime(obs, det_res)
        results.append(reg_res)
        
        # Highlight interesting rows or periodic samples
        is_highlight = (
            reg_res.regime.value != "NORMAL_DEMAND" or
            (obs.sequence_idx % 20 == 0) or
            obs.metadata.get("is_spike", False) or
            obs.metadata.get("true_change_idx", -1) == obs.sequence_idx or
            obs.metadata.get("is_event", False)
        )
        
        if is_highlight:
            print(
                f"{obs.sequence_idx:<6} | "
                f"{obs.timestamp[:19]:<20} | "
                f"{obs.demand_mw:<11.2f} | "
                f"{reg_res.change_probability:<10.4f} | "
                f"{reg_res.regime.value:<18} | "
                f"{reg_res.confidence:<6.2f} | "
                f"{reg_res.reason:<40}"
            )
            
    regime_counts = {}
    for r in results:
        v = r.regime.value
        regime_counts[v] = regime_counts.get(v, 0) + 1
        
    print("-" * 125)
    print(f"Summary for {scenario_name}:")
    print(f"  - Total Observations Processed: {len(results)}")
    print(f"  - Regime Breakdown: {regime_counts}")
    print(f"==========================================================================================\n")
    return results


def main():
    print("[AERIS] Starting Module 1 -> Module 2 Sequential Pipeline Demonstration...")
    
    # 1. Scenario 1: Stable Normal Demand
    s1_obs = SyntheticStreamGenerator.generate_scenario_1_stable(n_steps=50)
    run_pipeline_demo("TEST 1: Stable Normal Demand", s1_obs)
    
    # 2. Scenario 2: Isolated Spike
    s2_obs = SyntheticStreamGenerator.generate_scenario_2_isolated_spike(n_steps=50, spike_idx=25)
    run_pipeline_demo("TEST 2: Isolated Extreme Demand Spike", s2_obs)
    
    # 3. Scenario 3: Persistent Baseline Shift
    s3_obs = SyntheticStreamGenerator.generate_scenario_3_persistent_shift(n_steps=60, change_idx=25)
    run_pipeline_demo("TEST 3: Persistent Baseline Shift (500 MW -> 600 MW)", s3_obs)
    
    # 4. Scenario 4: Gradual Demand Ramp
    s4_obs = SyntheticStreamGenerator.generate_scenario_4_gradual_change(n_steps=60, ramp_start_idx=20, ramp_end_idx=45)
    run_pipeline_demo("TEST 4: Gradual Demand Ramp (500 MW -> 600 MW)", s4_obs)
    
    # 5. Scenario 5: Festival / Calendar Event
    s5_obs = SyntheticStreamGenerator.generate_scenario_5_festival_event(n_steps=50, festival_start_idx=20, festival_duration=10)
    run_pipeline_demo("TEST 5: Festival / Calendar Event (Diwali)", s5_obs)
    
    print("[SUCCESS] AERIS Module 2 Pipeline Demonstration Completed Successfully!")


if __name__ == "__main__":
    main()
