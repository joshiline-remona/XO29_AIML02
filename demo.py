"""
AERIS Module 1 Demonstration Script.

Runs sequential streaming ingestion and Bayesian Online Change-Point Detection (BOCPD)
across the 4 hackathon benchmark scenarios:
  1. Stable Noise
  2. Isolated Extreme Spike
  3. Persistent Baseline Shift (500 MW -> 600 MW)
  4. Gradual Ramp Change (500 MW -> 600 MW over 40 steps)
"""

import sys
from typing import List
from aeris.config import BOCPDConfig
from aeris.detection.bayesian_change_point import BOCPDDetector
from aeris.detection.models import DetectionResult
from aeris.ingestion.stream import ListStreamAdapter
from aeris.ingestion.synthetic import SyntheticStreamGenerator


def run_scenario_demo(scenario_name: str, observations) -> List[DetectionResult]:
    print(f"\n============================================================")
    print(f"   RUNNING DEMO: {scenario_name}")
    print(f"============================================================")
    
    config = BOCPDConfig(warmup_steps=10, change_threshold=0.5, min_confirm_steps=2)
    detector = BOCPDDetector(config=config)
    stream = ListStreamAdapter(observations)
    
    results: List[DetectionResult] = []
    print(f"{'Step':<6} | {'Timestamp':<20} | {'Demand (MW)':<11} | {'P(Change)':<10} | {'MAP Run':<8} | {'Status':<18} | {'Regime Mean':<11}")
    print("-" * 100)
    
    for obs in stream.stream():
        res = detector.process_observation(obs)
        results.append(res)
        
        # Print selected rows or interesting events
        is_highlight = (
            res.change_detected or 
            res.detector_status.value == "ANOMALY_SUSPECTED" or 
            (res.sequence_idx % 20 == 0) or
            obs.metadata.get("is_spike", False) or
            obs.metadata.get("true_change_idx", -1) == res.sequence_idx
        )
        
        if is_highlight:
            print(
                f"{res.sequence_idx:<6} | "
                f"{res.timestamp[:19]:<20} | "
                f"{res.observed_demand:<11.2f} | "
                f"{res.change_point_prob:<10.4f} | "
                f"{res.map_run_length:<8} | "
                f"{res.detector_status.value:<18} | "
                f"{res.posterior_mean:<11.2f}"
            )
            
    confirmed = [r for r in results if r.change_detected]
    anomalies = [r for r in results if r.detector_status.value == "ANOMALY_SUSPECTED"]
    
    print("-" * 100)
    print(f"Summary for {scenario_name}:")
    print(f"  - Total Observations Processed: {len(results)}")
    print(f"  - Isolated Anomalies Flagged:  {len(anomalies)}")
    print(f"  - Persistent Regime Shifts:    {len(confirmed)}")
    if confirmed:
        first_c = confirmed[0]
        print(f"  - First Shift Confirmed at Step: {first_c.sequence_idx} (Timestamp: {first_c.timestamp})")
    print(f"============================================================\n")
    return results


def main():
    print("[AERIS] Starting AERIS Module 1 (DETECT Stage) Streaming Demonstration...")
    
    # 1. Scenario 1: Stable noise
    s1_obs = SyntheticStreamGenerator.generate_scenario_1_stable(n_steps=60)
    run_scenario_demo("TEST 1: Stable Electricity Demand (Noise Only)", s1_obs)
    
    # 2. Scenario 2: Isolated Spike
    s2_obs = SyntheticStreamGenerator.generate_scenario_2_isolated_spike(n_steps=60, spike_idx=30)
    run_scenario_demo("TEST 2: Isolated Extreme Demand Spike", s2_obs)
    
    # 3. Scenario 3: Persistent Shift
    s3_obs = SyntheticStreamGenerator.generate_scenario_3_persistent_shift(n_steps=80, change_idx=35)
    run_scenario_demo("TEST 3: Persistent Baseline Shift (500 MW -> 600 MW)", s3_obs)
    
    # 4. Scenario 4: Gradual Change
    s4_obs = SyntheticStreamGenerator.generate_scenario_4_gradual_change(n_steps=80, ramp_start_idx=25, ramp_end_idx=55)
    run_scenario_demo("TEST 4: Gradual Demand Ramp (500 MW -> 600 MW)", s4_obs)
    
    print("[SUCCESS] AERIS Module 1 Demonstration Completed Successfully!")


if __name__ == "__main__":
    main()
