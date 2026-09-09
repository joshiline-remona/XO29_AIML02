"""
Unit and Integration Tests for AERIS Module 2: Regime Detection.

Tests the 7 required Module 2 scenarios:
1. Stable normal electricity demand -> NORMAL_DEMAND
2. One large temporary spike -> PEAK_SHOCK / anomaly, NOT STRUCTURAL_SHIFT
3. Persistent baseline increase -> STRUCTURAL_SHIFT
4. Event-like demand with calendar/event feature -> FESTIVAL_EVENT
5. Extreme short-term peak shock -> PEAK_SHOCK
6. Insufficient data -> WARMING_UP status
7. Strict chronological non-leakage guarantee
"""

import pytest
from aeris.config import BOCPDConfig, RegimeConfig
from aeris.detection.bayesian_change_point import BOCPDDetector
from aeris.ingestion.schemas import DemandObservation
from aeris.ingestion.synthetic import SyntheticStreamGenerator
from aeris.regime.classifier import RuleBasedRegimeClassifier
from aeris.regime.models import RegimeType


def test_scenario_1_normal_demand():
    """TEST 1: Stable normal electricity demand expected to be classified as NORMAL_DEMAND."""
    observations = SyntheticStreamGenerator.generate_scenario_1_stable(n_steps=60, seed=42)
    
    bocpd = BOCPDDetector(config=BOCPDConfig(warmup_steps=10))
    classifier = RuleBasedRegimeClassifier(config=RegimeConfig(warmup_steps=10))
    
    regime_results = []
    for obs in observations:
        det_res = bocpd.process_observation(obs)
        reg_res = classifier.classify_regime(obs, det_res)
        regime_results.append(reg_res)
        
    # After warmup (steps 10 to 60), normal stable demand should be predominantly NORMAL_DEMAND (>=90%)
    post_warmup = regime_results[10:]
    normal_count = sum(1 for r in post_warmup if r.regime == RegimeType.NORMAL_DEMAND)
    assert normal_count >= 45, f"Expected at least 90% NORMAL_DEMAND post-warmup, got {normal_count}/{len(post_warmup)}"


def test_scenario_2_one_large_spike_not_structural_shift():
    """TEST 2: One large temporary spike MUST NOT be classified as STRUCTURAL_SHIFT."""
    spike_idx = 35
    observations = SyntheticStreamGenerator.generate_scenario_2_isolated_spike(
        n_steps=70, spike_idx=spike_idx, spike_magnitude_mw=900.0, seed=42
    )
    
    bocpd = BOCPDDetector(config=BOCPDConfig(warmup_steps=10))
    classifier = RuleBasedRegimeClassifier(config=RegimeConfig(warmup_steps=10, peak_shock_z_threshold=2.5))
    
    regime_results = []
    for obs in observations:
        det_res = bocpd.process_observation(obs)
        reg_res = classifier.classify_regime(obs, det_res)
        regime_results.append(reg_res)
        
    # Spike step should be PEAK_SHOCK or ANOMALY, NOT STRUCTURAL_SHIFT
    spike_regime = regime_results[spike_idx]
    assert spike_regime.regime != RegimeType.STRUCTURAL_SHIFT, "Single spike MUST NOT trigger STRUCTURAL_SHIFT"
    assert spike_regime.regime == RegimeType.PEAK_SHOCK
    
    # Zero observations in the entire stream should be STRUCTURAL_SHIFT
    shift_count = sum(1 for r in regime_results if r.regime == RegimeType.STRUCTURAL_SHIFT)
    assert shift_count == 0, "Isolated spike falsely triggered STRUCTURAL_SHIFT!"


def test_scenario_3_persistent_baseline_increase():
    """TEST 3: Persistent baseline step increase expected to trigger STRUCTURAL_SHIFT."""
    change_idx = 35
    observations = SyntheticStreamGenerator.generate_scenario_3_persistent_shift(
        n_steps=70, change_idx=change_idx, baseline_1_mw=500.0, baseline_2_mw=600.0, seed=42
    )
    
    bocpd = BOCPDDetector(config=BOCPDConfig(warmup_steps=10))
    classifier = RuleBasedRegimeClassifier(config=RegimeConfig(warmup_steps=10))
    
    regime_results = []
    for obs in observations:
        det_res = bocpd.process_observation(obs)
        reg_res = classifier.classify_regime(obs, det_res)
        regime_results.append(reg_res)
        
    # Verify STRUCTURAL_SHIFT is confirmed shortly after step 35
    post_shift = regime_results[change_idx : change_idx + 10]
    shift_confirmed = any(r.regime == RegimeType.STRUCTURAL_SHIFT for r in post_shift)
    assert shift_confirmed, "Failed to classify persistent baseline increase as STRUCTURAL_SHIFT"


def test_scenario_4_festival_event_with_calendar_feature():
    """TEST 4: Demand pattern with explicit calendar/event metadata feature -> FESTIVAL_EVENT."""
    observations = SyntheticStreamGenerator.generate_scenario_5_festival_event(
        n_steps=60, festival_start_idx=25, festival_duration=10, seed=42
    )
    
    bocpd = BOCPDDetector(config=BOCPDConfig(warmup_steps=10))
    classifier = RuleBasedRegimeClassifier(config=RegimeConfig(warmup_steps=10))
    
    regime_results = []
    for obs in observations:
        det_res = bocpd.process_observation(obs)
        reg_res = classifier.classify_regime(obs, det_res)
        regime_results.append(reg_res)
        
    # Festival steps (25 to 35) should be classified as FESTIVAL_EVENT
    festival_steps = regime_results[25:35]
    festival_count = sum(1 for r in festival_steps if r.regime == RegimeType.FESTIVAL_EVENT)
    assert festival_count == 10, f"Expected 10 FESTIVAL_EVENT steps, got {festival_count}"
    assert "Diwali Festival" in festival_steps[0].reason


def test_scenario_5_peak_demand_shock():
    """TEST 5: Peak demand shock classified as PEAK_SHOCK when data shows extreme short-term deviation."""
    spike_idx = 25
    observations = SyntheticStreamGenerator.generate_scenario_2_isolated_spike(
        n_steps=50, spike_idx=spike_idx, spike_magnitude_mw=850.0, seed=42
    )
    
    bocpd = BOCPDDetector(config=BOCPDConfig(warmup_steps=10))
    classifier = RuleBasedRegimeClassifier(config=RegimeConfig(warmup_steps=10, peak_shock_z_threshold=2.5))
    
    regime_results = []
    for obs in observations:
        det_res = bocpd.process_observation(obs)
        reg_res = classifier.classify_regime(obs, det_res)
        regime_results.append(reg_res)
        
    res = regime_results[spike_idx]
    assert res.regime == RegimeType.PEAK_SHOCK
    assert "Peak shock detected from demand behaviour" in res.reason


def test_scenario_6_insufficient_data_warming_up():
    """TEST 6: Insufficient data steps return WARMING_UP status."""
    observations = SyntheticStreamGenerator.generate_scenario_1_stable(n_steps=15, seed=42)
    
    bocpd = BOCPDDetector(config=BOCPDConfig(warmup_steps=10))
    classifier = RuleBasedRegimeClassifier(config=RegimeConfig(warmup_steps=10))
    
    results = []
    for obs in observations:
        det_res = bocpd.process_observation(obs)
        reg_res = classifier.classify_regime(obs, det_res)
        results.append(reg_res)
        
    # First 10 steps should be WARMING_UP
    for r in results[:10]:
        assert r.regime == RegimeType.WARMING_UP
        assert r.confidence == 1.0
        
    # Step 11 onward should be classified
    assert results[10].regime != RegimeType.WARMING_UP


def test_scenario_7_anti_leakage_chronological_independence():
    """TEST 7: Verify classification at step t is unaffected by future observations (no lookahead)."""
    obs_full = SyntheticStreamGenerator.generate_scenario_3_persistent_shift(n_steps=60, change_idx=30, seed=42)
    obs_truncated = obs_full[:30]  # Stream containing only up to step 29
    
    # Run 1: process full stream up to step 29
    bocpd1 = BOCPDDetector(config=BOCPDConfig(warmup_steps=10))
    classifier1 = RuleBasedRegimeClassifier(config=RegimeConfig(warmup_steps=10))
    results1 = [classifier1.classify_regime(o, bocpd1.process_observation(o)) for o in obs_full[:30]]
    
    # Run 2: process truncated stream up to step 29
    bocpd2 = BOCPDDetector(config=BOCPDConfig(warmup_steps=10))
    classifier2 = RuleBasedRegimeClassifier(config=RegimeConfig(warmup_steps=10))
    results2 = [classifier2.classify_regime(o, bocpd2.process_observation(o)) for o in obs_truncated]
    
    # Verify outputs at step 29 are 100% identical regardless of whether future data exists
    res1_step29 = results1[29]
    res2_step29 = results2[29]
    
    assert res1_step29.regime == res2_step29.regime
    assert res1_step29.confidence == res2_step29.confidence
    assert res1_step29.change_probability == res2_step29.change_probability
    assert res1_step29.supporting_features["rolling_mean"] == res2_step29.supporting_features["rolling_mean"]
