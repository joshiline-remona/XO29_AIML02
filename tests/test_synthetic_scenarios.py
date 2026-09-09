"""
Integration tests for the 4 hackathon synthetic scenarios.
"""

from aeris.config import BOCPDConfig
from aeris.detection.bayesian_change_point import BOCPDDetector
from aeris.detection.models import DetectorStatus
from aeris.ingestion.stream import ListStreamAdapter
from aeris.ingestion.synthetic import SyntheticStreamGenerator


def test_scenario_1_stable_noise():
    """
    TEST 1: Stable electricity demand with normal random noise.
    EXPECTED: Low change-point probability and 0 false persistent change detections.
    """
    observations = SyntheticStreamGenerator.generate_scenario_1_stable(
        n_steps=100, baseline_mw=500.0, noise_std=10.0, seed=42
    )
    
    config = BOCPDConfig(warmup_steps=10, change_threshold=0.5)
    detector = BOCPDDetector(config=config)
    stream = ListStreamAdapter(observations)
    
    results = [detector.process_observation(obs) for obs in stream.stream()]
    
    # Check that after warmup, no persistent structural change was detected
    post_warmup_results = results[10:]
    confirmed_changes = [r for r in post_warmup_results if r.change_detected]
    assert len(confirmed_changes) == 0, f"Expected 0 false change detections, got {len(confirmed_changes)}"
    
    # Average change point probability should remain low
    avg_change_prob = sum(r.change_point_prob for r in post_warmup_results) / len(post_warmup_results)
    assert avg_change_prob < 0.25, f"Expected low average change prob, got {avg_change_prob}"


def test_scenario_2_isolated_spike():
    """
    TEST 2: One isolated extreme spike.
    EXPECTED: The detector may temporarily bump probability/score at the spike step,
    but should NOT permanently classify the system as a new regime or trigger structural change.
    """
    spike_idx = 45
    observations = SyntheticStreamGenerator.generate_scenario_2_isolated_spike(
        n_steps=100, baseline_mw=500.0, noise_std=10.0, spike_idx=spike_idx, spike_magnitude_mw=900.0, seed=42
    )
    
    config = BOCPDConfig(warmup_steps=10, min_confirm_steps=2)
    detector = BOCPDDetector(config=config)
    stream = ListStreamAdapter(observations)
    
    results = [detector.process_observation(obs) for obs in stream.stream()]
    
    # Check spike step
    spike_res = results[spike_idx]
    # Spike should NOT trigger persistent change confirmed (because next step reverts to normal)
    assert spike_res.change_detected is False, "Isolated spike should NOT confirm a persistent structural change"
    
    # Post-spike observation (step 46) should revert to low reset probability or stable status
    post_spike_res = results[spike_idx + 1]
    assert post_spike_res.change_detected is False
    
    # Total persistent change detections in entire run should be 0
    confirmed_changes = [r for r in results if r.change_detected]
    assert len(confirmed_changes) == 0, "Isolated spike caused false structural change confirmation!"


def test_scenario_3_persistent_shift():
    """
    TEST 3: Persistent baseline shift (500 MW -> 600 MW).
    EXPECTED: The Bayesian detector produces a strong change-point signal and confirms regime shift.
    """
    change_idx = 50
    observations = SyntheticStreamGenerator.generate_scenario_3_persistent_shift(
        n_steps=120, baseline_1_mw=500.0, baseline_2_mw=600.0, change_idx=change_idx, noise_std=10.0, seed=42
    )
    
    config = BOCPDConfig(warmup_steps=10, change_threshold=0.5, min_confirm_steps=2)
    detector = BOCPDDetector(config=config)
    stream = ListStreamAdapter(observations)
    
    results = [detector.process_observation(obs) for obs in stream.stream()]
    
    # Verify change point detection occurs shortly after step 50
    post_shift_results = results[change_idx:change_idx + 10]
    confirmed_changes = [r for r in post_shift_results if r.change_detected]
    
    assert len(confirmed_changes) > 0, "Detector failed to detect persistent baseline shift!"
    
    # Verify detected regime mean adapts toward 600 MW
    final_res = results[-1]
    assert 570.0 <= final_res.posterior_mean <= 630.0, f"Posterior mean should adapt near 600, got {final_res.posterior_mean}"


def test_scenario_4_gradual_change():
    """
    TEST 4: Gradual demand ramp (500 MW -> 600 MW across 40 steps).
    EXPECTED: Detector handles gradual ramp without breaking. Documents behavior (gradual shift causes lower peak reset prob per step than abrupt step change).
    """
    observations = SyntheticStreamGenerator.generate_scenario_4_gradual_change(
        n_steps=120, start_mw=500.0, end_mw=600.0, ramp_start_idx=40, ramp_end_idx=80, noise_std=10.0, seed=42
    )
    
    config = BOCPDConfig(warmup_steps=10)
    detector = BOCPDDetector(config=config)
    stream = ListStreamAdapter(observations)
    
    results = [detector.process_observation(obs) for obs in stream.stream()]
    
    # Verify processing completed for all steps
    assert len(results) == 120
    # Final posterior mean should adjust to higher demand level (~600 MW)
    final_res = results[-1]
    assert final_res.posterior_mean > 550.0
