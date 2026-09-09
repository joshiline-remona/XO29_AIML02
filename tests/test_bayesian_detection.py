"""
Unit tests for Bayesian Online Change-Point Detector math and state behavior.
"""

import pytest
import numpy as np
from aeris.config import BOCPDConfig
from aeris.detection.bayesian_change_point import BOCPDDetector
from aeris.detection.models import DetectorStatus
from aeris.ingestion.schemas import DemandObservation


def test_bocpd_initialization_and_reset():
    detector = BOCPDDetector()
    assert detector.step_count == 0
    assert len(detector.R) == 1
    assert detector.R[0] == 1.0


def test_bocpd_warmup_phase():
    config = BOCPDConfig(warmup_steps=5)
    detector = BOCPDDetector(config=config)
    
    for i in range(5):
        obs = DemandObservation(timestamp=f"2026-01-01T0{i}:00:00Z", demand_mw=500.0, sequence_idx=i)
        res = detector.process_observation(obs)
        assert res.detector_status == DetectorStatus.WARMUP
        assert res.change_detected is False


def test_bocpd_posterior_probabilities_sum_to_one():
    detector = BOCPDDetector()
    for i in range(15):
        obs = DemandObservation(timestamp=f"2026-01-01T{i:02d}:00:00Z", demand_mw=500.0 + (i % 3), sequence_idx=i)
        detector.process_observation(obs)
        assert np.isclose(np.sum(detector.R), 1.0)


def test_bocpd_pruning_bounds_state_size():
    config = BOCPDConfig(prune_threshold=1e-3)
    detector = BOCPDDetector(config=config)
    
    for i in range(100):
        obs = DemandObservation(timestamp=f"2026-01-01T{i//24:02d}:{i%24:02d}:00Z", demand_mw=500.0, sequence_idx=i)
        detector.process_observation(obs)
        # Verify active run lengths stay small (O(1))
        assert len(detector.R) <= 50
