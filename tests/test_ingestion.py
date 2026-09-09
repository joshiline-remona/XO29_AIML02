"""
Unit tests for AERIS Ingestion and Validation Layer.
"""

import pytest
from aeris.ingestion.schemas import DemandObservation
from aeris.ingestion.validator import StreamValidator, ValidationError, ChronologyError
from aeris.ingestion.stream import ListStreamAdapter


def test_valid_observation():
    validator = StreamValidator()
    obs = DemandObservation(timestamp="2026-01-01T00:00:00Z", demand_mw=500.0, sequence_idx=0)
    validator.validate_observation(obs)  # Should pass without error


def test_nan_demand_raises_validation_error():
    validator = StreamValidator()
    obs = DemandObservation(timestamp="2026-01-01T00:00:00Z", demand_mw=float("nan"), sequence_idx=0)
    with pytest.raises(ValidationError, match="finite"):
        validator.validate_observation(obs)


def test_out_of_bounds_demand_raises_validation_error():
    validator = StreamValidator()
    obs_negative = DemandObservation(timestamp="2026-01-01T00:00:00Z", demand_mw=-10.0, sequence_idx=0)
    with pytest.raises(ValidationError, match="outside permitted bounds"):
        validator.validate_observation(obs_negative)


def test_out_of_order_timestamp_raises_chronology_error():
    validator = StreamValidator()
    obs1 = DemandObservation(timestamp="2026-01-01T01:00:00Z", demand_mw=500.0, sequence_idx=0)
    obs2 = DemandObservation(timestamp="2026-01-01T00:30:00Z", demand_mw=505.0, sequence_idx=1)
    
    validator.validate_observation(obs1)
    with pytest.raises(ChronologyError, match="Chronological violation"):
        validator.validate_observation(obs2)


def test_out_of_order_sequence_index_raises_chronology_error():
    validator = StreamValidator()
    obs1 = DemandObservation(timestamp="2026-01-01T01:00:00Z", demand_mw=500.0, sequence_idx=5)
    obs2 = DemandObservation(timestamp="2026-01-01T02:00:00Z", demand_mw=505.0, sequence_idx=4)
    
    validator.validate_observation(obs1)
    with pytest.raises(ChronologyError, match="Sequence index violation"):
        validator.validate_observation(obs2)


def test_list_stream_adapter_sequential_iteration():
    raw_list = [
        DemandObservation(timestamp="2026-01-01T01:00:00Z", demand_mw=500.0, sequence_idx=0),
        DemandObservation(timestamp="2026-01-01T02:00:00Z", demand_mw=510.0, sequence_idx=1),
        DemandObservation(timestamp="2026-01-01T03:00:00Z", demand_mw=505.0, sequence_idx=2),
    ]
    stream_adapter = ListStreamAdapter(raw_list)
    results = list(stream_adapter.stream())
    
    assert len(results) == 3
    assert results[0].demand_mw == 500.0
    assert results[1].demand_mw == 510.0
    assert results[2].demand_mw == 505.0
    assert stream_adapter.processed_count == 3
