"""
Streaming Interface for Sequential Electricity-Demand Processing.

Ensures strict online evaluation (one observation at a time, t_0, t_1, ..., t_n)
with zero lookahead or data leakage.
"""

from abc import ABC, abstractmethod
from typing import Generator, Iterable, List, Optional
from aeris.ingestion.schemas import DemandObservation
from aeris.ingestion.validator import StreamValidator
from aeris.utils.logger import default_logger


class BaseStream(ABC):
    """
    Abstract Base Class for AERIS Data Streams.
    
    Subclasses can adapt local synthetic arrays, CSV files, or live network APIs (e.g. Organizer API).
    """
    
    def __init__(self, validator: Optional[StreamValidator] = None):
        self.validator = validator or StreamValidator()
        self.processed_count: int = 0

    @abstractmethod
    def read_next(self) -> Optional[DemandObservation]:
        """
        Fetch the next chronological observation.
        Returns None when the stream ends.
        """
        pass

    def stream(self) -> Generator[DemandObservation, None, None]:
        """
        Generator yielding validated observations strictly sequentially.
        """
        while True:
            obs = self.read_next()
            if obs is None:
                break
            
            # Validate observation before yielding to downstream consumers
            try:
                self.validator.validate_observation(obs)
            except Exception as e:
                default_logger.log_validation_error(obs.timestamp, obs.demand_mw, str(e))
                raise e
            
            self.processed_count += 1
            default_logger.log_observation_received(obs.timestamp, obs.demand_mw, self.processed_count)
            yield obs

    def __iter__(self) -> Generator[DemandObservation, None, None]:
        return self.stream()


class ListStreamAdapter(BaseStream):
    """
    Adapts an in-memory list or iterable of DemandObservation objects or dicts into a sequential stream.
    """
    
    def __init__(self, raw_data: Iterable[DemandObservation], validator: Optional[StreamValidator] = None):
        super().__init__(validator=validator)
        self._iterator = iter(raw_data)

    def read_next(self) -> Optional[DemandObservation]:
        try:
            return next(self._iterator)
        except StopIteration:
            return None
