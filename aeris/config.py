"""
Configuration parameters for AERIS Module 1 & Module 2.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationConfig:
    """Configuration for streaming data validation."""
    allow_future_timestamps: bool = False
    min_demand_mw: float = 0.0
    max_demand_mw: float = 100_000.0
    strict_chronological: bool = True


@dataclass
class BOCPDConfig:
    """
    Configuration for Bayesian Online Change-Point Detection (BOCPD).
    
    Attributes:
        lambda_hazard: Expected segment length in time steps (hazard rate H = 1 / lambda_hazard).
        mu_0: Prior mean of electricity demand (MW).
        kappa_0: Prior precision weight for mean.
        alpha_0: Prior shape parameter for Inverse-Gamma variance model.
        beta_0: Prior scale parameter for Inverse-Gamma variance model.
        change_threshold: Probability threshold P(reset) to flag change suspect/confirmed.
        anomaly_threshold: Probability threshold to flag temporary anomaly.
        min_confirm_steps: Consecutive high reset probability steps to confirm persistent structural change.
        prune_threshold: Probability floor below which run length hypotheses are pruned (maintains O(1) performance).
        warmup_steps: Number of initial steps before change detection triggers are enabled.
    """
    lambda_hazard: float = 100.0
    mu_0: float = 500.0
    kappa_0: float = 1.0
    alpha_0: float = 1.0
    beta_0: float = 100.0
    change_threshold: float = 0.5
    anomaly_threshold: float = 0.25
    min_confirm_steps: int = 2
    prune_threshold: float = 1e-5
    warmup_steps: int = 10


@dataclass
class RegimeConfig:
    """
    Configuration for Module 2 Contextual Regime Classifier.
    
    Attributes:
        rolling_window: Historical window size for online rolling statistics.
        peak_shock_z_threshold: Z-score threshold to flag a peak demand shock.
        change_prob_threshold: Minimum change-point probability to signal structural shift.
        warmup_steps: Minimum historical observations before confidence classification starts.
    """
    rolling_window: int = 24
    peak_shock_z_threshold: float = 2.5
    change_prob_threshold: float = 0.5
    warmup_steps: int = 10


@dataclass
class LoggerConfig:
    """Structured logging configuration."""
    level: str = "INFO"
    json_format: bool = True
    log_to_stdout: bool = True
    log_file: Optional[str] = None
