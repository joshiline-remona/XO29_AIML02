"""
Bayesian Online Change-Point Detection (BOCPD) for AERIS.

Implements online Bayesian recursive change-point detection (Adams & MacKay, 2007)
using a Normal-Inverse-Gamma (NIG) conjugate observation model.

Operates strictly online for timestamp t using only observations x_{1:t}.
Distinguishes isolated temporary anomalies from persistent structural demand shifts.
"""

from typing import Optional
import numpy as np
from scipy.stats import t as student_t

from aeris.config import BOCPDConfig
from aeris.detection.models import DetectionResult, DetectorStatus
from aeris.ingestion.schemas import DemandObservation
from aeris.utils.logger import default_logger


class BOCPDDetector:
    """
    Bayesian Online Change-Point Detector using Normal-Inverse-Gamma Conjugate Prior.
    
    Maintains a posterior distribution over run lengths R_t(r) = P(r_t = r | x_{1:t}).
    For each observation x_t:
      1. Evaluates predictive probability P(x_t | r_{t-1}, x^{(r_{t-1})}) via Student-t distribution.
      2. Updates recursive message passing for growth and reset probabilities.
      3. Prunes hypotheses with probability < prune_threshold for O(1) step time.
      4. Evaluates whether a high reset probability represents a transient anomaly or structural change.
    """
    
    def __init__(self, config: Optional[BOCPDConfig] = None):
        self.config = config or BOCPDConfig()
        self.reset_detector()

    def reset_detector(self) -> None:
        """Reset internal Bayesian state and historical run lengths."""
        # Active run length integers r
        self.run_lengths: np.ndarray = np.array([0], dtype=np.int64)
        self.R: np.ndarray = np.array([1.0], dtype=np.float64)  # P(r_0 = 0) = 1
        
        # Conjugate parameters for each active run length r:
        # mu: prior mean, kappa: mean precision weight, alpha: IG shape, beta: IG scale
        self.mu: np.ndarray = np.array([self.config.mu_0], dtype=np.float64)
        self.kappa: np.ndarray = np.array([self.config.kappa_0], dtype=np.float64)
        self.alpha: np.ndarray = np.array([self.config.alpha_0], dtype=np.float64)
        self.beta: np.ndarray = np.array([self.config.beta_0], dtype=np.float64)
        
        self.step_count: int = 0
        self.pending_change_step: Optional[int] = None
        self.last_confirmed_change_step: Optional[int] = None

    def _predictive_logpdf(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> np.ndarray:
        """
        Compute log predictive probability log P(x_t | r, parameters_r) for run length hypotheses.
        Under a Normal-Inverse-Gamma prior, the predictive distribution is Student-t.
        """
        df = 2.0 * alpha
        scale = np.sqrt((beta * (kappa + 1.0)) / (alpha * kappa))
        return student_t.logpdf(x, df=df, loc=mu, scale=scale)

    def process_observation(self, observation: DemandObservation) -> DetectionResult:
        """
        Process a single streaming observation x_t sequentially.
        Strictly online: uses only observation x_t and prior state R_{t-1}.
        """
        x_t = float(observation.demand_mw)
        self.step_count += 1
        
        # Hazard rate H(r) = 1 / lambda_hazard
        hazard = 1.0 / self.config.lambda_hazard
        log_hazard = np.log(hazard)
        log_1_minus_hazard = np.log(1.0 - hazard)
        
        # 1. Compute log predictive probability under reset prior (r = 0)
        log_pred_reset = float(
            self._predictive_logpdf(
                x_t,
                np.array([self.config.mu_0]),
                np.array([self.config.kappa_0]),
                np.array([self.config.alpha_0]),
                np.array([self.config.beta_0])
            )[0]
        )
        
        # 2. Compute log predictive probability under existing active run-length hypotheses (r > 0)
        log_preds_growth = self._predictive_logpdf(x_t, self.mu, self.kappa, self.alpha, self.beta)
        
        # 3. Recursive message passing in log-space
        log_growth = np.log(np.maximum(self.R, 1e-300)) + log_preds_growth + log_1_minus_hazard
        log_reset = log_pred_reset + log_hazard
        
        # Combined run lengths array: index 0 is r=0, indices 1:N are r_prev + 1
        new_run_lengths = np.empty(len(self.run_lengths) + 1, dtype=np.int64)
        new_run_lengths[0] = 0
        new_run_lengths[1:] = self.run_lengths + 1
        
        # Unnormalized posterior array
        log_unnorm = np.empty(len(self.R) + 1, dtype=np.float64)
        log_unnorm[0] = log_reset
        log_unnorm[1:] = log_growth
        
        # Log-Sum-Exp normalization for numerical stability
        max_log = np.max(log_unnorm)
        unnorm = np.exp(log_unnorm - max_log)
        evidence = np.sum(unnorm)
        
        if evidence > 0:
            new_R = unnorm / evidence
        else:
            new_R = np.full_like(unnorm, 1.0 / len(unnorm))
            
        change_point_prob = float(new_R[0])
        
        # 4. Update conjugate parameters for updated run length hypotheses
        new_mu = np.empty(len(new_R), dtype=np.float64)
        new_kappa = np.empty(len(new_R), dtype=np.float64)
        new_alpha = np.empty(len(new_R), dtype=np.float64)
        new_beta = np.empty(len(new_R), dtype=np.float64)
        
        # Hypothesis r=0 (reset): fresh prior updated with current observation x_t
        new_kappa[0] = self.config.kappa_0 + 1.0
        new_mu[0] = (self.config.kappa_0 * self.config.mu_0 + x_t) / new_kappa[0]
        new_alpha[0] = self.config.alpha_0 + 0.5
        new_beta[0] = self.config.beta_0 + (self.config.kappa_0 * (x_t - self.config.mu_0) ** 2) / (2.0 * new_kappa[0])
        
        # Hypotheses r > 0 (growth from previous r - 1)
        new_kappa[1:] = self.kappa + 1.0
        new_mu[1:] = (self.kappa * self.mu + x_t) / new_kappa[1:]
        new_alpha[1:] = self.alpha + 0.5
        new_beta[1:] = self.beta + (self.kappa * (x_t - self.mu) ** 2) / (2.0 * new_kappa[1:])
        
        # 5. Prune low-probability run length hypotheses for bounded O(1) step time
        keep_mask = new_R >= self.config.prune_threshold
        max_prob_idx = int(np.argmax(new_R))
        keep_mask[0] = True
        keep_mask[max_prob_idx] = True
        
        self.run_lengths = new_run_lengths[keep_mask]
        self.R = new_R[keep_mask]
        self.R /= np.sum(self.R)  # Re-normalize after pruning
        
        self.mu = new_mu[keep_mask]
        self.kappa = new_kappa[keep_mask]
        self.alpha = new_alpha[keep_mask]
        self.beta = new_beta[keep_mask]
        
        # 6. Extract MAP run length integer and posterior distribution statistics
        max_idx = int(np.argmax(self.R))
        map_run_length = int(self.run_lengths[max_idx])
        posterior_mean = float(self.mu[max_idx])
        var_scale = self.beta[max_idx] / max(self.alpha[max_idx] - 1.0, 0.5)
        posterior_std = float(np.sqrt(var_scale))
        
        # 7. Status determination & persistent change vs isolated anomaly logic
        is_warmup = self.step_count <= self.config.warmup_steps
        change_detected = False
        
        if is_warmup:
            status = DetectorStatus.WARMUP
            self.pending_change_step = None
        else:
            # Step A: Check if a new regime hypothesis was triggered at this step (r=0)
            if change_point_prob >= self.config.change_threshold:
                self.pending_change_step = self.step_count
                status = DetectorStatus.ANOMALY_SUSPECTED
                default_logger.log_anomaly_detected(observation.timestamp, x_t, change_point_prob)
            
            # Step B: Check if a pending regime hypothesis from previous step survived as MAP
            elif self.pending_change_step is not None:
                steps_since_pending = self.step_count - self.pending_change_step
                
                # A structural regime shift is confirmed if the MAP hypothesis matches the pending regime
                # AND the posterior std is bounded (< 80.0 MW), ruling out extreme single-point spikes.
                if map_run_length == steps_since_pending and float(self.R[max_idx]) >= self.config.change_threshold and posterior_std <= 80.0:
                    change_detected = True
                    status = DetectorStatus.CHANGE_CONFIRMED
                    self.last_confirmed_change_step = self.pending_change_step
                    self.pending_change_step = None  # Clear pending status after confirmation
                    default_logger.log_change_detected(
                        observation.timestamp,
                        x_t,
                        float(self.R[max_idx]),
                        posterior_mean,
                        posterior_std
                    )
                else:
                    # Survival failed or extreme spike variance detected
                    self.pending_change_step = None
                    status = DetectorStatus.STABLE
            else:
                status = DetectorStatus.STABLE
                
        default_logger.log_change_probability_updated(
            observation.timestamp,
            x_t,
            change_point_prob,
            map_run_length,
            status.value
        )

        return DetectionResult(
            timestamp=observation.timestamp,
            observed_demand=x_t,
            change_point_prob=change_point_prob,
            map_run_length=map_run_length,
            change_detected=change_detected,
            detector_status=status,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            sequence_idx=observation.sequence_idx,
            metadata=observation.metadata,
        )
