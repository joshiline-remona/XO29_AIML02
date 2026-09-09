# AERIS - Adaptive Electricity-Demand Forecasting System

AERIS is an adaptive electricity-demand forecasting system designed for dynamic power grid environments. AERIS follows a 4-stage sequential architecture:

$$\text{DETECT (Module 1)} \longrightarrow \text{UNDERSTAND (Module 2)} \longrightarrow \text{ROUTE (Module 3)} \longrightarrow \text{QUANTIFY (Module 4)}$$

---

## Module 1: Data Ingestion & Bayesian Change-Point Detection (DETECT Stage)

Module 1 receives time-stamped electricity-demand observations sequentially in real time and determines whether the underlying demand distribution has undergone a structural regime shift.

### Key Components:
- **Streaming Ingestion & Validator**: Enforces non-negative, finite demand values and strict monotonic timestamp ordering ($t_0 < t_1 < \dots < t_n$) to prevent chronological data leakage.
- **Bayesian Online Change-Point Detection (BOCPD)**: Implements Adams & MacKay (2007) with Normal-Inverse-Gamma (NIG) conjugate prior. Maintains posterior probabilities $P(r_t | x_{1:t})$ over run lengths $r_t$ in log-space with $O(1)$ pruning.

---

## Module 2: Contextual Classification & Regime Detection (UNDERSTAND Stage)

Module 2 consumes the Bayesian change-point signals from Module 1, online engineered demand features, and optional calendar/event context to classify live electricity demand behavior into 4 operational regimes:

1. **`NORMAL_DEMAND`**: Regular electricity demand behavior operating within expected baseline variance.
2. **`PEAK_SHOCK`**: A sudden, unusually large demand increase or extreme short-term deviation (e.g., equipment trips or sudden load spikes).
3. **`STRUCTURAL_SHIFT`**: A persistent baseline change in grid demand (e.g., industrial ramp-ups or permanent load shifts from 500 MW to 600 MW).
4. **`FESTIVAL_EVENT`**: Demand patterns associated with known/repeated calendar features or holiday events (e.g. Diwali).
5. **`WARMING_UP`**: Initial observations state before sufficient historical rolling window data is collected.

---

## How Module 1 Feeds Module 2

Module 2 subscribes directly to Module 1 outputs (`DetectionResult`):
- `change_point_prob`: Posterior probability $P(r_t=0 | x_{1:t})$ from BOCPD.
- `map_run_length`: Maximum A Posteriori steps elapsed since the last detected change point.
- `change_detected`: Boolean signal confirming a persistent structural shift.
- `detector_status`: High-level status (`WARMUP`, `STABLE`, `ANOMALY_SUSPECTED`, `CHANGE_CONFIRMED`).
- `posterior_mean` & `posterior_std`: Bayesian estimated parameters of the active regime.

---

## Online Feature Engineering & Anti-Data-Leakage

Module 2 computes features strictly online using only observations up to timestamp $t$ ($x_{\le t}$):

- **`rolling_mean`**: Moving baseline demand over rolling window $W$.
- **`rolling_std`**: Moving demand variance over rolling window $W$.
- **`z_score`**: Standardized deviation $(x_t - \text{rolling\_mean}) / \text{rolling\_std}$.
- **`volatility`**: Coefficient of variation ($\text{rolling\_std} / \text{rolling\_mean}$).
- **`trend_slope`**: Short-term linear regression slope over recent window.
- **`hour_of_day`, `day_of_week`, `is_weekend`**: Extracted dynamically from ISO timestamps.
- **`is_event`, `event_name`**: Optional contextual metadata flags.

---

## Distinguishing Isolated Spikes vs Structural Shifts

A single abnormal observation does **NOT** automatically become a `STRUCTURAL_SHIFT`.

- **Isolated Extreme Spike**:
  When a single spike arrives (e.g. 900 MW when baseline is 500 MW), Module 1 flags `ANOMALY_SUSPECTED` and Module 2 classifies it as `PEAK_SHOCK` with reason *"Peak shock detected from demand behaviour (Z-score = X.XX)"*. When demand reverts back to baseline at step $t+1$, the regime run-length history is preserved, preventing false structural adaptations.
  
- **Persistent Baseline Shift**:
  When demand shifts permanently (e.g. 500 MW to 600 MW), Module 1 verifies that the new regime hypothesis survives with high probability and bounded variance across consecutive steps. Upon confirmation (`change_detected = True`), Module 2 transitions the classification to `STRUCTURAL_SHIFT`.

---

## Regime Classification Output Schema

Each observation yields a `RegimeClassificationResult`:

```json
{
  "timestamp": "2026-01-01T12:30:00+00:00",
  "regime": "STRUCTURAL_SHIFT",
  "confidence": 0.95,
  "reason": "Persistent demand baseline shift confirmed by Bayesian change detector",
  "change_probability": 0.9132,
  "supporting_features": {
    "demand_mw": 602.4,
    "z_score": 2.85,
    "rolling_mean": 552.1,
    "rolling_std": 17.6,
    "hour_of_day": 12,
    "is_weekend": false
  },
  "sequence_idx": 35
}
```

---

## Limitations

1. **Weather/Context Dependencies**: Without explicit weather or temperature metadata inputs, Module 2 does not guess physical causes (e.g., heatwaves). It reports facts derived purely from demand behavior (*"Peak shock detected from demand behaviour"*).
2. **Gradual Ramp Delays**: Slow linear ramps accumulate variance gradually over multiple steps before triggering a sharp structural shift classification.

---

## Installation & Test Execution

### Installation:
```bash
pip install -r requirements.txt
```

### Run All Tests (Module 1 + Module 2):
```bash
pytest -v
```

### Run Module 1 & Module 2 Demonstrations:
```bash
# Module 1 (BOCPD Detection)
python demo.py

# Module 2 (End-to-End Ingestion -> Detection -> Regime Classification)
python demo_module2.py
```
