# AERIS - Adaptive Electricity-Demand Forecasting System
## Module 1: Online Data Ingestion & Bayesian Change-Point Detection (DETECT Stage)

AERIS is an adaptive electricity-demand forecasting system designed for dynamic power grid environments. AERIS follows a 4-stage pipeline:

$$\text{DETECT} \longrightarrow \text{UNDERSTAND} \longrightarrow \text{ROUTE} \longrightarrow \text{QUANTIFY}$$

**Module 1 implements the DETECT stage.**

---

## 1. What Module 1 Does

The DETECT stage receives time-stamped electricity-demand observations sequentially in real time and determines whether the underlying demand distribution has undergone a structural regime shift.

Specifically, Module 1 distinguishes between:
1. **Normal random noise / fluctuations**: Standard operational variance around a baseline.
2. **Isolated temporary anomalies**: Single extreme spikes (e.g. sensor glitch, trip event) that revert immediately to baseline.
3. **Persistent structural changes**: Step changes in grid baselines (e.g., industrial ramp-up, structural load shifts from 500 MW to 600 MW).

---

## 2. Why Bayesian Change-Point Detection (BOCPD)?

Traditional threshold-based rules or moving averages fail in streaming time-series:
- Moving average windows lag severely and blur change boundaries.
- Static thresholding triggers false alarms on single extreme spikes.
- Offline change-point algorithms require future data lookahead (violating real-time constraints).

**Bayesian Online Change-Point Detection (BOCPD)** (Adams & MacKay, 2007) solves these challenges by maintaining an exact recursive posterior probability distribution over the **run length** $r_t$ (the number of time steps elapsed since the last change-point). 

### Key Mathematical Formulations:
- **Run Length State**: $r_t \in \{0, 1, 2, \dots\}$.
- **Hazard Function**: $H(r_t) = \frac{1}{\lambda}$ (constant prior hazard rate over expected segment duration $\lambda$).
- **Conjugate Model**: Normal-Inverse-Gamma (NIG) prior over unknown demand mean $\mu$ and unknown variance $\sigma^2$.
- **Predictive Probability**: $P(x_t | r_{t-1}, x^{(r_{t-1})})$ follows a Student-t distribution evaluated efficiently in log space.
- **Recursive Bayesian Updating**:
  - Growth probability ($r \to r+1$): $\tilde{R}_t(r+1) = R_{t-1}(r) \cdot P(x_t | r) \cdot (1 - H(r))$
  - Reset probability ($r \to 0$): $\tilde{R}_t(0) = \sum_{r} R_{t-1}(r) \cdot P(x_t | r) \cdot H(r)$
- **Pruning**: Low probability run-length hypotheses ($< 10^{-5}$) are pruned per step, ensuring $O(1)$ constant runtime per observation.

---

## 3. How Online Processing & Anti-Data-Leakage Work

Module 1 is engineered strictly for **online processing**:
- For every timestamp $t$, the system only consumes observations $x_{\le t}$.
- The streaming pipeline processes data observation-by-observation via Python generators.
- No future index lookups, global re-centering, or historical retro-active updates are permitted.
- The `StreamValidator` enforces monotonic timestamp ordering ($t_{curr} > t_{prev}$) and sequence continuity, raising a `ChronologyError` if out-of-order data is detected.

---

## 4. How Isolated Anomalies vs Structural Changes are Handled

A single abnormal observation must **NOT** automatically trigger a structural change or cause model retraining in downstream modules.

- **Isolated Extreme Spike**:
  When an extreme spike occurs at $t$, the likelihood under the current regime hypothesis drops, causing $P(r_t = 0)$ to rise temporarily. The detector flags `ANOMALY_SUSPECTED`.
  However, at step $t+1$, when demand returns to the baseline, the observation $x_{t+1}$ strongly matches the old regime hypothesis ($r_{t+1} = r_{t-1} + 2$). The Bayesian posterior $R_{t+1}$ instantly shifts back to the long run length, collapsing $P(r_{t+1}=0)$ back to 0.

- **Persistent Structural Change**:
  A structural shift is confirmed (`CHANGE_CONFIRMED` and `change_detected = True`) only when the reset probability $P(r_t=0)$ remains above the change threshold for at least `min_confirm_steps` (default 2 consecutive steps).

---

## 5. Bayesian Model Assumptions & Limitations

### Assumptions:
1. Observations within a regime are independently and identically distributed (i.i.d.) Gaussian given mean $\mu$ and variance $\sigma^2$.
2. The prior probability of a change point at any step is governed by a constant hazard rate $H(r) = 1/\lambda$.
3. Prior baseline demand parameter beliefs are initialized via Normal-Inverse-Gamma hyper-parameters $(\mu_0, \kappa_0, \alpha_0, \beta_0)$.

### Limitations:
- **Gradual Ramp Changes**: BOCPD is optimized for abrupt step shifts. For slow linear ramps (e.g. 500 MW to 600 MW over 40 steps), the reset probability per step is lower than for a sudden step change. Module 1 handles gradual drifts gracefully without crashing, though detecting gradual ramps yields a slight delay until variance accumulates.

---

## 6. Installation & Environment Setup

### Requirements:
- Python 3.11+
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `pytest >= 7.0.0`

### Installation:
```bash
pip install -r requirements.txt
```

---

## 7. Running Tests & Verification

Run the full test suite (unit tests & 4 synthetic benchmark scenarios):

```bash
pytest -v
```

All 4 benchmark tests are included:
- **TEST 1**: Stable demand with noise (verifies low change prob & 0 false alarms).
- **TEST 2**: Isolated extreme spike (verifies temporary anomaly flag without structural change).
- **TEST 3**: Persistent baseline shift 500 MW -> 600 MW (verifies strong change-point signal and confirmation).
- **TEST 4**: Gradual change (verifies smooth handling and documented behavior).

---

## 8. Running the Synthetic Streaming Demonstration

To watch Module 1 process all 4 benchmark streams sequentially in real time:

```bash
python demo.py
```

### Module 1 Output Schema (per observation):
- `timestamp`: String (ISO 8601)
- `observed_demand`: Float (MW)
- `change_point_prob`: Float ($P(r_t=0 | x_{1:t})$)
- `map_run_length`: Integer (MAP estimate of current regime duration)
- `change_detected`: Boolean (`True` if persistent shift confirmed)
- `detector_status`: Enum (`WARMUP`, `STABLE`, `ANOMALY_SUSPECTED`, `CHANGE_CONFIRMED`)
- `posterior_mean`: Float (Estimated demand mean of current regime)
- `posterior_std`: Float (Estimated standard deviation of current regime)
