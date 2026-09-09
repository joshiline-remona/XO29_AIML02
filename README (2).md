# AERIS

## Adaptive Electricity Regime Intelligence System

**Team Name:** VERVE
**Team Leader:** Joshiline Remona

> Detect demand shifts. Understand the regime. Select the right model. Forecast with confidence.

---

## Overview

**AERIS** is an adaptive electricity demand forecasting system designed to handle **Concept Drift** in time-series data.

Unlike conventional forecasting systems that assume historical demand patterns will remain stable, AERIS continuously monitors electricity consumption, identifies changes in demand behavior, recognizes the current demand regime, dynamically selects the most suitable forecasting model, and provides an adaptive uncertainty interval.

AERIS also considers **calendar and event-driven demand patterns**, including festivals and holidays such as Diwali, Pongal, Christmas, Eid, and New Year, because electricity consumption can change significantly during these periods.

### Core Pipeline

```text
Historical + Recent Demand
          |
          v
   Feature Engineering
          |
          v
Festival / Holiday Intelligence
          |
          v
Bayesian Change-Point Detection
          |
          v
     Regime Detection
          |
          v
    Dynamic Model Router
          |
          v
    Best Model Selection
          |
          v
Forecast + Adaptive Conformal Prediction
          |
          v
Prediction + Uncertainty Interval
          |
          v
   Error & Drift Monitoring
          |
          v
      Continuous Adaptation
```

---

# Problem Statement

Electricity demand is not stationary.

Consumption can change because of:

* Daily and weekly patterns
* Seasonal changes
* Weather conditions
* Public holidays
* Festivals
* Sudden demand spikes
* Industrial or commercial activity
* Changing consumer behavior
* Long-term structural changes

A traditional forecasting model trained on historical data may perform well under normal conditions but fail when the underlying demand pattern changes.

This phenomenon is known as **Concept Drift**.

### The Challenge

> How can an electricity demand forecasting system recognize when demand behavior changes and automatically adapt its forecasting strategy?

---

# Proposed Solution

AERIS approaches forecasting as an intelligent decision-making problem rather than simply training one prediction model.

It combines four major intelligence layers:

## 1. Bayesian Change-Point Detection

AERIS continuously monitors demand behavior and estimates whether the underlying statistical pattern has changed.

It helps distinguish between:

```text
Temporary Spike
      |
      v
Normal Pattern
```

and:

```text
Persistent Change
      |
      v
New Demand Behavior
```

This prevents the system from treating every unusual observation as concept drift.

---

## 2. Regime Detection

After identifying significant changes, AERIS determines the current **demand regime**.

Possible regimes include:

```text
NORMAL
HIGH DEMAND
LOW DEMAND
FESTIVAL / HOLIDAY
POST-FESTIVAL
UNUSUAL / SHIFTED
```

The detected regime provides context to the forecasting system.

---

## 3. Dynamic Model Routing

AERIS maintains multiple forecasting models instead of depending on one model.

Example model portfolio:

```text
                 +-- Stable Model
                 |
Current Regime --+-- Recent-Trend Model
                 |
                 +-- Gradient Boosting Model
                 |
                 +-- Statistical Model
```

The system evaluates recent model performance and dynamically routes the prediction to the model most suitable for the current regime.

Therefore:

> The best model for normal demand does not have to be the best model during a festival or after a structural shift.

---

## 4. Adaptive Conformal Prediction

AERIS provides both a point forecast and an adaptive uncertainty interval.

Instead of producing only:

```text
Tomorrow's Demand = 31.4
```

AERIS can provide:

```text
Tomorrow's Demand = 31.4
Expected Range = [27.8, 35.2]
```

The prediction interval adapts according to recent forecasting errors and changing uncertainty.

This makes the forecast more useful for grid planning and risk-aware decision making.

---

# Festival and Holiday Intelligence

A major feature of AERIS is the ability to incorporate **event-driven electricity demand**.

Electricity consumption can change significantly during periods such as:

* Diwali
* Pongal
* Christmas
* New Year
* Eid
* Navratri
* Regional festivals
* Public holidays
* Extended holiday periods

Instead of allowing the drift detector to incorrectly classify every festival spike as concept drift, AERIS explicitly models these events.

## Festival Features

| Feature                    | Description                                    |
| -------------------------- | ---------------------------------------------- |
| `is_festival`              | Whether the date is associated with a festival |
| `festival_name`            | Name of the festival                           |
| `is_holiday`               | Public holiday indicator                       |
| `festival_type`            | Major, regional, or public holiday             |
| `days_to_festival`         | Days before the festival                       |
| `days_after_festival`      | Days after the festival                        |
| `festival_window`          | Indicates the festival period                  |
| `is_weekend`               | Weekend indicator                              |
| `festival_intensity`       | Historical demand impact                       |
| `same_event_previous_year` | Previous year's corresponding event            |

### Festival-Aware Forecasting

Instead of interpreting:

```text
Normal Demand
      |
      v
Festival Demand Spike
      |
      v
Lower Demand
```

as:

```text
CONCEPT DRIFT
```

AERIS can recognize:

```text
Known Festival Event
        |
        v
Expected Demand Change
        |
        v
Festival Regime
        |
        v
Festival-appropriate Model
        |
        v
Forecast
```

This allows AERIS to distinguish **predictable demand variation** from **genuine concept drift**.

---

# Dataset

The current dataset contains daily electricity demand observations.

### Dataset Characteristics

* **Format:** Excel (`.xlsx`)
* **Observations:** 550 daily records
* **Date Range:** January 2024 – July 2025
* **Target:** Electricity demand
* **Additional Features:** `feature_1` to `feature_6`

Example structure:

```text
timestamp | feature_1 | feature_2 | ... | feature_6 | target
```

---

# Feature Engineering

AERIS combines historical demand, temporal patterns, and event information.

## Time Features

```text
Day of Week
Day of Month
Month
Year
Weekend
```

## Cyclic Features

```text
sin(day_of_week)
cos(day_of_week)

sin(month)
cos(month)
```

## Demand Lag Features

```text
lag_1
lag_2
lag_7
lag_14
lag_21
lag_30
lag_60
```

## Rolling Statistics

```text
rolling_mean_7
rolling_std_7

rolling_mean_14
rolling_std_14

rolling_mean_30
rolling_std_30
```

## Event Features

```text
is_festival
festival_name
is_holiday
days_to_festival
days_after_festival
festival_window
festival_intensity
```

## Original Dataset Features

```text
feature_1
feature_2
feature_3
feature_4
feature_5
feature_6
```

---

# Intelligent Forecasting Architecture

```text
                 ELECTRICITY DEMAND DATA
                           |
                           v
                  FEATURE ENGINEERING
                           |
              +------------+------------+
              |                         |
              v                         v
       Demand Patterns          Festival/Holiday
                                  Intelligence
              |                         |
              +------------+------------+
                           |
                           v
                CHANGE-POINT DETECTOR
                           |
                           v
                    REGIME DETECTOR
                           |
              +------------+------------+
              |                         |
              v                         v
        Normal Regime           Festival Regime
              |                         |
              +------------+------------+
                           |
                           v
                     MODEL ROUTER
                           |
              +------------+------------+
              |            |            |
              v            v            v
           XGBoost     Recent Model  Statistical
              |            |            |
              +------------+------------+
                           |
                           v
                     POINT FORECAST
                           |
                           v
              ADAPTIVE CONFORMAL
                    PREDICTION
                           |
                           v
               FORECAST + UNCERTAINTY
                           |
                           v
                    ACTUAL DEMAND
                           |
                           v
                 ERROR / DRIFT MONITOR
                           |
                           +---------> ADAPT
```

---

# How AERIS Adapts

AERIS follows a continuous feedback loop:

```text
DETECT
   |
   v
Identify possible change
   |
   v
UNDERSTAND
   |
   v
Determine current regime
   |
   v
SELECT
   |
   v
Choose the best-performing model
   |
   v
FORECAST
   |
   v
Generate demand prediction
   |
   v
QUANTIFY
   |
   v
Generate adaptive prediction interval
   |
   v
MONITOR
   |
   v
Compare prediction with actual demand
   |
   v
ADAPT
   |
   v
Update model, routing, and uncertainty
   |
   v
REPEAT
```

---

# Key Innovation

## Conventional Forecasting

```text
Historical Data
      |
      v
One Model
      |
      v
Forecast
```

## AERIS

```text
Historical + Recent Data
          |
          v
Festival/Holiday Context
          |
          v
Change Detection
          |
          v
Regime Understanding
          |
          v
Dynamic Model Selection
          |
          v
Adaptive Forecasting
          |
          v
Uncertainty Quantification
          |
          v
Continuous Monitoring
```

### Core Idea

> AERIS does not assume that the future will always behave like the past. It identifies the current demand environment, accounts for predictable events, detects genuine behavioral changes, and adapts its forecasting strategy accordingly.

---

# Evaluation Strategy

AERIS will be compared against conventional approaches.

## Baseline 1 — Static Model

One model trained on historical data with no adaptation.

## Baseline 2 — Periodic Retraining

Model retrained at fixed intervals.

## Proposed — AERIS

Adaptive regime-aware forecasting with dynamic routing and uncertainty estimation.

### Evaluation Metrics

### Forecast Accuracy

* MAE
* RMSE
* MAPE

### Uncertainty Quality

* Prediction interval coverage
* Average interval width

### Adaptation Performance

* Change-point detection accuracy
* Detection delay
* Model switching frequency
* Number of retrainings
* Performance before and after drift

### Festival Performance

* Festival-period MAE
* Festival-period RMSE
* Normal-period vs festival-period accuracy

---

# Dashboard

The AERIS dashboard can display:

## Demand Forecast

```text
Actual Demand
Predicted Demand
Prediction Interval
```

## Regime Status

```text
Current Regime: FESTIVAL
Confidence: 87%
```

## Change Detection

```text
Change Probability: 0.82
Status: STRUCTURAL SHIFT
```

## Model Router

```text
Selected Model: Recent-Trend Model
Reason: Best performance in current regime
```

## Event Intelligence

```text
Upcoming Event: Diwali
Festival Window: Active
Expected Demand Effect: Elevated
```

---

# Technology Stack

| Component        | Technology                                    |
| ---------------- | --------------------------------------------- |
| Programming      | Python                                        |
| Data Processing  | Pandas, NumPy                                 |
| Forecasting      | XGBoost / LightGBM / Statistical Models       |
| Change Detection | Bayesian Change-Point Detection               |
| Regime Detection | Machine Learning / Statistical Classification |
| Model Routing    | Dynamic Performance-Based Selection           |
| Uncertainty      | Adaptive Conformal Prediction                 |
| Visualization    | Plotly / Matplotlib                           |
| Dashboard        | Streamlit                                     |
| Dataset          | Excel                                         |
| Version Control  | Git / GitHub                                  |

---

# Project Structure

```text
AERIS/
|
+-- data/
|   +-- xo.xlsx
|
+-- src/
|   +-- preprocessing.py
|   +-- feature_engineering.py
|   +-- festival_features.py
|   +-- forecasting.py
|   +-- change_point.py
|   +-- regime_detection.py
|   +-- model_router.py
|   +-- conformal_prediction.py
|   +-- adaptation.py
|   +-- evaluation.py
|
+-- dashboard/
|   +-- app.py
|
+-- notebooks/
|   +-- exploratory_analysis.ipynb
|
+-- results/
|   +-- forecasts.csv
|   +-- metrics.csv
|   +-- figures/
|
+-- main.py
+-- requirements.txt
+-- README.md
```

---

# Preventing Data Leakage

Since electricity demand is a time series, AERIS uses **chronological validation**.

No future information is allowed to influence historical predictions.

```text
TRAIN --------------------> TEST
Past                         Future
```

Evaluation uses:

* Walk-forward validation
* Expanding or rolling training windows
* TimeSeriesSplit where appropriate
* Lagged features generated only from available historical observations
* Festival information that would have been known at forecast time

---

# Installation

```bash
git clone <repository-url>

cd AERIS

pip install -r requirements.txt
```

Run the forecasting pipeline:

```bash
python main.py
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

---

# Real-World Applications

AERIS can support:

* Electricity grid planning
* Demand-response systems
* Power generation scheduling
* Renewable energy integration
* Peak-load management
* Energy storage planning
* Utility risk management
* Festival-period load planning

---

# Expected Impact

AERIS aims to improve electricity demand forecasting by making predictions:

### More Accurate

Through regime-specific model selection.

### More Adaptive

Through continuous change detection and monitoring.

### More Reliable

Through uncertainty-aware forecasting.

### More Context-Aware

Through festivals, holidays, calendar effects, and historical demand patterns.

### More Operationally Useful

By indicating not only **what demand is expected**, but also **how confident the system is**.

---



> **AERIS is an Adaptive Electricity Regime Intelligence System designed to forecast electricity demand under changing conditions.**
>
> Instead of relying on a single forecasting model, AERIS detects genuine changes in demand behavior, identifies the current demand regime, considers predictable events such as festivals and holidays, dynamically selects the model best suited to that regime, and produces an adaptive uncertainty interval.
>
> **AERIS does not just predict demand. It understands when the demand environment changes and adapts accordingly.**

---

# One-Line USP

> **AERIS transforms electricity forecasting from static prediction into context-aware, regime-adaptive intelligence.**

---

# Team

**Team Name:** VERVE

**Team Leader:** Joshiline Remona

**Project:** AERIS — Adaptive Electricity Regime Intelligence System
