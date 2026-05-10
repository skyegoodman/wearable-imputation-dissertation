# Evaluating Deep Multivariate Imputation Models on Wearable Smartwatch Data 

Deep learning framework for imputing missing multimodal smartwatch data under realistic wearable missingness patterns.

This repository contains the reproducible codebase accompanying my MEng Engineering Mathematics dissertation at the University of Bristol:

> **Evaluating Deep Multivariate Imputation Models on Wearable Smartwatch Data**

The project evaluates state-of-the-art deep learning imputation models on real-world physiological smartwatch data collected from a person with epilepsy using a Garmin wearable device.

---

# Overview

Wearable devices are increasingly being explored for longitudinal health monitoring and seizure-risk forecasting. However, smartwatch data are heavily affected by structured missingness caused by:

- battery depletion
- charging periods
- sensor dropout
- poor skin contact
- motion artefacts
- Bluetooth synchronisation failures

Unlike many benchmark datasets used in multivariate time-series imputation research, wearable missingness is:

- contiguous across time
- highly co-missing between features
- driven by shared hardware dependencies
- heterogeneous across sensing modalities

Standard random-point masking protocols therefore produce unrealistically favourable evaluation conditions for wearable data.

This dissertation develops a wearable-realistic training and evaluation framework designed specifically for multimodal smartwatch missingness.

---

# Dissertation Goals

The project had three primary objectives:

1. Characterise the missingness structure of real wearable smartwatch data

2. Develop a wearable-realistic masking and evaluation protocol preserving:
   - contiguous missing runs
   - feature-specific outage distributions
   - co-missingness structure between wearable channels

3. Adapt and evaluate two state-of-the-art deep learning imputation models:
   - BRITS
   - SAITS

against classical interpolation baselines under realistic wearable conditions.

---

# Models

## BRITS

### Bidirectional Recurrent Imputation for Time Series

Original paper:

- Cao et al. (2018)
- **BRITS: Bidirectional Recurrent Imputation for Time Series**
- https://arxiv.org/pdf/1805.10572

BRITS is a bidirectional recurrent imputation architecture using:

- recurrent hidden-state dynamics
- temporal decay mechanisms
- explicit cross-feature regression
- forward/backward consistency regularisation

This repository extends BRITS with:

- wearable-realistic block masking
- curriculum masking schedules
- masked reconstruction objectives
- modified evaluation protocols
- time-of-day encodings
- circadian harmonic auxiliary channels

The modified training pipeline progressively exposes the model to increasingly difficult missingness regimes:

- typical gaps
- moderate gaps
- severe gaps

with final curriculum sampling weights:

```python
[0.25, 0.30, 0.45]
```

corresponding to:

```text
typical / moderate / severe
```

---

## SAITS

### Self-Attention-based Imputation for Time Series

Original paper:

- Du et al. (2022)
- **SAITS: Self-Attention-based Imputation for Time Series**
- https://arxiv.org/pdf/2202.08516

SAITS is a Transformer-based imputation architecture using:

- diagonally masked self-attention (DMSA)
- dual self-attention blocks
- weighted imputation fusion
- parallel temporal attention

This repository evaluates SAITS under structured wearable missingness rather than standard random-point masking protocols.

---

# Key Research Contribution

A major contribution of this dissertation is the introduction of a **wearable-realistic masking framework**.

Rather than randomly withholding isolated values, the pipeline:

- mines real missing-run templates from the wearable dataset
- preserves co-missingness between smartwatch features
- injects contiguous block masks during training and evaluation
- stratifies outages into:
  - typical
  - moderate
  - severe

This produces substantially more realistic evaluation conditions for wearable physiological data.

---

# Dataset

The dataset consists of multimodal Garmin smartwatch recordings collected within a University of Bristol epilepsy study.

Features include:

| Feature | Description |
|---|---|
| `hr` | Heart rate |
| `ibi` | Inter-beat interval |
| `pulseOx` | Blood oxygen saturation |
| `device_stress` | Garmin stress metric |
| `breathsPerMinute` | Respiratory rate |
| `steps` | Daily cumulative steps |
| `steps_rate` | Interval-level step rate |
| `bodyBattery` | Garmin recovery metric |
| `sleep` | Sleep stage |

The repository does **not** contain raw participant data due to ethical and privacy restrictions.

---

# Missingness Structure

The dissertation demonstrates that smartwatch missingness is highly sensor-coupled.

Examples include:

- `hr` and `ibi` failing almost deterministically together
- PPG-derived features exhibiting shared dropout structure
- accelerometer outages frequently corresponding to near-global device failure

This violates the assumptions underlying standard random-point masking evaluation.

The repository therefore implements:

- contiguous block masking
- co-missingness-aware masking
- feature-specific outage distributions
- severity-bucket evaluation

---

# Repository Purpose

This repository is primarily intended as a:

- reproducible research codebase
- dissertation companion repository
- implementation reference for wearable-realistic imputation protocols

The focus is reproducibility of:

- preprocessing
- masking generation
- training pipelines
- evaluation protocols
- model adaptation

---

# Repository Structure

```text
.
├── brits/
│   ├── src/                     # Core BRITS training/evaluation pipeline
│   ├── models/                  # BRITS model implementations
│   ├── notebooks/               # Experimental notebooks
│   ├── outputs/                 # Saved checkpoints and outputs
│   ├── config.example.yaml
│   ├── data_schema.md
│   └── requirements.txt
│
├── saits/
│   ├── src/                     # SAITS training/evaluation pipeline
│   ├── notebooks/
│   ├── outputs/
│   ├── config.example.yaml
│   ├── data_schema.md
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

---

# Training Protocol

The modified BRITS training pipeline includes:

- wearable-realistic block masking
- curriculum learning over outage severity
- masked reconstruction loss
- validation under realistic block masks
- early stopping
- checkpoint metadata for auxiliary temporal channels

The curriculum schedule progressively introduces more severe outages during training.

Validation is performed under realistic block masking rather than random-point holdout.

---

# Evaluation

Models were evaluated using both pointwise and distributional metrics.

## Pointwise Metrics

- Mean Absolute Error (MAE)
- Symmetric Mean Absolute Percentage Error (sMAPE)

## Distributional Fidelity

- Kernel Density Estimation (KDE)
- Jensen-Shannon Distance (JSD)

Distributional analysis was included because low pointwise error alone does not guarantee physiologically realistic imputations.

---

# Key Findings

Some major findings from the dissertation include:

- Wearable missingness is highly sensor-coupled
- Random-point masking substantially overestimates imputation performance
- Deep models benefit strongly from multimodal physiological structure
- Cross-feature helper availability collapses during severe outages
- BRITS performs strongly on dynamic cardiac features under long gaps
- SAITS achieved the strongest overall distributional fidelity
- Linear interpolation remained highly competitive for smoother features
- Deep models suppress cardiac distribution tails under severe missingness

---

# Circadian Modelling Experiments

Additional experiments explored incorporating temporal priors through:

- time-of-day encodings
- per-feature harmonic auxiliary channels
- rolling circadian fitting

Feature selection for circadian modelling used Lomb–Scargle periodograms to identify strong 24-hour periodicity in irregularly sampled wearable signals.

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- SciPy
- Matplotlib
- Jupyter Notebook

---

# Ethical Statement

Raw participant data are not included in this repository.

---

# Author

**Skye Goodman**  
MEng Engineering Mathematics  
University of Bristol
