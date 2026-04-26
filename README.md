# Wearable Imputation Pipelines

This repository contains public, data-free versions of the dissertation wearable
imputation pipelines.

- `brits/`: BRITS with time-of-day inputs and circadian harmonic priors for periodic wearable features.
- `saits/`: SAITS with realistic block masks used as the MIT masking task. SAITS is kept sensor-only: no time-of-day or harmonic auxiliary channels.

Protected Labfront/Garmin participant data are not included. Supply your own
preprocessed wearable time-series file using the schema in each pipeline folder.
