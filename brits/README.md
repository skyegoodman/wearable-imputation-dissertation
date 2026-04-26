# BRITS Pipeline

This folder contains the core BRITS wearable imputation pipeline.

The BRITS model uses:

- 9 wearable sensor features
- 2 time-of-day channels (`sin`, `cos`)
- 5 circadian harmonic priors for periodic features
- empirical realistic block-mask training
- curriculum buckets: `typ` -> `typ, mod` -> `typ, mod, sev`
- ramped block-loss weighting
- evaluation buckets: `typ`, `mod`, and `sev_75_100`

## Run

From the repository root:

```bash
python -m brits.src.train --config brits/config.example.yaml
python -m brits.src.evaluate --config brits/config.example.yaml
```

Edit `brits/config.example.yaml` or copy it to your own config path before running.
The original protected Labfront/Garmin data are not included.
