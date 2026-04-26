# SAITS Pipeline

This folder contains the core PyPOTS SAITS wearable imputation pipeline.

This public SAITS version is sensor-only:

- 9 wearable sensor features
- no time-of-day channels
- no circadian harmonic channels
- empirical realistic block masks used as the MIT masking task
- curriculum buckets: `typ` -> `typ, mod` -> `typ, mod, sev`
- precomputed target/bucket masked training windows are built before batching
- evaluation buckets: `typ`, `mod`, and `sev_75_100`

## Run

From the repository root:

```bash
python -m saits.src.train --config saits/config.example.yaml
python -m saits.src.evaluate --config saits/config.example.yaml
```

Edit `saits/config.example.yaml` or copy it to your own config path before running.
The original protected Labfront/Garmin data are not included.
