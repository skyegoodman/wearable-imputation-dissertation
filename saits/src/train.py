from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repo root is importable when run as script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from saits.src.config import load_config
from saits.src.data import build_saits_datasets
from saits.src.dynamic_train import (
    build_precomputed_realistic_artifacts,
    build_train_val_windows,
    train_dynamic_realistic,
    train_precomputed_realistic,
)
from saits.src.model import build_model
from saits.src.utils import ensure_dir, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="saits/config.example.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    ensure_dir(cfg["train"]["saving_dir"])

    train_set, val_set, _, split = build_saits_datasets(cfg)
    features = list(cfg["data"]["features"])

    print("Split summary:")
    print(f"  split_time: {split.split_time}")
    print(f"  val_start_time: {split.val_start_time}")
    print(f"  train windows: {train_set['X'].shape[0]}")
    print(f"  val windows:   {val_set['X'].shape[0]}")

    use_dynamic = bool(cfg["train"].get("use_dynamic_realistic_train", False))
    if use_dynamic:
        realistic_mode = str(cfg["train"].get("realistic_train_mode", "precomputed"))
        if realistic_mode == "precomputed":
            artifacts = build_precomputed_realistic_artifacts(
                cfg=cfg,
                split=split,
                attrs=features,
                include_tod=bool(cfg["model"].get("include_tod", False)),
                include_harmonic=bool(cfg["model"].get("include_harmonic", False)),
                harmonic_features=list(cfg["model"].get("harmonic_features", [])),
                harmonic_min_obs=int(cfg["model"].get("harmonic_min_obs", 100)),
            )
            _ = train_precomputed_realistic(
                cfg=cfg,
                attrs=features,
                train_bank=artifacts["train_bank"],
                val_bank=artifacts["val_bank"],
            )
        else:
            train_X, val_X = build_train_val_windows(cfg, split, features)
            _ = train_dynamic_realistic(
                cfg=cfg,
                split=split,
                train_X=train_X,
                val_X=val_X,
                attrs=features,
            )
    else:
        n_steps = train_set["X"].shape[1]
        n_features = train_set["X"].shape[2]
        model = build_model(cfg, n_steps=n_steps, n_features=n_features)
        model.fit(train_set=train_set, val_set=val_set)

    print("Training completed.")


if __name__ == "__main__":
    main()
