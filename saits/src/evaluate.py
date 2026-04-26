from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

# Ensure repo root is importable when run as script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from saits.src.config import load_config
from saits.src.data import (
    build_mask_library_from_df,
    build_realistic_test_grid_with_meta,
    build_saits_datasets,
)
from saits.src.model import build_model
from saits.src.utils import set_seed


def masked_metrics(imputed: np.ndarray, X_ori: np.ndarray, indicating_mask: np.ndarray) -> tuple[float, float]:
    m = indicating_mask.astype(bool)
    if not np.any(m):
        return float("nan"), float("nan")
    err = imputed[m] - X_ori[m]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse


def masked_mae_per_feature(imputed: np.ndarray, X_ori: np.ndarray, indicating_mask: np.ndarray, attrs: list[str]) -> dict[str, float]:
    m = indicating_mask.astype(bool)
    out = {}
    for j, a in enumerate(attrs):
        mj = m[:, :, j]
        if not np.any(mj):
            out[a] = float("nan")
            continue
        err = np.abs(imputed[:, :, j][mj] - X_ori[:, :, j][mj])
        out[a] = float(np.mean(err))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="saits/config.example.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    train_set, val_set, test_set, split = build_saits_datasets(cfg)

    n_steps = train_set["X"].shape[1]
    n_features = train_set["X"].shape[2]
    model = build_model(cfg, n_steps=n_steps, n_features=n_features)

    use_dynamic = bool(cfg.get("train", {}).get("use_dynamic_realistic_train", False))
    if use_dynamic:
        ckpt_path = Path(cfg["train"]["saving_dir"]) / "saits_dynamic_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Dynamic training checkpoint not found: {ckpt_path}. "
                "Run python -m saits.src.train first."
            )
        module = None
        for attr in ("model", "_model", "backbone", "_backbone"):
            m = getattr(model, attr, None)
            if isinstance(m, torch.nn.Module):
                module = m
                break
        if module is None:
            raise RuntimeError("Could not locate internal SAITS torch module for loading checkpoint.")
        state = torch.load(ckpt_path, map_location="cpu")
        module.load_state_dict(state)
    else:
        # Keep old behavior for non-dynamic mode.
        model.fit(train_set=train_set, val_set=val_set)

    attrs = list(cfg["data"]["features"])

    # Baseline random-point test report
    imputed_test = model.impute(test_set)
    base_mae, _ = masked_metrics(
        imputed=imputed_test,
        X_ori=test_set["X_ori"],
        indicating_mask=test_set["indicating_mask"],
    )
    base_by_feat = masked_mae_per_feature(
        imputed=imputed_test,
        X_ori=test_set["X_ori"],
        indicating_mask=test_set["indicating_mask"],
        attrs=attrs,
    )

    print("\n" + "#" * 100)
    print("FINAL TEST RESULTS (report these)")
    print(f"Random-point test MAE (overall): {base_mae:.4f}")
    print("Random-point test MAE (per feature):")
    for k in attrs:
        print(f"  {k:16s}: {base_by_feat[k]:.4f}")

    # Realistic bucket report
    grid = build_realistic_test_grid_with_meta(cfg, split)
    buckets = list(cfg.get("eval", {}).get("buckets", ["typ", "mod", "sev"]))
    rows = []
    for b in buckets:
        for target in attrs:
            ds, meta = grid[(target, b)]
            imputed = model.impute(ds)
            mae, _ = masked_metrics(
                imputed=imputed,
                X_ori=ds["X_ori"],
                indicating_mask=ds["indicating_mask"],
            )
            Ls = [int(m.get("L", 0)) for m in meta if bool(m.get("placed", False))]
            helpers = [float(m.get("avg_helpers_observed", 0.0)) for m in meta if bool(m.get("placed", False))]
            L_mean = float(np.mean(Ls)) if Ls else 0.0
            L_median = float(np.median(Ls)) if Ls else 0.0
            avg_helpers_observed = float(np.mean(helpers)) if helpers else 0.0
            rows.append(
                {
                    "target": target,
                    "bucket": b,
                    "mae": float(mae),
                    "L_mean": L_mean,
                    "L_median": L_median,
                    "avg_helpers_observed": avg_helpers_observed,
                }
            )

    results_df = pd.DataFrame(rows).sort_values(["target", "bucket"])
    print("\nRealistic-BLOCK test results (per target x bucket):")
    print(results_df.to_string(index=False))

    print("\nRealistic-BLOCK test MAE (avg over targets) by bucket:")
    for b in buckets:
        avg_b = results_df[results_df["bucket"] == b]["mae"].mean()
        print(f"  {b}: {avg_b:.4f}")

    # Frequency-weighted headline MAE
    eval_cfg = cfg.get("eval", {})
    q = tuple(float(x) for x in eval_cfg.get("quantiles", [0.50, 0.75, 0.90]))
    mask_lib = build_mask_library_from_df(
        split.train,
        attrs,
        quantiles=(q[0], q[1], q[2]),
        min_run=int(eval_cfg.get("min_run", 1)),
        max_runs_per_bin=eval_cfg.get("max_runs_per_bin"),
    )

    weights = {}
    for target in attrs:
        counts = {}
        for b in buckets:
            if b == "sev_75_100":
                counts[b] = len(mask_lib[target]["sev"]) + len(mask_lib[target]["ext"])
            else:
                counts[b] = len(mask_lib[target][b])
        total = sum(counts.values())
        weights[target] = {b: (counts[b] / total if total > 0 else 0.0) for b in buckets}

    weighted_rows = []
    for target in attrs:
        mae_b = {}
        for b in buckets:
            mae_b[b] = float(
                results_df[
                    (results_df["target"] == target) & (results_df["bucket"] == b)
                ]["mae"].iloc[0]
            )
        terms = []
        valid_weight_sum = 0.0
        for b in buckets:
            w = float(weights[target][b])
            m = float(mae_b[b])
            if w <= 0.0 or np.isnan(m):
                continue
            terms.append(w * m)
            valid_weight_sum += w
        mae_weighted = float(sum(terms) / valid_weight_sum) if valid_weight_sum > 0 else float("nan")
        row = {
            "target": target,
            "MAE_weighted": mae_weighted,
        }
        for b in buckets:
            row[f"w_{b}"] = weights[target][b]
        for b in buckets:
            row[f"MAE_{b}"] = mae_b[b]
        weighted_rows.append(row)

    weighted_df = pd.DataFrame(weighted_rows).sort_values("MAE_weighted", ascending=False)
    print("\nFrequency-weighted MAE per target (real-world headline):")
    print(weighted_df.to_string(index=False))

    overall_weighted = float(weighted_df["MAE_weighted"].mean())
    print(f"\nOverall frequency-weighted MAE (mean over targets): {overall_weighted:.4f}")


if __name__ == "__main__":
    main()
