from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from saits.src.data import (
    append_auxiliary_channels,
    build_mask_library_from_df,
    compute_feature_stats,
    compute_harmonic_stats,
    infos_to_dataset,
    make_realistic_eval_set,
    materialize_realistic_window_infos,
    make_windows,
    normalize_windows,
    _fit_weekly_24h_harmonics,
)
from saits.src.model import build_model


def _pick_device(cfg: dict) -> torch.device:
    train_device = cfg["train"].get("device")
    if train_device:
        return torch.device(str(train_device))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _locate_torch_module(model_obj) -> torch.nn.Module:
    for attr in ("model", "_model", "backbone", "_backbone"):
        m = getattr(model_obj, attr, None)
        if isinstance(m, torch.nn.Module):
            return m
    raise RuntimeError(
        "Could not find internal torch module on PyPOTS SAITS object. "
        "This PyPOTS version may require adapter updates."
    )


def _batch_to_torch(batch_np: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in batch_np.items():
        out[k] = torch.from_numpy(v).to(device=device, dtype=torch.float32)
    return out


def _assemble_inputs(batch_t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    X_in = batch_t["X"]
    missing_mask = (~torch.isnan(X_in)).to(torch.float32)
    X_filled = torch.nan_to_num(X_in, nan=0.0)
    return {
        "X": X_filled,
        "X_ori": torch.nan_to_num(batch_t["X_ori"], nan=0.0),
        "missing_mask": missing_mask,
        "indicating_mask": batch_t["indicating_mask"],
    }


def _extract_loss(model_forward_out):
    if isinstance(model_forward_out, dict):
        for key in ("loss", "training_loss", "imputation_loss"):
            if key in model_forward_out:
                return model_forward_out[key]
    if torch.is_tensor(model_forward_out):
        return model_forward_out
    raise RuntimeError("Could not extract training loss from SAITS forward output.")


def _extract_imputed(model_forward_out):
    if isinstance(model_forward_out, dict):
        for key in ("imputed_data", "imputation", "X_hat"):
            if key in model_forward_out and torch.is_tensor(model_forward_out[key]):
                return model_forward_out[key]
    return None


def _masked_mae(imputed: np.ndarray, X_ori: np.ndarray, indicating_mask: np.ndarray) -> float:
    m = indicating_mask.astype(bool)
    if not np.any(m):
        return float("nan")
    err = np.abs(imputed[m] - X_ori[m])
    return float(np.mean(err))


def _eval_realistic_val_avg(
    model_obj,
    val_X: np.ndarray,
    attrs: list[str],
    mask_library: dict,
    buckets: list[str],
    targets: list[str],
    cfg: dict,
    seed: int,
) -> float:
    train_cfg = cfg["train"]
    maes = []
    for b_idx, bucket in enumerate(buckets):
        for i, target in enumerate(targets):
            ds = make_realistic_eval_set(
                X=val_X,
                attributes=attrs,
                mask_library=mask_library,
                holdout_target=target,
                bucket=bucket,
                seed=seed + 1000 + 100 * b_idx + i,
                max_resample_tries=int(train_cfg.get("realistic_max_resample_tries", 25)),
                min_target_gt_frac=float(train_cfg.get("realistic_min_target_gt_frac", 0.80)),
                min_helper_template_match_frac=float(
                    train_cfg.get("realistic_min_helper_template_match_frac", 0.80)
                ),
                target_only_eval=bool(train_cfg.get("realistic_target_only_eval", True)),
                stratify_starts=bool(train_cfg.get("realistic_stratify_starts", True)),
            )
            imputed = model_obj.impute(ds)
            maes.append(_masked_mae(imputed, ds["X_ori"], ds["indicating_mask"]))
    return float(np.nanmean(maes))


def curriculum(epoch: int, max_epochs: int):
    frac = epoch / max_epochs
    if frac <= 0.33:
        return ["typ"], np.array([1.0], dtype=float)
    if frac <= 0.66:
        return ["typ", "mod"], np.array([0.6, 0.4], dtype=float)
    return ["typ", "mod", "sev"], np.array([0.25, 0.30, 0.45], dtype=float)


def validation_buckets_for_epoch(epoch: int, max_epochs: int):
    return ["mod"]


def lambda_ramp(
    epoch: int,
    max_epochs: int,
    ramp_frac: float = 0.25,
    max_lambda: float = 0.5,
):
    ramp_epochs = max(1, int(max_epochs * ramp_frac))
    if ramp_epochs <= 1:
        return float(max_lambda)
    if epoch >= ramp_epochs:
        return float(max_lambda)
    return float(max_lambda * (epoch - 1) / max(1, ramp_epochs - 1))


def build_precomputed_realistic_artifacts(
    cfg: dict,
    split,
    attrs: list[str],
    include_tod: bool = False,
    include_harmonic: bool = False,
    harmonic_features: list[str] | None = None,
    harmonic_min_obs: int = 100,
):
    train_cfg = cfg["train"]
    win_cfg = cfg["window"]
    seed = int(cfg["seed"])

    seq_len = int(win_cfg["seq_len"])
    train_stride = int(win_cfg.get("train_stride", win_cfg.get("stride", 60)))
    eval_stride = int(win_cfg.get("eval_stride", train_stride))
    mean, std = compute_feature_stats(split.train, attrs)

    q = tuple(float(x) for x in train_cfg.get("realistic_quantiles", [0.50, 0.75, 0.90]))
    mask_library = build_mask_library_from_df(
        split.train,
        attrs,
        quantiles=(q[0], q[1], q[2]),
        min_run=int(train_cfg.get("realistic_min_run", 2)),
        max_runs_per_bin=train_cfg.get("realistic_max_runs_per_bin"),
    )

    buckets = list(train_cfg.get("realistic_train_buckets", ["typ", "mod", "sev"]))
    train_targets_cfg = train_cfg.get("realistic_train_targets")
    train_targets = list(attrs if train_targets_cfg is None else train_targets_cfg)
    report_targets_cfg = train_cfg.get("report_targets")
    report_targets = list(attrs if report_targets_cfg is None else report_targets_cfg)

    train_info_bank: dict[tuple[str, str], list[dict]] = {}
    running_seed = seed + 1000
    for bucket in buckets:
        for target in train_targets:
            train_info_bank[(bucket, target)] = materialize_realistic_window_infos(
                df_part=split.train,
                features=attrs,
                seq_len=seq_len,
                stride=train_stride,
                mean=mean,
                std=std,
                holdout_target=target,
                bucket=bucket,
                mask_library=mask_library,
                seed=running_seed,
                max_resample_tries=int(train_cfg.get("realistic_max_resample_tries", 25)),
                min_target_gt_frac=float(train_cfg.get("realistic_min_target_gt_frac", 0.80)),
                min_helper_template_match_frac=float(
                    train_cfg.get("realistic_min_helper_template_match_frac", 0.80)
                ),
                target_only_eval=bool(train_cfg.get("realistic_target_only_eval", True)),
                stratify_starts=bool(train_cfg.get("realistic_stratify_starts", True)),
            )
            running_seed += 1000

    val_info_bank: dict[tuple[str, str], list[dict]] = {}
    val_seed_base = seed + 50000
    for b_idx, bucket in enumerate(buckets):
        for i, target in enumerate(report_targets):
            val_info_bank[(bucket, target)] = materialize_realistic_window_infos(
                df_part=split.val,
                features=attrs,
                seq_len=seq_len,
                stride=eval_stride,
                mean=mean,
                std=std,
                holdout_target=target,
                bucket=bucket,
                mask_library=mask_library,
                seed=val_seed_base + 100 * b_idx + i,
                max_resample_tries=int(train_cfg.get("realistic_max_resample_tries", 25)),
                min_target_gt_frac=float(train_cfg.get("realistic_min_target_gt_frac", 0.80)),
                min_helper_template_match_frac=float(
                    train_cfg.get("realistic_min_helper_template_match_frac", 0.80)
                ),
                target_only_eval=bool(train_cfg.get("realistic_target_only_eval", True)),
                stratify_starts=bool(train_cfg.get("realistic_stratify_starts", True)),
            )

    harmonic_stats = None
    if include_harmonic:
        harmonic_features = list(harmonic_features or [])
        if len(harmonic_features) == 0:
            raise ValueError("harmonic_features must be provided when include_harmonic=True.")

        all_train_infos = []
        for infos in train_info_bank.values():
            _fit_weekly_24h_harmonics(
                record_infos=infos,
                attributes=attrs,
                mean=mean,
                std=std,
                harmonic_features=harmonic_features,
                min_obs=harmonic_min_obs,
            )
            all_train_infos.extend(infos)

        harmonic_stats = compute_harmonic_stats(all_train_infos)

        for infos in val_info_bank.values():
            _fit_weekly_24h_harmonics(
                record_infos=infos,
                attributes=attrs,
                mean=mean,
                std=std,
                harmonic_features=harmonic_features,
                min_obs=harmonic_min_obs,
            )

    train_bank = {
        key: infos_to_dataset(
            append_auxiliary_channels(
                infos,
                include_tod=include_tod,
                include_harmonic=include_harmonic,
                harmonic_stats=harmonic_stats,
            )
        )
        for key, infos in train_info_bank.items()
    }

    val_bank = {
        key: infos_to_dataset(
            append_auxiliary_channels(
                infos,
                include_tod=include_tod,
                include_harmonic=include_harmonic,
                harmonic_stats=harmonic_stats,
            )
        )
        for key, infos in val_info_bank.items()
    }

    return {
        "train_bank": train_bank,
        "val_bank": val_bank,
        "mask_library": mask_library,
        "mean": mean,
        "std": std,
        "buckets": buckets,
        "train_targets": train_targets,
        "report_targets": report_targets,
        "harmonic_stats": harmonic_stats,
        "input_features": int(next(iter(train_bank.values()))["X"].shape[2]),
    }


def train_dynamic_realistic(
    cfg: dict,
    split,
    train_X: np.ndarray,
    val_X: np.ndarray,
    attrs: list[str],
):
    train_cfg = cfg["train"]
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed + 123)

    q = tuple(float(x) for x in train_cfg.get("realistic_quantiles", [0.50, 0.75, 0.90]))
    mask_library = build_mask_library_from_df(
        split.train,
        attrs,
        quantiles=(q[0], q[1], q[2]),
        min_run=int(train_cfg.get("realistic_min_run", 2)),
        max_runs_per_bin=train_cfg.get("realistic_max_runs_per_bin"),
    )

    buckets = list(train_cfg.get("realistic_train_buckets", ["typ", "mod", "sev"]))
    train_targets_cfg = train_cfg.get("realistic_train_targets")
    train_targets = list(attrs if train_targets_cfg is None else train_targets_cfg)
    report_targets_cfg = train_cfg.get("report_targets")
    report_targets = list(attrs if report_targets_cfg is None else report_targets_cfg)

    n_steps = int(train_X.shape[1])
    n_features = int(train_X.shape[2])
    model_obj = build_model(cfg, n_steps=n_steps, n_features=n_features)
    module = _locate_torch_module(model_obj)

    device = _pick_device(cfg)
    module.to(device)
    module.train()

    opt = torch.optim.Adam(
        module.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    steps_cfg = train_cfg.get("steps_per_epoch")
    steps_per_epoch = max(1, len(train_X) // batch_size) if steps_cfg is None else int(steps_cfg)
    patience = int(train_cfg.get("patience", 10))
    min_delta = float(train_cfg.get("min_delta", 0.0))
    ramp_frac = float(train_cfg.get("ramp_frac", 0.25))
    max_lambda = float(train_cfg.get("max_lambda", 0.5))
    use_weighted_block_objective = bool(train_cfg.get("use_weighted_block_objective", False))
    cfg_name = train_cfg.get("log_cfg_name", "SAITS")

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    full_phase_start_epoch = int(np.floor(0.66 * epochs)) + 1

    print("Dynamic realistic-mask training enabled.")
    print(f"  train windows: {len(train_X)} | val windows: {len(val_X)}")
    print(f"  buckets: {buckets}")
    print(f"  train targets: {train_targets}")
    print(f"  val target set: {report_targets}")

    for epoch in range(1, epochs + 1):
        module.train()
        active_buckets, p_bucket = curriculum(epoch, epochs)
        active_buckets = [b for b in active_buckets if b in buckets]
        p_bucket = p_bucket[: len(active_buckets)]
        p_bucket = p_bucket / p_bucket.sum()
        lam = lambda_ramp(epoch, epochs, ramp_frac=ramp_frac, max_lambda=max_lambda)

        sum_total = 0.0
        sum_brits = 0.0
        sum_block = 0.0
        sum_block_pts = 0.0
        sum_scale = 0.0

        for step in range(steps_per_epoch):
            target = str(rng.choice(train_targets))
            target_idx = int(attrs.index(target))
            bucket = str(rng.choice(active_buckets, p=p_bucket))
            idx = rng.integers(0, len(train_X), size=batch_size)
            X_batch = train_X[idx]

            batch_np = make_realistic_eval_set(
                X=X_batch,
                attributes=attrs,
                mask_library=mask_library,
                holdout_target=target,
                bucket=bucket,
                seed=seed + epoch * 1_000_000 + step * 1000,
                max_resample_tries=int(train_cfg.get("realistic_max_resample_tries", 25)),
                min_target_gt_frac=float(train_cfg.get("realistic_min_target_gt_frac", 0.80)),
                min_helper_template_match_frac=float(
                    train_cfg.get("realistic_min_helper_template_match_frac", 0.80)
                ),
                target_only_eval=bool(train_cfg.get("realistic_target_only_eval", True)),
                stratify_starts=bool(train_cfg.get("realistic_stratify_starts", True)),
            )

            batch_t = _batch_to_torch(batch_np, device)
            inputs = _assemble_inputs(batch_t)

            opt.zero_grad()
            try:
                out = module(inputs, calc_criterion=True)
            except TypeError:
                try:
                    out = module(inputs, training=True)
                except TypeError:
                    out = module(inputs)
            loss_brits = _extract_loss(out)

            imputed = _extract_imputed(out)
            if imputed is None:
                loss_block = torch.zeros((), device=device, dtype=torch.float32)
                n_block = torch.zeros((), device=device, dtype=torch.float32)
            else:
                xori_target = torch.nan_to_num(batch_t["X_ori"][:, :, target_idx], nan=0.0)
                imputed_target = imputed[:, :, target_idx]
                mask_target = batch_t["indicating_mask"][:, :, target_idx]
                abs_err = torch.abs(imputed_target - xori_target) * mask_target
                n_block = mask_target.sum()
                loss_block = abs_err.sum() / (n_block + 1e-8)

            with torch.no_grad():
                scale = (loss_brits.detach() / (loss_block.detach() + 1e-8)).clamp(1.0, 2000.0)

            if use_weighted_block_objective:
                loss_total = loss_brits + lam * scale * loss_block
            else:
                loss_total = loss_brits

            loss_total.backward()
            opt.step()
            sum_total += float(loss_total.item())
            sum_brits += float(loss_brits.item())
            sum_block += float(loss_block.item())
            sum_block_pts += float(n_block.item())
            sum_scale += float(scale.item())

        val_buckets = validation_buckets_for_epoch(epoch, epochs)
        val_mae = _eval_realistic_val_avg(
            model_obj=model_obj,
            val_X=val_X,
            attrs=attrs,
            mask_library=mask_library,
            buckets=val_buckets,
            targets=report_targets,
            cfg=cfg,
            seed=seed + epoch * 10_000,
        )
        logs = {
            "loss_total": sum_total / max(1, steps_per_epoch),
            "loss_brits": sum_brits / max(1, steps_per_epoch),
            "loss_block": sum_block / max(1, steps_per_epoch),
            "scale": sum_scale / max(1, steps_per_epoch),
            "lambda": lam,
            "active_buckets": active_buckets,
            "avg_block_points_per_batch": sum_block_pts / max(1, steps_per_epoch),
        }
        cfg_log = {
            "name": cfg_name,
            "lr": float(train_cfg.get("lr", 1e-3)),
            "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
            "batch_size": int(train_cfg.get("batch_size", 64)),
            "d_model": int(cfg["model"].get("d_model", 256)),
        }
        print(
            f"cfg={cfg_log} | epoch={epoch:02d} "
            f"| train_total={logs['loss_total']:.4f} "
            f"| train_brits={logs['loss_brits']:.4f} "
            f"| train_block={logs['loss_block']:.4f} "
            f"| scale={logs['scale']:.1f} "
            f"| lambda={logs['lambda']:.3f} "
            f"| buckets={logs['active_buckets']} "
            f"| val_buckets={val_buckets} "
            f"| val_real_block(avg@{report_targets})={val_mae:.4f}"
        )

        eligible_for_selection = epoch >= full_phase_start_epoch
        if eligible_for_selection:
            improved = (best_val - val_mae) > min_delta
            if improved or best_state is None:
                best_val = val_mae
                best_epoch = epoch
                best_state = copy.deepcopy(module.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(
                        f"Early stopping at epoch {epoch}; "
                        f"best epoch={best_epoch}, best val_real={best_val:.4f}"
                    )
                    break

    if best_state is not None:
        module.load_state_dict(best_state)

    save_dir = Path(train_cfg["saving_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "saits_dynamic_best.pt"
    torch.save(module.state_dict(), ckpt_path)
    print(f"Saved dynamic-trained SAITS weights to: {ckpt_path}")

    return model_obj


def train_precomputed_realistic(
    cfg: dict,
    attrs: list[str],
    train_bank: dict[tuple[str, str], dict[str, np.ndarray]],
    val_bank: dict[tuple[str, str], dict[str, np.ndarray]],
):
    train_cfg = cfg["train"]
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed + 123)

    buckets = list(train_cfg.get("realistic_train_buckets", ["typ", "mod", "sev"]))
    train_targets_cfg = train_cfg.get("realistic_train_targets")
    train_targets = list(attrs if train_targets_cfg is None else train_targets_cfg)
    report_targets_cfg = train_cfg.get("report_targets")
    report_targets = list(attrs if report_targets_cfg is None else report_targets_cfg)

    sample_key = next(iter(train_bank.keys()))
    n_steps = int(train_bank[sample_key]["X"].shape[1])
    n_features = int(train_bank[sample_key]["X"].shape[2])
    model_obj = build_model(cfg, n_steps=n_steps, n_features=n_features)
    module = _locate_torch_module(model_obj)

    device = _pick_device(cfg)
    module.to(device)
    module.train()

    opt = torch.optim.Adam(
        module.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    steps_cfg = train_cfg.get("steps_per_epoch")
    steps_per_epoch = max(1, train_bank[sample_key]["X"].shape[0] // batch_size) if steps_cfg is None else int(steps_cfg)
    patience = int(train_cfg.get("patience", 10))
    min_delta = float(train_cfg.get("min_delta", 0.0))
    ramp_frac = float(train_cfg.get("ramp_frac", 0.25))
    max_lambda = float(train_cfg.get("max_lambda", 0.5))
    use_weighted_block_objective = bool(train_cfg.get("use_weighted_block_objective", False))
    cfg_name = train_cfg.get("log_cfg_name", "SAITS")

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    full_phase_start_epoch = int(np.floor(0.66 * epochs)) + 1

    print("Precomputed realistic-mask training enabled.")
    print(f"  buckets: {buckets}")
    print(f"  train targets: {train_targets}")
    print(f"  val target set: {report_targets}")

    for epoch in range(1, epochs + 1):
        module.train()
        active_buckets, p_bucket = curriculum(epoch, epochs)
        active_buckets = [b for b in active_buckets if b in buckets]
        p_bucket = p_bucket[: len(active_buckets)]
        p_bucket = p_bucket / p_bucket.sum()
        lam = lambda_ramp(epoch, epochs, ramp_frac=ramp_frac, max_lambda=max_lambda)

        sum_total = 0.0
        sum_brits = 0.0
        sum_block = 0.0
        sum_block_pts = 0.0
        sum_scale = 0.0

        for step in range(steps_per_epoch):
            target = str(rng.choice(train_targets))
            target_idx = int(attrs.index(target))
            bucket = str(rng.choice(active_buckets, p=p_bucket))
            ds_np = train_bank[(bucket, target)]
            idx = rng.integers(0, ds_np["X"].shape[0], size=batch_size)
            batch_np = {k: v[idx] for k, v in ds_np.items() if isinstance(v, np.ndarray)}

            batch_t = _batch_to_torch(batch_np, device)
            inputs = _assemble_inputs(batch_t)

            opt.zero_grad()
            try:
                out = module(inputs, calc_criterion=True)
            except TypeError:
                try:
                    out = module(inputs, training=True)
                except TypeError:
                    out = module(inputs)
            loss_brits = _extract_loss(out)

            imputed = _extract_imputed(out)
            if imputed is None:
                loss_block = torch.zeros((), device=device, dtype=torch.float32)
                n_block = torch.zeros((), device=device, dtype=torch.float32)
            else:
                xori_target = torch.nan_to_num(batch_t["X_ori"][:, :, target_idx], nan=0.0)
                imputed_target = imputed[:, :, target_idx]
                mask_target = batch_t["indicating_mask"][:, :, target_idx]
                abs_err = torch.abs(imputed_target - xori_target) * mask_target
                n_block = mask_target.sum()
                loss_block = abs_err.sum() / (n_block + 1e-8)

            with torch.no_grad():
                scale = (loss_brits.detach() / (loss_block.detach() + 1e-8)).clamp(1.0, 2000.0)

            if use_weighted_block_objective:
                loss_total = loss_brits + lam * scale * loss_block
            else:
                loss_total = loss_brits

            loss_total.backward()
            opt.step()
            sum_total += float(loss_total.item())
            sum_brits += float(loss_brits.item())
            sum_block += float(loss_block.item())
            sum_block_pts += float(n_block.item())
            sum_scale += float(scale.item())

        val_buckets = validation_buckets_for_epoch(epoch, epochs)
        maes = []
        for bucket in val_buckets:
            for target in report_targets:
                ds = val_bank[(bucket, target)]
                imputed = model_obj.impute(ds)
                maes.append(_masked_mae(imputed, ds["X_ori"], ds["indicating_mask"]))
        val_mae = float(np.nanmean(maes))

        logs = {
            "loss_total": sum_total / max(1, steps_per_epoch),
            "loss_brits": sum_brits / max(1, steps_per_epoch),
            "loss_block": sum_block / max(1, steps_per_epoch),
            "scale": sum_scale / max(1, steps_per_epoch),
            "lambda": lam,
            "active_buckets": active_buckets,
            "avg_block_points_per_batch": sum_block_pts / max(1, steps_per_epoch),
        }
        cfg_log = {
            "name": cfg_name,
            "lr": float(train_cfg.get("lr", 1e-3)),
            "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
            "batch_size": int(train_cfg.get("batch_size", 64)),
            "d_model": int(cfg["model"].get("d_model", 256)),
        }
        print(
            f"cfg={cfg_log} | epoch={epoch:02d} "
            f"| train_total={logs['loss_total']:.4f} "
            f"| train_brits={logs['loss_brits']:.4f} "
            f"| train_block={logs['loss_block']:.4f} "
            f"| scale={logs['scale']:.1f} "
            f"| lambda={logs['lambda']:.3f} "
            f"| buckets={logs['active_buckets']} "
            f"| val_buckets={val_buckets} "
            f"| val_real_block(avg@{report_targets})={val_mae:.4f}"
        )

        eligible_for_selection = epoch >= full_phase_start_epoch
        if eligible_for_selection:
            improved = (best_val - val_mae) > min_delta
            if improved or best_state is None:
                best_val = val_mae
                best_epoch = epoch
                best_state = copy.deepcopy(module.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(
                        f"Early stopping at epoch {epoch}; "
                        f"best epoch={best_epoch}, best val_real={best_val:.4f}"
                    )
                    break

    if best_state is not None:
        module.load_state_dict(best_state)

    save_dir = Path(train_cfg["saving_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "saits_dynamic_best.pt"
    torch.save(module.state_dict(), ckpt_path)
    print(f"Saved precomputed-trained SAITS weights to: {ckpt_path}")

    return model_obj


def build_train_val_windows(cfg: dict, split, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    win_cfg = cfg["window"]
    seq_len = int(win_cfg["seq_len"])
    train_stride = int(win_cfg.get("train_stride", win_cfg.get("stride", 60)))
    eval_stride = int(win_cfg.get("eval_stride", train_stride))
    mean, std = compute_feature_stats(split.train, features)
    train_X = normalize_windows(make_windows(split.train, features, seq_len, train_stride), mean, std)
    val_X = normalize_windows(make_windows(split.val, features, seq_len, eval_stride), mean, std)
    if len(train_X) == 0 or len(val_X) == 0:
        raise ValueError("No windows for dynamic training. Adjust seq_len/stride.")
    return train_X, val_X
