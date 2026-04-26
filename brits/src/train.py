from __future__ import annotations

import argparse
from pathlib import Path
import copy
import json

import numpy as np
import torch
import torch.optim as optim

from brits.models import baseline_brits
from brits.src.config import load_config
from brits.src.data import (
    build_block_record_bank,
    build_context,
    build_train_record_bank,
    input_size,
    make_loader_bank,
)
from brits.src.utils import move_batch_to_device, set_seed


def curriculum(epoch: int, max_epochs: int) -> tuple[list[str], np.ndarray]:
    frac = epoch / max_epochs
    if frac <= 0.33:
        return ['typ'], np.array([1.0], dtype=float)
    if frac <= 0.66:
        return ['typ', 'mod'], np.array([0.6, 0.4], dtype=float)
    return ['typ', 'mod', 'sev'], np.array([0.25, 0.30, 0.45], dtype=float)


def validation_buckets_for_epoch(epoch: int, max_epochs: int) -> list[str]:
    return ['mod']


def lambda_ramp(epoch: int, max_epochs: int, ramp_frac: float = 0.25, max_lambda: float = 0.5) -> float:
    ramp_epochs = max(1, int(max_epochs * ramp_frac))
    if ramp_epochs <= 1 or epoch >= ramp_epochs:
        return float(max_lambda)
    return float(max_lambda * (epoch - 1) / max(1, ramp_epochs - 1))


def eval_imputation(model, loader, device: torch.device) -> float:
    model.eval()
    total_abs, total_n = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            ret = model.run_on_batch(batch, optimizer=None)
            abs_err = torch.abs(ret['imputations'] - ret['evals']) * ret['eval_masks']
            total_abs += float(abs_err.sum().item())
            total_n += float(ret['eval_masks'].sum().item())
    return total_abs / (total_n + 1e-8)


def eval_realistic_val_avg(model, val_loader_bank: dict, report_targets: list[str], buckets: list[str], device: torch.device) -> float:
    maes = []
    for bucket in buckets:
        for target in report_targets:
            key = (bucket, target)
            if key in val_loader_bank:
                maes.append(eval_imputation(model, val_loader_bank[key], device))
    return float(np.nanmean(maes))


def train_one_epoch_realistic(model, optimizer, loader_bank, attrs, train_targets, epoch, max_epochs, steps_per_epoch, rng, device, ramp_frac, max_lambda):
    model.train()
    active_buckets, p_bucket = curriculum(epoch, max_epochs)
    active_buckets = [b for b in active_buckets if any(k[0] == b for k in loader_bank)]
    p_bucket = p_bucket[: len(active_buckets)]
    p_bucket = p_bucket / p_bucket.sum()
    lam = lambda_ramp(epoch, max_epochs, ramp_frac=ramp_frac, max_lambda=max_lambda)
    iters = {(b, t): iter(loader_bank[(b, t)]) for b in active_buckets for t in train_targets if (b, t) in loader_bank}

    sum_total = 0.0
    sum_brits = 0.0
    sum_block = 0.0
    sum_scale = 0.0
    sum_block_pts = 0.0

    for _ in range(steps_per_epoch):
        bucket = str(rng.choice(active_buckets, p=p_bucket))
        target = str(rng.choice(train_targets))
        key = (bucket, target)
        if key not in iters:
            continue
        try:
            batch = next(iters[key])
        except StopIteration:
            iters[key] = iter(loader_bank[key])
            batch = next(iters[key])

        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        ret = model(batch)
        loss_brits = ret['loss']
        abs_err = torch.abs(ret['imputations'] - ret['evals']) * ret['eval_masks']
        n_block = ret['eval_masks'].sum()
        loss_block = abs_err.sum() / (n_block + 1e-8)
        with torch.no_grad():
            scale = (loss_brits.detach() / (loss_block.detach() + 1e-8)).clamp(1.0, 2000.0)
        loss_total = loss_brits + lam * scale * loss_block
        loss_total.backward()
        optimizer.step()

        sum_total += float(loss_total.item())
        sum_brits += float(loss_brits.item())
        sum_block += float(loss_block.item())
        sum_scale += float(scale.item())
        sum_block_pts += float(n_block.item())

    denom = max(1, steps_per_epoch)
    return {
        'loss_total': sum_total / denom,
        'loss_brits': sum_brits / denom,
        'loss_block': sum_block / denom,
        'scale': sum_scale / denom,
        'lambda': lam,
        'active_buckets': active_buckets,
        'avg_block_points_per_batch': sum_block_pts / denom,
    }


def fit_with_early_stopping(cfg, ctx, train_loader_bank, val_loader_bank, device):
    train_cfg = cfg['training']
    model_cfg = cfg['model']
    seed = int(cfg.get('seed', 42))
    set_seed(seed)

    model = baseline_brits.Model(
        rnn_hid_size=int(model_cfg.get('hid_size', 128)),
        impute_weight=float(model_cfg.get('impute_weight', 1.0)),
        label_weight=float(model_cfg.get('label_weight', 0.0)),
        input_size=input_size(ctx),
        sensor_size=len(ctx.attrs),
        decay_mode='original',
    ).to(device)
    model.rits_f.dropout.p = float(model_cfg.get('dropout', 0.25))
    model.rits_b.dropout.p = float(model_cfg.get('dropout', 0.25))

    opt = optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get('lr', 1e-3)),
        weight_decay=float(train_cfg.get('weight_decay', 0.0)),
    )
    rng = np.random.default_rng(seed + 123)

    epochs = int(train_cfg.get('epochs', 40))
    patience = int(train_cfg.get('patience', 5))
    min_delta = float(train_cfg.get('min_delta', 0.0))
    steps_per_epoch = int(train_cfg.get('steps_per_epoch', 400))
    train_targets = list(train_cfg.get('realistic_train_targets') or ctx.attrs)
    report_targets = list(train_cfg.get('report_targets') or ctx.attrs)
    ramp_frac = float(train_cfg.get('ramp_frac', 0.25))
    max_lambda = float(train_cfg.get('max_lambda', 0.5))
    full_phase_start_epoch = int(np.floor(0.66 * epochs)) + 1

    best_val = float('inf')
    best_epoch = 0
    best_state = None
    wait = 0
    last_val = float('nan')

    for epoch in range(1, epochs + 1):
        logs = train_one_epoch_realistic(
            model=model,
            optimizer=opt,
            loader_bank=train_loader_bank,
            attrs=ctx.attrs,
            train_targets=train_targets,
            epoch=epoch,
            max_epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            rng=rng,
            device=device,
            ramp_frac=ramp_frac,
            max_lambda=max_lambda,
        )
        val_buckets = validation_buckets_for_epoch(epoch, epochs)
        last_val = eval_realistic_val_avg(model, val_loader_bank, report_targets, val_buckets, device)
        print(
            f"epoch={epoch:02d} train_total={logs['loss_total']:.4f} "
            f"train_brits={logs['loss_brits']:.4f} train_block={logs['loss_block']:.4f} "
            f"lambda={logs['lambda']:.3f} buckets={logs['active_buckets']} "
            f"val_real_block(avg@{report_targets})={last_val:.4f}"
        )

        if epoch >= full_phase_start_epoch:
            improved = (best_val - last_val) > min_delta
            if improved or best_state is None:
                best_val = last_val
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f'Early stopping at epoch {epoch}; best epoch={best_epoch}, best val={best_val:.4f}')
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {'best_epoch': best_epoch, 'best_val_mae': best_val}


def save_artifacts(cfg, ctx, model, fit_info, harmonic_stats) -> None:
    output_dir = Path(cfg.get('outputs', {}).get('dir', 'brits/outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg.get('outputs', {}).get('run_name', 'brits_tod_harmonic')
    ckpt_path = output_dir / f'{run_name}_best.pt'
    meta_path = output_dir / f'{run_name}_meta.json'

    harmonic_payload = None
    if harmonic_stats is not None:
        harmonic_payload = {
            'mean': harmonic_stats['mean'].tolist(),
            'std': harmonic_stats['std'].tolist(),
        }

    payload = {
        'run_name': run_name,
        'config': cfg,
        'best_epoch': fit_info['best_epoch'],
        'best_val_mae': fit_info['best_val_mae'],
        'model_state_dict': model.state_dict(),
        'input_size': input_size(ctx),
        'sensor_size': len(ctx.attrs),
        'include_tod': bool(cfg['model'].get('include_tod', True)),
        'include_harmonic': bool(cfg['model'].get('include_harmonic', True)),
        'harmonic_features': ctx.harmonic_features,
        'harmonic_stats': harmonic_payload,
        'attrs': ctx.attrs,
    }
    torch.save(payload, ckpt_path)

    meta = {k: v for k, v in payload.items() if k != 'model_state_dict'}
    meta['checkpoint_path'] = str(ckpt_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Saved checkpoint: {ckpt_path}')
    print(f'Saved metadata:   {meta_path}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='brits/config.example.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get('seed', 42)))
    ctx = build_context(cfg)

    device = torch.device(str(cfg.get('training', {}).get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')))
    train_bank, harmonic_stats = build_train_record_bank(ctx)
    val_bank = build_block_record_bank(
        ctx,
        ctx.split.val,
        harmonic_stats=harmonic_stats,
        seed=int(cfg.get('seed', 42)) + 50000,
        buckets=list(cfg.get('training', {}).get('realistic_train_buckets', ['typ', 'mod', 'sev'])),
        targets=list(cfg.get('training', {}).get('report_targets') or ctx.attrs),
        is_train_flag=0,
    )
    train_loader_bank = make_loader_bank(train_bank, batch_size=int(cfg['training'].get('batch_size', 64)), shuffle=True)
    val_loader_bank = make_loader_bank(val_bank, batch_size=int(cfg.get('evaluation', {}).get('batch_size', 32)), shuffle=False)

    model, fit_info = fit_with_early_stopping(cfg, ctx, train_loader_bank, val_loader_bank, device)
    save_artifacts(cfg, ctx, model, fit_info, harmonic_stats)


if __name__ == '__main__':
    main()
