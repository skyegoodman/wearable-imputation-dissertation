from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from brits.models import baseline_brits
from brits.src.config import load_config
from brits.src.data import build_block_record_bank, build_context, build_random_point_records, input_size, make_loader
from brits.src.train import eval_imputation
from brits.src.utils import move_batch_to_device, safe_nanmean, safe_nanstd, set_seed


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get('config')
    model_cfg = cfg.get('model', {}) if cfg else {}
    model = baseline_brits.Model(
        rnn_hid_size=int(model_cfg.get('hid_size', 128)),
        impute_weight=float(model_cfg.get('impute_weight', 1.0)),
        label_weight=float(model_cfg.get('label_weight', 0.0)),
        input_size=int(ckpt['input_size']),
        sensor_size=int(ckpt.get('sensor_size', 9)),
        decay_mode='original',
    ).to(device)
    model.rits_f.dropout.p = float(model_cfg.get('dropout', 0.25))
    model.rits_b.dropout.p = float(model_cfg.get('dropout', 0.25))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def eval_block_metrics(model, loader, target: str, attrs: list[str], device: torch.device) -> dict:
    model.eval()
    d = attrs.index(target)
    total_abs = 0.0
    total_n = 0.0
    block_lengths = []
    helpers_sum = 0.0
    block_steps = 0.0
    helper_idx = [i for i in range(len(attrs)) if i != d]

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            ret = model.run_on_batch(batch, optimizer=None)
            imput = ret['imputations'][:, :, d]
            evals = ret['evals'][:, :, d]
            eval_masks = ret['eval_masks'][:, :, d]
            masks = batch['forward']['masks'][:, :, : len(attrs)]
            err = torch.abs(imput - evals) * eval_masks
            total_abs += float(err.sum().item())
            total_n += float(eval_masks.sum().item())
            block = eval_masks > 0.5
            block_lengths.extend(block.sum(dim=1).cpu().numpy().astype(int).tolist())
            helper_obs = masks[:, :, helper_idx].sum(dim=-1)
            helpers_sum += float((helper_obs * block).sum().item())
            block_steps += float(block.sum().item())

    lens = np.asarray([x for x in block_lengths if x > 0], dtype=int)
    return {
        'mae': total_abs / (total_n + 1e-8),
        'L_mean': float(lens.mean()) if len(lens) else float('nan'),
        'L_median': float(np.median(lens)) if len(lens) else float('nan'),
        'avg_helpers_observed': helpers_sum / (block_steps + 1e-8) if block_steps > 0 else float('nan'),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='brits/config.example.yaml')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get('seed', 42)))
    ctx = build_context(cfg)
    device = torch.device(str(cfg.get('training', {}).get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')))

    run_name = cfg.get('outputs', {}).get('run_name', 'brits_tod_harmonic')
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(cfg.get('outputs', {}).get('dir', 'brits/outputs')) / f'{run_name}_best.pt'
    model, ckpt = load_model(checkpoint, device)

    harmonic_stats = None
    if ckpt.get('harmonic_stats') is not None:
        harmonic_stats = {
            'mean': np.asarray(ckpt['harmonic_stats']['mean'], dtype=np.float32),
            'std': np.asarray(ckpt['harmonic_stats']['std'], dtype=np.float32),
        }

    random_records = build_random_point_records(ctx, harmonic_stats=harmonic_stats, seed=int(cfg.get('seed', 42)) + 70000)
    random_loader = make_loader(random_records, batch_size=int(cfg.get('evaluation', {}).get('batch_size', 32)), shuffle=False)
    random_mae = eval_imputation(model, random_loader, device)
    print(f'Random-point test MAE: {random_mae:.4f}')

    buckets = list(cfg.get('evaluation', {}).get('buckets', ['typ', 'mod', 'sev']))
    test_bank = build_block_record_bank(
        ctx,
        ctx.split.test,
        harmonic_stats=harmonic_stats,
        seed=int(cfg.get('seed', 42)) + 80000,
        buckets=buckets,
        targets=ctx.attrs,
        is_train_flag=0,
    )

    rows = []
    for bucket in buckets:
        for target in ctx.attrs:
            loader = make_loader(test_bank[(bucket, target)], batch_size=int(cfg.get('evaluation', {}).get('batch_size', 32)), shuffle=False)
            rows.append({'target': target, 'bucket': bucket, **eval_block_metrics(model, loader, target, ctx.attrs, device)})

    df = pd.DataFrame(rows).sort_values(['target', 'bucket'])
    print('\nRealistic block test results:')
    print(df.to_string(index=False))
    print('\nAverage MAE by bucket:')
    for bucket in buckets:
        print(f"  {bucket}: {safe_nanmean(df[df['bucket'] == bucket]['mae']):.4f}")


if __name__ == '__main__':
    main()
