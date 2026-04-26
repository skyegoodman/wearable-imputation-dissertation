from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy

import numpy as np
import pandas as pd

from brits.src import data_loader
from brits.src.split import time_split_first_months_with_val


@dataclass
class BritsDataContext:
    cfg: dict
    attrs: list[str]
    harmonic_features: list[str]
    split: object
    means: np.ndarray
    stds: np.ndarray
    mask_lib: dict
    mask_lib_75_100: dict


def load_dataframe(input_path: str, datetime_col: str | None) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f'Input file not found: {path}')

    if path.suffix.lower() == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if datetime_col:
        if datetime_col not in df.columns:
            raise ValueError(f"datetime_col '{datetime_col}' not found in input data")
        df[datetime_col] = pd.to_datetime(df[datetime_col], utc=False)
        df = df.set_index(datetime_col)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('Input data must have a DatetimeIndex after loading.')
    return df.sort_index()


def build_context(cfg: dict) -> BritsDataContext:
    data_cfg = cfg['data']
    split_cfg = cfg.get('split', {})
    mask_cfg = cfg.get('realistic_masking', {})

    attrs = list(data_cfg['features'])
    harmonic_features = list(cfg.get('model', {}).get('harmonic_features', []))
    df = load_dataframe(data_cfg['input_path'], data_cfg.get('datetime_col'))

    missing = [c for c in attrs if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required feature columns: {missing}')

    split = time_split_first_months_with_val(
        df,
        months_train=int(split_cfg.get('months_train', 4)),
        val_fraction=float(split_cfg.get('val_fraction', 0.1)),
        min_train_rows=int(split_cfg.get('min_train_rows', 1000)),
        min_val_rows=int(split_cfg.get('min_val_rows', 200)),
    )
    means = split.train[attrs].mean(skipna=True).to_numpy(dtype=np.float32)
    stds = split.train[attrs].std(skipna=True).to_numpy(dtype=np.float32)
    stds = np.where((stds <= 0.0) | np.isnan(stds), 1.0, stds).astype(np.float32)

    q = tuple(float(x) for x in mask_cfg.get('quantiles', [0.50, 0.75, 0.90]))
    mask_lib = data_loader.build_mask_library_from_df(
        split.train,
        attributes=attrs,
        group_col=None,
        quantiles=(q[0], q[1], q[2]),
        min_run=int(mask_cfg.get('min_run', 1)),
        max_runs_per_bin=mask_cfg.get('max_runs_per_bin', 5000),
    )
    mask_lib_75_100 = copy.deepcopy(mask_lib)
    for target in attrs:
        mask_lib_75_100[target]['sev_75_100'] = list(mask_lib[target]['sev']) + list(mask_lib[target]['ext'])

    return BritsDataContext(
        cfg=cfg,
        attrs=attrs,
        harmonic_features=harmonic_features,
        split=split,
        means=means,
        stds=stds,
        mask_lib=mask_lib,
        mask_lib_75_100=mask_lib_75_100,
    )


def _target_gt_frac_by_target(ctx: BritsDataContext) -> dict[str, float]:
    mask_cfg = ctx.cfg.get('realistic_masking', {})
    default = float(mask_cfg.get('min_target_gt_frac', 0.80))
    overrides = dict(mask_cfg.get('min_target_gt_frac_by_target') or {})
    return {a: float(overrides.get(a, default)) for a in ctx.attrs}


def build_record_infos_from_df(
    ctx: BritsDataContext,
    df: pd.DataFrame,
    seq_len: int,
    stride: int,
    seed: int,
    is_train_flag: int,
    holdout_mode: str,
    holdout_target: str | None = None,
    holdout_bucket: str = 'typ',
    mask_library: dict | None = None,
    p_point_holdout: float = 0.0,
    include_harmonic: bool = False,
) -> list[dict]:
    mask_cfg = ctx.cfg.get('realistic_masking', {})
    ds = data_loader.WearableWindowSet(
        df=df,
        attributes=ctx.attrs,
        mean=ctx.means,
        std=ctx.stds,
        seq_len=seq_len,
        stride=stride,
        p_point_holdout=p_point_holdout,
        seed=seed,
        is_train_flag=is_train_flag,
        deterministic_holdout=True,
        holdout_mode=holdout_mode,
        holdout_target=holdout_target,
        cond_avail=None,
        target_only_eval=bool(mask_cfg.get('target_only_eval', True)),
        holdout_bucket=holdout_bucket,
        mask_library=mask_library,
        max_resample_tries=int(mask_cfg.get('max_resample_tries', 25)),
        min_target_gt_frac=float(mask_cfg.get('min_target_gt_frac', 0.80)),
        min_target_gt_frac_by_target=_target_gt_frac_by_target(ctx),
        min_helper_template_match_frac=float(mask_cfg.get('min_helper_template_match_frac', 0.80)),
        gap_len_sampler=None,
        stratify_starts=bool(mask_cfg.get('stratify_starts', True)),
        include_tod=False,
    )
    infos = data_loader._materialize_record_infos(ds)
    if include_harmonic:
        infos = data_loader._fit_weekly_24h_harmonics(
            record_infos=infos,
            attributes=ctx.attrs,
            mean=ctx.means,
            std=ctx.stds,
            harmonic_features=ctx.harmonic_features,
            min_obs=int(ctx.cfg.get('model', {}).get('harmonic_min_obs', 100)),
        )
    return infos


def finalize_records(
    record_infos: list[dict],
    include_tod: bool,
    include_harmonic: bool,
    harmonic_stats: dict[str, np.ndarray] | None,
) -> list[dict]:
    return data_loader._append_auxiliary_channels(
        record_infos=record_infos,
        include_tod=include_tod,
        include_harmonic=include_harmonic,
        harmonic_stats=harmonic_stats,
    )


def input_size(ctx: BritsDataContext) -> int:
    model_cfg = ctx.cfg.get('model', {})
    n = len(ctx.attrs)
    if bool(model_cfg.get('include_tod', True)):
        n += 2
    if bool(model_cfg.get('include_harmonic', True)):
        n += len(ctx.harmonic_features)
    return n


def build_train_record_bank(ctx: BritsDataContext) -> tuple[dict[tuple[str, str], list[dict]], dict | None]:
    cfg = ctx.cfg
    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('training', {})
    win_cfg = cfg.get('window', {})

    include_tod = bool(model_cfg.get('include_tod', True))
    include_harmonic = bool(model_cfg.get('include_harmonic', True))
    buckets = list(train_cfg.get('realistic_train_buckets', ['typ', 'mod', 'sev']))
    targets = list(train_cfg.get('realistic_train_targets') or ctx.attrs)
    seq_len = int(win_cfg.get('seq_len', 360))
    stride = int(win_cfg.get('train_stride', 60))

    info_bank = {}
    running_seed = int(cfg.get('seed', 42)) + 1000
    for bucket in buckets:
        for target in targets:
            info_bank[(bucket, target)] = build_record_infos_from_df(
                ctx,
                ctx.split.train,
                seq_len=seq_len,
                stride=stride,
                seed=running_seed,
                is_train_flag=1,
                holdout_mode='realistic_block',
                holdout_target=target,
                holdout_bucket=bucket,
                mask_library=ctx.mask_lib,
                p_point_holdout=0.0,
                include_harmonic=include_harmonic,
            )
            running_seed += 1000

    harmonic_stats = None
    if include_harmonic:
        all_infos = []
        for infos in info_bank.values():
            all_infos.extend(infos)
        harmonic_stats = data_loader._compute_harmonic_stats(all_infos)

    record_bank = {
        key: finalize_records(infos, include_tod, include_harmonic, harmonic_stats)
        for key, infos in info_bank.items()
    }
    return record_bank, harmonic_stats


def build_block_record_bank(
    ctx: BritsDataContext,
    df: pd.DataFrame,
    harmonic_stats: dict | None,
    seed: int,
    buckets: list[str],
    targets: list[str] | None = None,
    is_train_flag: int = 0,
) -> dict[tuple[str, str], list[dict]]:
    cfg = ctx.cfg
    model_cfg = cfg.get('model', {})
    win_cfg = cfg.get('window', {})
    include_tod = bool(model_cfg.get('include_tod', True))
    include_harmonic = bool(model_cfg.get('include_harmonic', True))
    seq_len = int(win_cfg.get('seq_len', 360))
    stride = int(win_cfg.get('eval_stride', seq_len))
    targets = list(targets or ctx.attrs)

    out = {}
    for b_idx, bucket in enumerate(buckets):
        mask_lib = ctx.mask_lib_75_100 if bucket == 'sev_75_100' else ctx.mask_lib
        for t_idx, target in enumerate(targets):
            infos = build_record_infos_from_df(
                ctx,
                df,
                seq_len=seq_len,
                stride=stride,
                seed=int(seed) + 100 * b_idx + t_idx,
                is_train_flag=is_train_flag,
                holdout_mode='realistic_block',
                holdout_target=target,
                holdout_bucket=bucket,
                mask_library=mask_lib,
                include_harmonic=include_harmonic,
            )
            out[(bucket, target)] = finalize_records(infos, include_tod, include_harmonic, harmonic_stats)
    return out


def build_random_point_records(ctx: BritsDataContext, harmonic_stats: dict | None, seed: int, df: pd.DataFrame | None = None) -> list[dict]:
    cfg = ctx.cfg
    model_cfg = cfg.get('model', {})
    win_cfg = cfg.get('window', {})
    eval_cfg = cfg.get('evaluation', {})
    include_tod = bool(model_cfg.get('include_tod', True))
    include_harmonic = bool(model_cfg.get('include_harmonic', True))
    infos = build_record_infos_from_df(
        ctx,
        ctx.split.test if df is None else df,
        seq_len=int(win_cfg.get('seq_len', 360)),
        stride=int(win_cfg.get('eval_stride', win_cfg.get('seq_len', 360))),
        seed=seed,
        is_train_flag=0,
        holdout_mode='random_point',
        p_point_holdout=float(eval_cfg.get('eval_holdout_prob', 0.10)),
        include_harmonic=include_harmonic,
    )
    return finalize_records(infos, include_tod, include_harmonic, harmonic_stats)


def make_loader_bank(record_bank: dict, batch_size: int, shuffle: bool) -> dict:
    return {
        key: data_loader.get_loader_from_precomputed_records(records, batch_size=batch_size, shuffle=shuffle)
        for key, records in record_bank.items()
    }


def make_loader(records: list[dict], batch_size: int, shuffle: bool = False):
    return data_loader.get_loader_from_precomputed_records(records, batch_size=batch_size, shuffle=shuffle)
