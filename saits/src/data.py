from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from saits.src.split import time_split_first_months_with_val


def load_dataframe(input_path: str, datetime_col: str | None) -> pd.DataFrame:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    if datetime_col is not None:
        if datetime_col not in df.columns:
            raise ValueError(f"datetime_col '{datetime_col}' not in dataframe columns")
        df[datetime_col] = pd.to_datetime(df[datetime_col], utc=False)
        df = df.set_index(datetime_col)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataframe index must be a DatetimeIndex after loading.")

    return df.sort_index()


def make_windows(df_part: pd.DataFrame, features: list[str], seq_len: int, stride: int) -> np.ndarray:
    arr = df_part[features].to_numpy(dtype=np.float32)
    n = len(arr)
    if n < seq_len:
        return np.empty((0, seq_len, len(features)), dtype=np.float32)

    windows = []
    for s in range(0, n - seq_len + 1, stride):
        windows.append(arr[s : s + seq_len])
    return np.asarray(windows, dtype=np.float32)


def compute_feature_stats(train_df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mean = train_df[features].mean(skipna=True).to_numpy(dtype=np.float32)
    std = train_df[features].std(skipna=True).to_numpy(dtype=np.float32)
    std = np.where((std <= 0.0) | np.isnan(std), 1.0, std).astype(np.float32)
    return mean, std


def normalize_windows(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean[None, None, :]) / (std[None, None, :] + 1e-8)).astype(np.float32)


def make_windows_with_timestamps(
    df_part: pd.DataFrame, features: list[str], seq_len: int, stride: int
) -> tuple[np.ndarray, list[pd.DatetimeIndex]]:
    arr = df_part[features].to_numpy(dtype=np.float32)
    idx = df_part.index
    n = len(arr)
    if n < seq_len:
        return np.empty((0, seq_len, len(features)), dtype=np.float32), []

    windows = []
    stamps: list[pd.DatetimeIndex] = []
    for s in range(0, n - seq_len + 1, stride):
        windows.append(arr[s : s + seq_len])
        stamps.append(idx[s : s + seq_len])
    return np.asarray(windows, dtype=np.float32), stamps


def _minute_of_day_from_timestamps(timestamps: pd.DatetimeIndex | np.ndarray) -> np.ndarray:
    ts = pd.DatetimeIndex(timestamps)
    return (ts.hour * 60 + ts.minute + ts.second / 60.0).to_numpy(dtype=np.float64)


def _design_matrix_24h(t_minutes: np.ndarray) -> np.ndarray:
    phase = (2.0 * np.pi * np.asarray(t_minutes, dtype=np.float64)) / 1440.0
    return np.column_stack(
        [
            np.ones_like(phase, dtype=np.float64),
            np.sin(phase),
            np.cos(phase),
        ]
    )


def _fit_24h_harmonic(t_minutes: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
    A = _design_matrix_24h(t_minutes)
    beta, *_ = np.linalg.lstsq(A, np.asarray(y_obs, dtype=np.float64), rcond=None)
    return beta.astype(np.float64)


def _predict_24h_harmonic(t_minutes: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return (_design_matrix_24h(t_minutes) @ np.asarray(beta, dtype=np.float64)).astype(np.float32)


def _find_true_runs(x: np.ndarray) -> list[tuple[int, int, int]]:
    x = np.asarray(x, dtype=bool)
    if x.size == 0:
        return []
    y = np.r_[False, x, False].astype(np.int8)
    starts = np.where(np.diff(y) == 1)[0]
    ends = np.where(np.diff(y) == -1)[0]
    return [(int(s), int(e), int(e - s)) for s, e in zip(starts, ends)]


def build_mask_library_from_df(
    df: pd.DataFrame,
    attributes: list[str],
    quantiles: tuple[float, float, float] = (0.50, 0.75, 0.90),
    min_run: int = 1,
    max_runs_per_bin: int | None = None,
) -> dict:
    q50, q75, q90 = quantiles
    attrs = list(attributes)
    bins = ("typ", "mod", "sev", "ext")

    def _bin_name(L: int, Q50: int, Q75: int, Q90: int) -> str:
        if L <= Q50:
            return "typ"
        if L <= Q75:
            return "mod"
        if L <= Q90:
            return "sev"
        return "ext"

    m = df[attrs].isna().to_numpy()  # True=missing

    run_lengths: dict[str, list[int]] = {t: [] for t in attrs}
    for d, target in enumerate(attrs):
        for _, _, L in _find_true_runs(m[:, d]):
            if L >= min_run:
                run_lengths[target].append(int(L))

    quant_by_target: dict[str, dict[str, int]] = {}
    for target in attrs:
        lens = np.asarray(run_lengths[target], dtype=int)
        if lens.size == 0:
            quant_by_target[target] = {"Q50": 0, "Q75": 0, "Q90": 0}
        else:
            quant_by_target[target] = {
                "Q50": int(np.quantile(lens, q50)),
                "Q75": int(np.quantile(lens, q75)),
                "Q90": int(np.quantile(lens, q90)),
            }

    library: dict[str, dict] = {t: {b: [] for b in bins} for t in attrs}
    for t in attrs:
        library[t]["_quantiles"] = quant_by_target[t]

    for d, target in enumerate(attrs):
        Q50 = quant_by_target[target]["Q50"]
        Q75 = quant_by_target[target]["Q75"]
        Q90 = quant_by_target[target]["Q90"]
        for s, e, L in _find_true_runs(m[:, d]):
            if L < min_run:
                continue
            b = _bin_name(int(L), Q50, Q75, Q90)
            mask_slice = m[s:e, :].copy()  # (L,D) True=missing
            library[target][b].append({"L": int(L), "mask": mask_slice})

    if max_runs_per_bin is not None:
        rng = np.random.default_rng(0)
        for t in attrs:
            for b in bins:
                lst = library[t][b]
                if len(lst) > max_runs_per_bin:
                    idx = rng.choice(len(lst), size=max_runs_per_bin, replace=False)
                    library[t][b] = [lst[i] for i in idx]

    return library


def apply_eval_holdout(X: np.ndarray, holdout_prob: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    observed = ~np.isnan(X)
    sampled = (rng.random(X.shape) < holdout_prob) & observed

    X_in = X.copy()
    X_in[sampled] = np.nan
    indicating_mask = sampled.astype(np.float32)

    return {
        "X": X_in,
        "X_ori": X.copy(),
        "indicating_mask": indicating_mask,
    }


def _activity_proxy(attrs: list[str], evals: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if "steps_rate" in attrs:
        k = attrs.index("steps_rate")
        x = evals[:, k].copy()
        x[~observed[:, k]] = np.nan
        return np.abs(np.nan_to_num(x, nan=0.0))
    if "hr" in attrs:
        k = attrs.index("hr")
        x = evals[:, k].copy()
        x[~observed[:, k]] = np.nan
        x = np.nan_to_num(x, nan=0.0)
        dx = np.diff(x, prepend=x[0])
        return np.abs(dx)
    return np.zeros(evals.shape[0], dtype=np.float32)


def _pick_start_stratified(rng: np.random.Generator, valid_starts: np.ndarray, proxy: np.ndarray, stratify_starts: bool) -> int:
    idx = np.flatnonzero(valid_starts)
    if idx.size == 0:
        raise RuntimeError("No valid starts.")
    if not stratify_starts:
        return int(rng.choice(idx))
    scores = proxy[idx]
    if scores.size < 10:
        return int(rng.choice(idx))
    q1, q2 = np.quantile(scores, [0.33, 0.66])
    b = int(rng.integers(0, 3))
    if b == 0:
        cand = idx[scores <= q1]
    elif b == 1:
        cand = idx[(scores > q1) & (scores <= q2)]
    else:
        cand = idx[scores > q2]
    if cand.size == 0:
        cand = idx
    return int(rng.choice(cand))


def _helper_template_match_frac(observed: np.ndarray, y: int, t0: int, t1: int, mask_pat: np.ndarray | None) -> float | None:
    if mask_pat is None:
        return None
    L = t1 - t0
    helper_expected_present = ~np.delete(mask_pat[:L, :], y, axis=1)
    denom = float(helper_expected_present.sum())
    if denom <= 0.0:
        return None
    helper_observed = np.delete(observed[t0:t1, :], y, axis=1)
    match = helper_observed & helper_expected_present
    return float(match.sum()) / denom


def _sample_real_run_pattern(
    rng: np.random.Generator,
    mask_library: dict | None,
    target: str,
    bucket: str,
    T: int,
) -> tuple[int, np.ndarray] | None:
    if mask_library is None or target not in mask_library:
        return None
    bucket_dict = mask_library[target]
    if bucket not in bucket_dict or len(bucket_dict[bucket]) == 0:
        return None
    run = bucket_dict[bucket][int(rng.integers(0, len(bucket_dict[bucket])))]
    L = int(run["L"])
    mask = np.asarray(run["mask"], dtype=bool)
    if L > T:
        L = T
        mask = mask[:T, :]
    return L, mask


def _make_realistic_block_holdout_one(
    evals: np.ndarray,
    attrs: list[str],
    target: str,
    bucket: str,
    mask_library: dict,
    seed: int,
    max_resample_tries: int,
    min_target_gt_frac: float,
    min_helper_template_match_frac: float,
    target_only_eval: bool,
    stratify_starts: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    observed = ~np.isnan(evals)
    T, D = evals.shape
    holdout = np.zeros((T, D), dtype=bool)
    eval_masks = np.zeros((T, D), dtype=np.float32)

    if target not in attrs:
        return holdout, eval_masks, {"placed": False, "L": 0, "avg_helpers_observed": 0.0}
    y = attrs.index(target)
    proxy = _activity_proxy(attrs, evals, observed)

    for _ in range(max_resample_tries):
        sampled = _sample_real_run_pattern(rng, mask_library, target, bucket, T)
        if sampled is None:
            return holdout, eval_masks, {"placed": False, "L": 0, "avg_helpers_observed": 0.0}
        L, mask_pat = sampled
        target_obs = observed[:, y]
        valid = np.zeros((T - L + 1,), dtype=bool)
        for t0 in range(0, T - L + 1):
            valid[t0] = float(target_obs[t0:t0 + L].mean()) >= min_target_gt_frac
        if not valid.any():
            continue

        t0 = _pick_start_stratified(rng, valid, proxy, stratify_starts)
        t1 = t0 + L
        cand_holdout = np.zeros((T, D), dtype=bool)
        cand_holdout[t0:t1, y] = True
        helper_missing = mask_pat[:L, :].copy()
        helper_missing[:, y] = True
        cand_holdout[t0:t1, :] |= helper_missing

        frac_match = _helper_template_match_frac(observed, y, t0, t1, mask_pat)
        if frac_match is not None and frac_match < min_helper_template_match_frac:
            continue

        holdout = cand_holdout
        if target_only_eval:
            eval_masks[t0:t1, y] = observed[t0:t1, y].astype(np.float32)
        else:
            eval_masks[holdout & observed] = 1.0
        eff_obs = observed[t0:t1, :] & (~holdout[t0:t1, :])
        helper_idx = [j for j in range(D) if j != y]
        if len(helper_idx) > 0:
            helper_obs = eff_obs[:, helper_idx].sum(axis=1).astype(np.float32)
            avg_helpers_observed = float(helper_obs.mean()) if helper_obs.size else 0.0
        else:
            avg_helpers_observed = 0.0
        meta = {
            "placed": True,
            "L": int(L),
            "avg_helpers_observed": avg_helpers_observed,
        }
        return holdout, eval_masks, meta

    return holdout, eval_masks, {"placed": False, "L": 0, "avg_helpers_observed": 0.0}


def make_realistic_eval_set(
    X: np.ndarray,
    attributes: list[str],
    mask_library: dict,
    holdout_target: str,
    bucket: str,
    seed: int,
    max_resample_tries: int,
    min_target_gt_frac: float,
    min_helper_template_match_frac: float,
    target_only_eval: bool,
    stratify_starts: bool,
    return_meta: bool = False,
) -> dict:
    X_in = X.copy()
    X_ori = X.copy()
    indicating_mask = np.zeros_like(X, dtype=np.float32)
    meta_rows = []

    for i in range(X.shape[0]):
        holdout, eval_masks, meta = _make_realistic_block_holdout_one(
            evals=X_ori[i],
            attrs=attributes,
            target=holdout_target,
            bucket=bucket,
            mask_library=mask_library,
            seed=seed + i,
            max_resample_tries=max_resample_tries,
            min_target_gt_frac=min_target_gt_frac,
            min_helper_template_match_frac=min_helper_template_match_frac,
            target_only_eval=target_only_eval,
            stratify_starts=stratify_starts,
        )
        X_in[i][holdout] = np.nan
        indicating_mask[i] = eval_masks.astype(np.float32)
        meta_rows.append(meta)

    out = {"X": X_in, "X_ori": X_ori, "indicating_mask": indicating_mask}
    if return_meta:
        out["meta"] = meta_rows
    return out


def materialize_realistic_window_infos(
    df_part: pd.DataFrame,
    features: list[str],
    seq_len: int,
    stride: int,
    mean: np.ndarray,
    std: np.ndarray,
    holdout_target: str,
    bucket: str,
    mask_library: dict,
    seed: int,
    max_resample_tries: int,
    min_target_gt_frac: float,
    min_helper_template_match_frac: float,
    target_only_eval: bool,
    stratify_starts: bool,
) -> list[dict]:
    windows_raw, timestamps = make_windows_with_timestamps(df_part, features, seq_len, stride)
    windows = normalize_windows(windows_raw, mean, std)
    infos: list[dict] = []

    for i in range(windows.shape[0]):
        X_ori = windows[i].copy()
        holdout, eval_masks, meta = _make_realistic_block_holdout_one(
            evals=X_ori,
            attrs=features,
            target=holdout_target,
            bucket=bucket,
            mask_library=mask_library,
            seed=seed + i,
            max_resample_tries=max_resample_tries,
            min_target_gt_frac=min_target_gt_frac,
            min_helper_template_match_frac=min_helper_template_match_frac,
            target_only_eval=target_only_eval,
            stratify_starts=stratify_starts,
        )
        X_in = X_ori.copy()
        X_in[holdout] = np.nan
        infos.append(
            {
                "X": X_in.astype(np.float32),
                "X_ori": X_ori.astype(np.float32),
                "indicating_mask": eval_masks.astype(np.float32),
                "timestamps": pd.DatetimeIndex(timestamps[i]),
                "start_timestamp": pd.Timestamp(timestamps[i][0]),
                "meta": meta,
            }
        )

    return infos


def _fit_weekly_24h_harmonics(
    record_infos: list[dict],
    attributes: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    harmonic_features: list[str],
    min_obs: int = 100,
) -> list[dict]:
    attrs = list(attributes)
    feat_idx = {feat: attrs.index(feat) for feat in harmonic_features}

    week_groups: dict[tuple[int, int], list[dict]] = {}
    for info in record_infos:
        iso = info["start_timestamp"].isocalendar()
        key = (int(iso.year), int(iso.week))
        week_groups.setdefault(key, []).append(info)

    last_good: dict[str, np.ndarray] = {}
    for week_key in sorted(week_groups.keys()):
        infos = week_groups[week_key]
        coeffs: dict[str, np.ndarray] = {}
        for feat in harmonic_features:
            d = feat_idx[feat]
            t_list = []
            y_list = []
            for info in infos:
                X_ori = info["X_ori"]
                X_in = info["X"]
                observed = ~np.isnan(X_in[:, d])
                if not np.any(observed):
                    continue
                raw = X_ori[:, d] * (float(std[d]) + 1e-8) + float(mean[d])
                t_minutes = _minute_of_day_from_timestamps(info["timestamps"])[observed]
                t_list.append(t_minutes)
                y_list.append(raw[observed])

            n_obs = int(sum(len(x) for x in y_list))
            beta = None
            if n_obs >= int(min_obs):
                beta = _fit_24h_harmonic(np.concatenate(t_list), np.concatenate(y_list))
                last_good[feat] = beta
            elif feat in last_good:
                beta = last_good[feat]
            elif n_obs >= 3:
                beta = _fit_24h_harmonic(np.concatenate(t_list), np.concatenate(y_list))
            else:
                beta = np.zeros(3, dtype=np.float64)
            coeffs[feat] = beta

        for info in infos:
            t_minutes_all = _minute_of_day_from_timestamps(info["timestamps"])
            harm = np.zeros((len(t_minutes_all), len(harmonic_features)), dtype=np.float32)
            for j, feat in enumerate(harmonic_features):
                harm[:, j] = _predict_24h_harmonic(t_minutes_all, coeffs[feat])
            info["harmonic_raw"] = harm
            info["week_key"] = week_key

    return record_infos


def compute_harmonic_stats(record_infos: list[dict]) -> dict[str, np.ndarray]:
    mats = [info["harmonic_raw"] for info in record_infos if "harmonic_raw" in info]
    if len(mats) == 0:
        raise ValueError("No harmonic priors found when computing harmonic stats.")
    X = np.concatenate(mats, axis=0).astype(np.float32)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std <= 1e-8, 1.0, std).astype(np.float32)
    return {"mean": mean, "std": std}


def append_auxiliary_channels(
    record_infos: list[dict],
    include_tod: bool,
    include_harmonic: bool,
    harmonic_stats: dict[str, np.ndarray] | None,
) -> list[dict]:
    out: list[dict] = []
    for info in record_infos:
        X = info["X"].copy()
        X_ori = info["X_ori"].copy()
        indicating_mask = info["indicating_mask"].copy()

        aux_parts = []
        if include_tod:
            t_minutes = _minute_of_day_from_timestamps(info["timestamps"])
            phase = (2.0 * np.pi * t_minutes) / 1440.0
            aux_parts.append(np.stack([np.sin(phase), np.cos(phase)], axis=1).astype(np.float32))

        if include_harmonic:
            if harmonic_stats is None:
                raise ValueError("harmonic_stats must be provided when include_harmonic=True.")
            harm_raw = info["harmonic_raw"].astype(np.float32)
            harm = (harm_raw - harmonic_stats["mean"]) / (harmonic_stats["std"] + 1e-8)
            aux_parts.append(harm.astype(np.float32))

        if aux_parts:
            aux = np.concatenate(aux_parts, axis=1).astype(np.float32)
            X = np.concatenate([X, aux], axis=1)
            X_ori = np.concatenate([X_ori, aux], axis=1)
            aux_ind = np.zeros((aux.shape[0], aux.shape[1]), dtype=np.float32)
            indicating_mask = np.concatenate([indicating_mask, aux_ind], axis=1)

        out.append(
            {
                "X": X.astype(np.float32),
                "X_ori": X_ori.astype(np.float32),
                "indicating_mask": indicating_mask.astype(np.float32),
                "timestamps": info["timestamps"],
                "start_timestamp": info["start_timestamp"],
                "meta": info.get("meta", {}),
                "week_key": info.get("week_key"),
            }
        )
    return out


def infos_to_dataset(record_infos: list[dict], include_meta: bool = False) -> dict:
    if len(record_infos) == 0:
        raise ValueError("No record infos available.")
    out = {
        "X": np.stack([info["X"] for info in record_infos]).astype(np.float32),
        "X_ori": np.stack([info["X_ori"] for info in record_infos]).astype(np.float32),
        "indicating_mask": np.stack([info["indicating_mask"] for info in record_infos]).astype(np.float32),
    }
    if include_meta:
        out["meta"] = [info.get("meta", {}) for info in record_infos]
        out["start_timestamps"] = [info["start_timestamp"] for info in record_infos]
    return out


def build_saits_datasets(cfg: dict) -> tuple[dict, dict, dict, object]:
    data_cfg = cfg["data"]
    win_cfg = cfg["window"]
    eval_cfg = cfg["eval"]
    seed = int(cfg["seed"])

    df = load_dataframe(
        input_path=data_cfg["input_path"],
        datetime_col=data_cfg.get("datetime_col"),
    )

    features = list(data_cfg["features"])
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features in dataframe: {missing}")

    split = time_split_first_months_with_val(
        df=df,
        months_train=int(data_cfg["months_train"]),
        val_fraction=float(data_cfg["val_fraction"]),
        min_train_rows=int(data_cfg["min_train_rows"]),
        min_val_rows=int(data_cfg["min_val_rows"]),
    )

    seq_len = int(win_cfg["seq_len"])
    train_stride = int(win_cfg.get("train_stride", win_cfg.get("stride", 60)))
    eval_stride = int(win_cfg.get("eval_stride", train_stride))
    mean, std = compute_feature_stats(split.train, features)

    train_X = normalize_windows(make_windows(split.train, features, seq_len, train_stride), mean, std)
    val_X = normalize_windows(make_windows(split.val, features, seq_len, eval_stride), mean, std)
    test_X = normalize_windows(make_windows(split.test, features, seq_len, eval_stride), mean, std)

    if train_X.shape[0] == 0:
        raise ValueError("No training windows produced. Adjust seq_len/stride.")
    if val_X.shape[0] == 0:
        raise ValueError("No validation windows produced. Adjust seq_len/stride.")
    if test_X.shape[0] == 0:
        raise ValueError("No test windows produced. Adjust seq_len/stride.")

    train_set = {"X": train_X}
    val_set = apply_eval_holdout(val_X, float(eval_cfg["eval_holdout_prob"]), seed + 1)
    test_set = apply_eval_holdout(test_X, float(eval_cfg["eval_holdout_prob"]), seed + 2)

    return train_set, val_set, test_set, split


def build_realistic_test_grid(cfg: dict, split: object) -> dict[tuple[str, str], dict]:
    data_cfg = cfg["data"]
    win_cfg = cfg["window"]
    eval_cfg = cfg["eval"]
    seed = int(cfg["seed"])
    features = list(data_cfg["features"])

    seq_len = int(win_cfg["seq_len"])
    eval_stride = int(win_cfg.get("eval_stride", win_cfg.get("train_stride", win_cfg.get("stride", 60))))
    mean, std = compute_feature_stats(split.train, features)
    test_X = normalize_windows(make_windows(split.test, features, seq_len, eval_stride), mean, std)

    q = tuple(float(x) for x in eval_cfg.get("quantiles", [0.50, 0.75, 0.90]))
    mask_library = build_mask_library_from_df(
        split.train,
        features,
        quantiles=(q[0], q[1], q[2]),
        min_run=int(eval_cfg.get("min_run", 1)),
        max_runs_per_bin=eval_cfg.get("max_runs_per_bin"),
    )
    mask_library_75_100 = {
        target: {
            **mask_library[target],
            "sev_75_100": list(mask_library[target]["sev"]) + list(mask_library[target]["ext"]),
        }
        for target in features
    }

    buckets = list(eval_cfg.get("buckets", ["typ", "mod", "sev"]))
    out: dict[tuple[str, str], dict] = {}
    for target in features:
        for bucket in buckets:
            use_mask_library = mask_library_75_100 if bucket == "sev_75_100" else mask_library
            out[(target, bucket)] = make_realistic_eval_set(
                X=test_X,
                attributes=features,
                mask_library=use_mask_library,
                holdout_target=target,
                bucket=bucket,
                seed=seed + (1000 * (features.index(target) + 1)) + (10 * (buckets.index(bucket) + 1)),
                max_resample_tries=int(eval_cfg.get("max_resample_tries", 25)),
                min_target_gt_frac=float(eval_cfg.get("min_target_gt_frac", 0.90)),
                min_helper_template_match_frac=float(eval_cfg.get("min_helper_template_match_frac", 0.80)),
                target_only_eval=bool(eval_cfg.get("target_only_eval", True)),
                stratify_starts=bool(eval_cfg.get("stratify_starts", True)),
            )
    return out


def build_realistic_test_grid_with_meta(
    cfg: dict, split: object
) -> dict[tuple[str, str], tuple[dict, list[dict]]]:
    data_cfg = cfg["data"]
    win_cfg = cfg["window"]
    eval_cfg = cfg["eval"]
    seed = int(cfg["seed"])
    features = list(data_cfg["features"])

    seq_len = int(win_cfg["seq_len"])
    eval_stride = int(win_cfg.get("eval_stride", win_cfg.get("train_stride", win_cfg.get("stride", 60))))
    mean, std = compute_feature_stats(split.train, features)
    test_X = normalize_windows(make_windows(split.test, features, seq_len, eval_stride), mean, std)

    q = tuple(float(x) for x in eval_cfg.get("quantiles", [0.50, 0.75, 0.90]))
    mask_library = build_mask_library_from_df(
        split.train,
        features,
        quantiles=(q[0], q[1], q[2]),
        min_run=int(eval_cfg.get("min_run", 1)),
        max_runs_per_bin=eval_cfg.get("max_runs_per_bin"),
    )
    mask_library_75_100 = {
        target: {
            **mask_library[target],
            "sev_75_100": list(mask_library[target]["sev"]) + list(mask_library[target]["ext"]),
        }
        for target in features
    }

    buckets = list(eval_cfg.get("buckets", ["typ", "mod", "sev"]))
    out: dict[tuple[str, str], tuple[dict, list[dict]]] = {}
    for target in features:
        for bucket in buckets:
            use_mask_library = mask_library_75_100 if bucket == "sev_75_100" else mask_library
            ds = make_realistic_eval_set(
                X=test_X,
                attributes=features,
                mask_library=use_mask_library,
                holdout_target=target,
                bucket=bucket,
                seed=seed + (1000 * (features.index(target) + 1)) + (10 * (buckets.index(bucket) + 1)),
                max_resample_tries=int(eval_cfg.get("max_resample_tries", 25)),
                min_target_gt_frac=float(eval_cfg.get("min_target_gt_frac", 0.90)),
                min_helper_template_match_frac=float(eval_cfg.get("min_helper_template_match_frac", 0.80)),
                target_only_eval=bool(eval_cfg.get("target_only_eval", True)),
                stratify_starts=bool(eval_cfg.get("stratify_starts", True)),
                return_meta=True,
            )
            meta = ds.pop("meta")
            out[(target, bucket)] = (ds, meta)
    return out
