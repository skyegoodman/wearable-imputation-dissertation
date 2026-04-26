from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ======================================================================================
# Utilities
# ======================================================================================

def _compute_deltas(masks: np.ndarray) -> np.ndarray:
    """
    input masks: (T, D) float32 with 1=observed, 0=missing
    returns deltas: (T, D) float32 = timesteps since last observed
    """
    T, D = masks.shape
    deltas = np.zeros((T, D), dtype=np.float32)
    last = np.zeros(D, dtype=np.float32)

    for t in range(T):
        last = np.where(masks[t] > 0, 0.0, last + 1.0)
        deltas[t] = last

    return deltas


def _make_forward_fills(values_with_nans: np.ndarray) -> np.ndarray:
    """
    input values: (T, D) float32 with NaNs
    returns forward-filled (T, D) with no NaNs, where each point is replaced by the most recent known value;
    leading NaNs -> 0.0
    """
    return (
        pd.DataFrame(values_with_nans)
        .ffill()
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def _find_true_runs(x: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Consecutive True runs in a 1D boolean array.
    Returns list of (start, end, length) where end is exclusive.
    """
    x = np.asarray(x, dtype=bool)
    n = int(x.size)
    if n == 0:
        return []
    y = np.r_[False, x, False].astype(np.int8)
    starts = np.where(np.diff(y) == 1)[0]
    ends = np.where(np.diff(y) == -1)[0]
    runs = [(int(s), int(e), int(e - s)) for s, e in zip(starts, ends)]
    return runs


def build_mask_library_from_df(
    df: pd.DataFrame,
    attributes: list[str],
    group_col: str | None = None,
    quantiles: tuple[float, float, float] = (0.50, 0.75, 0.90),
    min_run: int = 1,
    max_runs_per_bin: int | None = None,
) -> dict:
    """
    Build an empirical "missing-run library" from real data.

    For each target feature X:
      - find real consecutive missing runs of X
      - store the full missingness mask slice for ALL features during that run: mask_run (L,D) bool (True=missing)
      - bin runs by X's run-length quantiles into typ/mod/sev/ext:
          typ: L <= Q50
          mod: Q50 < L <= Q75
          sev: Q75 < L <= Q90
          ext: L > Q90
    """
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

    if group_col is None:
        groups = [("__all__", df)]
    else:
        groups = list(df.groupby(group_col, sort=False))

    # pass 1: collect run lengths per target
    run_lengths: dict[str, list[int]] = {t: [] for t in attrs}
    for _, g in groups:
        m = g[attrs].isna().to_numpy()  # (N,D) True=missing
        for d, target in enumerate(attrs):
            runs = _find_true_runs(m[:, d])
            for s, e, L in runs:
                if L >= min_run:
                    run_lengths[target].append(int(L))

    # quantiles per target
    quant_by_target: dict[str, dict[str, int]] = {}
    for target in attrs:
        lens = np.asarray(run_lengths[target], dtype=int)
        if lens.size == 0:
            quant_by_target[target] = {"Q50": 0, "Q75": 0, "Q90": 0}
        else:
            Q50 = int(np.quantile(lens, q50))
            Q75 = int(np.quantile(lens, q75))
            Q90 = int(np.quantile(lens, q90))
            quant_by_target[target] = {"Q50": Q50, "Q75": Q75, "Q90": Q90}

    # pass 2: store mask slices
    library: dict[str, dict] = {t: {b: [] for b in bins} for t in attrs}
    for t in attrs:
        library[t]["_quantiles"] = quant_by_target[t]

    for _, g in groups:
        m = g[attrs].isna().to_numpy()
        for d, target in enumerate(attrs):
            Q50 = quant_by_target[target]["Q50"]
            Q75 = quant_by_target[target]["Q75"]
            Q90 = quant_by_target[target]["Q90"]
            runs = _find_true_runs(m[:, d])
            for s, e, L in runs:
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
                    keep_idx = rng.choice(len(lst), size=max_runs_per_bin, replace=False)
                    library[t][b] = [lst[i] for i in keep_idx]

    return library


# ======================================================================================
# Dataset
# ======================================================================================

class WearableWindowSet(Dataset):
    """
    BRITS-style window records from a dataframe.

    Key change vs your previous version:
      - realistic_block placement no longer requires target to be observed at *every* timestep in the block.
        Instead it requires at least min_target_gt_frac (global or per-target override).
      - eval_masks on target are set ONLY where ground truth exists (observed==True).
      - helper feasibility for realistic_block is based on mask-template consistency:
        among helper points that the sampled mask expects to be present, at least
        min_helper_template_match_frac must actually be observed in the eval window.
      - returns batch-level meta info to support "coverage reports".
    """

    def __init__(
        self,
        df: pd.DataFrame,
        attributes: list[str],
        mean: np.ndarray,
        std: np.ndarray,
        seq_len: int = 360,
        stride: int = 60,
        p_point_holdout: float = 0.10,
        seed: int = 0,
        is_train_flag: int = 1,
        deterministic_holdout: bool = True,

        # holdout modes
        holdout_mode: str = "random_point",
        holdout_target: str | None = None,
        cond_avail: dict | None = None,
        target_only_eval: bool = True,

        # realistic block controls
        holdout_bucket: str = "typ",  # "typ"|"mod"|"sev"|"ext"
        mask_library: dict | None = None,
        max_resample_tries: int = 25,

        # DEPRECATED: retained for backward compatibility; no longer used in placement checks
        min_avg_helpers_observed: float | None = None,
        min_frac_timesteps_with_any_helper: float | None = None,

        # DEPRECATED: retained for backward compatibility; no longer used in placement checks
        min_avg_helpers_observed_by_target: dict[str, float] | None = None,
        min_frac_timesteps_with_any_helper_by_target: dict[str, float] | None = None,

        # target ground-truth fraction required inside the block
        min_target_gt_frac: float = 0.90,
        min_target_gt_frac_by_target: dict[str, float] | None = None,

        # helper feasibility based on sampled mask-template "present" helper points
        min_helper_template_match_frac: float = 0.80,

        # optional: if no mask_library, allow gap length sampler
        gap_len_sampler=None,  # callable(target, rng, T)->L

        # optional: stratify start positions by activity proxy
        stratify_starts: bool = True,
        include_tod: bool = True,
    ):
        super().__init__()
        self.df = df.sort_index()
        self.attributes = list(attributes)

        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

        self.seq_len = int(seq_len)
        self.stride = int(stride)

        self.p_point_holdout = float(p_point_holdout)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.is_train_flag = float(is_train_flag)
        self.deterministic_holdout = bool(deterministic_holdout)

        self.holdout_mode = str(holdout_mode)
        self.holdout_target = holdout_target
        self.cond_avail = cond_avail or {}
        self.target_only_eval = bool(target_only_eval)

        self.holdout_bucket = str(holdout_bucket)
        self.mask_library = mask_library
        self.max_resample_tries = int(max_resample_tries)

        self.min_avg_helpers_observed = min_avg_helpers_observed
        self.min_frac_timesteps_with_any_helper = min_frac_timesteps_with_any_helper
        self.min_avg_helpers_observed_by_target = min_avg_helpers_observed_by_target or {}
        self.min_frac_timesteps_with_any_helper_by_target = min_frac_timesteps_with_any_helper_by_target or {}

        self.min_target_gt_frac = float(min_target_gt_frac)
        self.min_target_gt_frac_by_target = min_target_gt_frac_by_target or {}
        self.min_helper_template_match_frac = float(min_helper_template_match_frac)

        self.gap_len_sampler = gap_len_sampler
        self.stratify_starts = bool(stratify_starts)
        self.include_tod = bool(include_tod)

        n = len(self.df)
        self.starts = list(range(0, max(0, n - self.seq_len + 1), self.stride))

    def __len__(self):
        return len(self.starts)

    def _time_of_day_channels(self, index: pd.Index, T: int) -> np.ndarray:
        """
        Return (T,2) raw time-of-day channels in [-1, 1]:
          tod_sin = sin(2*pi*minute_of_day/1440)
          tod_cos = cos(2*pi*minute_of_day/1440)
        """
        if isinstance(index, pd.DatetimeIndex):
            minute_of_day = (index.hour * 60 + index.minute).to_numpy(dtype=np.float32)
        else:
            # Fallback when index is not datetime-like.
            minute_of_day = (np.arange(T, dtype=np.float32) % 1440.0)

        phase = (2.0 * np.pi * minute_of_day) / 1440.0
        tod_sin = np.sin(phase).astype(np.float32)
        tod_cos = np.cos(phase).astype(np.float32)
        return np.stack([tod_sin, tod_cos], axis=1).astype(np.float32)

    def _pack_direction(
        self,
        values: np.ndarray,
        masks: np.ndarray,
        deltas: np.ndarray,
        evals: np.ndarray,
        eval_masks: np.ndarray,
        forwards: np.ndarray,
    ):
        return {
            "values": torch.from_numpy(values.astype(np.float32)),
            "masks": torch.from_numpy(masks.astype(np.float32)),
            "deltas": torch.from_numpy(deltas.astype(np.float32)),
            # BRITS code often expects evals numeric; keep NaNs as 0 but rely on eval_masks for scoring
            "evals": torch.from_numpy(np.nan_to_num(evals, nan=0.0).astype(np.float32)),
            "eval_masks": torch.from_numpy(eval_masks.astype(np.float32)),
            "forwards": torch.from_numpy(np.nan_to_num(forwards, nan=0.0).astype(np.float32)),
        }

    # ----------------------------------------------------------------------------------
    # Realistic point holdout (existing)
    # ----------------------------------------------------------------------------------
    def _make_realistic_point_holdout(
        self,
        rng: np.random.Generator,
        evals: np.ndarray,
        observed: np.ndarray,
        target: str,
        p_target: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        T, D = evals.shape
        holdout = np.zeros((T, D), dtype=bool)
        eval_masks = np.zeros((T, D), dtype=np.float32)

        if target not in self.attributes:
            raise ValueError(f"Target '{target}' not in attributes list: {self.attributes}")

        y = self.attributes.index(target)

        target_obs = observed[:, y]
        pick = (rng.random(T) < p_target) & target_obs
        holdout[pick, y] = True

        avail_row = self.cond_avail.get(target, {})
        n_pick = int(pick.sum())

        for j, feat in enumerate(self.attributes):
            if j == y or n_pick == 0:
                continue

            a = float(avail_row.get(feat, 1.0))
            p_drop = 1.0 - a

            if p_drop <= 0.0:
                continue
            if p_drop >= 1.0:
                holdout[pick, j] = True
            else:
                holdout[pick, j] = (rng.random(n_pick) < p_drop)

        if self.target_only_eval:
            eval_masks[pick, y] = 1.0
        else:
            eval_masks[holdout & observed] = 1.0

        return holdout, eval_masks

    # ----------------------------------------------------------------------------------
    # Realistic block holdout (UPDATED)
    # ----------------------------------------------------------------------------------
    def _activity_proxy(self, evals: np.ndarray, observed: np.ndarray) -> np.ndarray:
        """
        Activity proxy for start stratification.
        Uses absolute steps_rate if available; else abs diff(hr).
        evals are normalized values with NaNs where missing.
        """
        T, D = evals.shape
        attrs = self.attributes

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

        return np.zeros(T, dtype=np.float32)

    def _pick_start_stratified(
        self,
        rng: np.random.Generator,
        valid_starts: np.ndarray,
        proxy: np.ndarray,
    ) -> int:
        idx = np.flatnonzero(valid_starts)
        if idx.size == 0:
            raise RuntimeError("No valid starts.")
        if not self.stratify_starts:
            return int(rng.choice(idx))

        scores = proxy[idx]
        if scores.size < 10:
            return int(rng.choice(idx))

        q1, q2 = np.quantile(scores, [0.33, 0.66])
        bin_choice = int(rng.integers(0, 3))
        if bin_choice == 0:
            cand = idx[scores <= q1]
        elif bin_choice == 1:
            cand = idx[(scores > q1) & (scores <= q2)]
        else:
            cand = idx[scores > q2]

        if cand.size == 0:
            cand = idx
        return int(rng.choice(cand))

    def _helper_template_match_frac(
        self,
        observed: np.ndarray,
        y: int,
        t0: int,
        t1: int,
        mask_pat: np.ndarray | None,
    ) -> float | None:
        """
        Compute helper template match fraction for a candidate block:
          - expected_present are helper points where sampled mask is NOT missing
          - match fraction is observed helper points among expected_present

        Returns:
          float in [0, 1] when computable, else None when undefined/not applicable.
        """
        if mask_pat is None:
            return None

        L = t1 - t0
        helper_expected_present = ~np.delete(mask_pat[:L, :], y, axis=1)  # (L, D-1), True=expected present
        denom = float(helper_expected_present.sum())
        if denom <= 0.0:
            return None

        helper_observed = np.delete(observed[t0:t1, :], y, axis=1)  # (L, D-1), True=observed in data
        match = helper_observed & helper_expected_present
        return float(match.sum()) / denom

    def _passes_mask_template_helper_match(
        self,
        observed: np.ndarray,
        y: int,
        t0: int,
        t1: int,
        mask_pat: np.ndarray | None,
    ) -> bool:
        """
        Feasibility based on sampled mask-template consistency for helper channels:
          - Let expected_present be helper points where sampled mask is NOT missing.
          - Require at least min_helper_template_match_frac of expected_present points
            to be observed in the actual window.

        If no empirical mask is used (mask_pat is None), pass this check.
        If expected_present has zero points, pass this check.
        """
        frac_match = self._helper_template_match_frac(
            observed=observed,
            y=y,
            t0=t0,
            t1=t1,
            mask_pat=mask_pat,
        )
        if frac_match is None:
            return True
        return frac_match >= self.min_helper_template_match_frac

    def _sample_real_run_pattern(
        self,
        rng: np.random.Generator,
        target: str,
        bucket: str,
        T: int,
    ) -> tuple[int, np.ndarray] | None:
        """
        Returns (L, mask_slice_missing) where mask_slice_missing is (L,D) bool True=missing,
        clipped to <=T if needed. If no candidates, returns None.
        """
        if self.mask_library is None:
            return None
        if target not in self.mask_library:
            return None
        bucket_dict = self.mask_library[target]
        if bucket not in bucket_dict or len(bucket_dict[bucket]) == 0:
            return None
        run = bucket_dict[bucket][int(rng.integers(0, len(bucket_dict[bucket])))]
        L = int(run["L"])
        mask = np.asarray(run["mask"], dtype=bool)
        if L > T:
            L = T
            mask = mask[:T, :]
        return L, mask

    def _min_gt_frac_for_target(self, target: str) -> float:
        return float(self.min_target_gt_frac_by_target.get(target, self.min_target_gt_frac))

    def _make_realistic_block_holdout(
        self,
        rng: np.random.Generator,
        evals: np.ndarray,
        observed: np.ndarray,
        target: str,
        bucket: str,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Create a realistic BLOCK holdout for one target feature:
          - sample (L, mask_pattern) from empirical run library (preferred)
          - choose t0 such that target has >= min_gt_frac observed inside block (so scoring is defined)
          - paste mask pattern into the window: hold out target + helpers according to pattern
          - eval_masks is target-only by default and ONLY where ground truth exists

        Returns:
          holdout (T,D) bool
          eval_masks (T,D) float32
          meta dict containing placement stats for coverage reporting
        """
        T, D = evals.shape
        holdout = np.zeros((T, D), dtype=bool)
        eval_masks = np.zeros((T, D), dtype=np.float32)

        if target not in self.attributes:
            raise ValueError(f"Target '{target}' not in attributes list: {self.attributes}")
        y = self.attributes.index(target)

        proxy = self._activity_proxy(evals, observed)
        min_gt_frac = self._min_gt_frac_for_target(target)
        # Try a few times to find a feasible paste location / pattern
        for attempt in range(self.max_resample_tries):
            sampled = self._sample_real_run_pattern(rng, target, bucket, T)
            if sampled is not None:
                L, mask_pat = sampled
            else:
                if self.gap_len_sampler is not None:
                    L = int(self.gap_len_sampler(target, rng, T))
                else:
                    L = max(1, int(min(30, T)))
                L = max(1, min(L, T))
                mask_pat = None

            # build valid starts by target gt fraction (NOT all())
            target_obs = observed[:, y]
            valid = np.zeros((T - L + 1,), dtype=bool)
            # fast vector-ish loop
            for t0 in range(0, T - L + 1):
                gt_frac = float(target_obs[t0:t0 + L].mean())
                valid[t0] = (gt_frac >= min_gt_frac)

            if not valid.any():
                continue

            t0 = self._pick_start_stratified(rng, valid, proxy)
            t1 = t0 + L

            # build candidate holdout
            cand_holdout = np.zeros((T, D), dtype=bool)
            cand_holdout[t0:t1, y] = True  # always hide target inside block

            if mask_pat is not None:
                helper_missing = mask_pat[:L, :].copy()  # True=missing
                helper_missing[:, y] = True
                cand_holdout[t0:t1, :] |= helper_missing
            else:
                avail_row = self.cond_avail.get(target, {})
                for j, feat in enumerate(self.attributes):
                    if j == y:
                        continue
                    a = float(avail_row.get(feat, 1.0))
                    p_drop = 1.0 - a
                    if p_drop <= 0.0:
                        continue
                    if p_drop >= 1.0:
                        cand_holdout[t0:t1, j] = True
                    else:
                        if rng.random() < p_drop:
                            cand_holdout[t0:t1, j] = True

            # helper feasibility check:
            # among helper points expected present by sampled mask, require sufficient observed match in data.
            if not self._passes_mask_template_helper_match(
                observed=observed,
                y=y,
                t0=t0,
                t1=t1,
                mask_pat=mask_pat,
            ):
                continue

            # accept
            holdout = cand_holdout

            helper_match_frac = self._helper_template_match_frac(
                observed=observed,
                y=y,
                t0=t0,
                t1=t1,
                mask_pat=mask_pat,
            )

            # IMPORTANT: score only where ground truth exists
            if self.target_only_eval:
                eval_masks[t0:t1, y] = observed[t0:t1, y].astype(np.float32)
            else:
                eval_masks[holdout & observed] = 1.0

            # meta for coverage
            gt_frac_actual = float(observed[t0:t1, y].mean())
            # effective helper obs inside block (after applying holdout)
            eff_obs = observed[t0:t1, :] & (~holdout[t0:t1, :])
            eff_helpers = np.delete(eff_obs, y, axis=1)
            n_helpers = eff_helpers.sum(axis=1).astype(np.float32)
            avg_helpers = float(n_helpers.mean()) if n_helpers.size else 0.0
            frac_any = float((n_helpers >= 1).mean()) if n_helpers.size else 0.0
            n_eval_pts = float(eval_masks[t0:t1, y].sum())

            meta = {
                "placed": True,
                "target": target,
                "bucket": bucket,
                "t0": int(t0),
                "t1": int(t1),
                "L": int(L),
                "gt_frac_required": float(min_gt_frac),
                "gt_frac_actual": gt_frac_actual,
                "avg_helpers_observed": avg_helpers,
                "frac_timesteps_with_any_helper": frac_any,
                "min_helper_template_match_frac": float(self.min_helper_template_match_frac),
                "helper_template_match_frac": (
                    float(helper_match_frac) if helper_match_frac is not None else None
                ),
                "n_eval_points": n_eval_pts,
                "attempt": int(attempt + 1),
            }
            return holdout, eval_masks, meta

        # failed to place anything
        meta = {
            "placed": False,
            "target": target,
            "bucket": bucket,
            "gt_frac_required": float(self._min_gt_frac_for_target(target)),
            "attempts": int(self.max_resample_tries),
        }
        return holdout, eval_masks, meta

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        w = self.df.iloc[s: s + self.seq_len][self.attributes].astype(np.float32)
        T = len(w)

        evals = w.to_numpy(dtype=np.float32)
        evals = (evals - self.mean) / (self.std + 1e-8)
        observed = ~np.isnan(evals)

        rng = np.random.default_rng(self.seed + idx) if self.deterministic_holdout else self.rng

        meta = {}
        # HOLDOUT + EVAL MASKS
        if self.holdout_mode == "random_point":
            if self.p_point_holdout > 0.0:
                holdout = (rng.random(evals.shape) < self.p_point_holdout) & observed
            else:
                holdout = np.zeros_like(observed, dtype=bool)
            eval_masks = holdout.astype(np.float32)
            meta = {"placed": bool(eval_masks.sum() > 0), "mode": "random_point"}

        elif self.holdout_mode == "realistic_point":
            if self.holdout_target is None:
                raise ValueError("holdout_target must be set when holdout_mode='realistic_point'.")
            holdout, eval_masks = self._make_realistic_point_holdout(
                rng=rng,
                evals=evals,
                observed=observed,
                target=self.holdout_target,
                p_target=self.p_point_holdout,
            )
            meta = {
                "placed": bool(eval_masks.sum() > 0),
                "mode": "realistic_point",
                "target": self.holdout_target,
            }

        elif self.holdout_mode == "realistic_block":
            if self.holdout_target is None:
                raise ValueError("holdout_target must be set when holdout_mode='realistic_block'.")
            holdout, eval_masks, meta = self._make_realistic_block_holdout(
                rng=rng,
                evals=evals,
                observed=observed,
                target=self.holdout_target,
                bucket=self.holdout_bucket,
            )
            meta["mode"] = "realistic_block"

        else:
            raise ValueError(f"Unknown holdout_mode: {self.holdout_mode}")

        # BUILD VALUES / MASKS
        values = evals.copy()
        values[holdout] = np.nan

        # Append raw time-of-day channels (not z-normalized).
        if self.include_tod:
            tod_channels = self._time_of_day_channels(index=w.index, T=T)  # (T,2)
            evals = np.concatenate([evals, tod_channels], axis=1)
            values = np.concatenate([values, tod_channels], axis=1)
            eval_masks = np.concatenate([eval_masks, np.zeros((T, 2), dtype=np.float32)], axis=1)

        masks = (~np.isnan(values)).astype(np.float32)
        values_in = np.nan_to_num(values, nan=0.0).astype(np.float32)

        deltas_f = _compute_deltas(masks)
        if self.include_tod:
            deltas_f[:, -2:] = 0.0
        forwards_f = _make_forward_fills(values)
        forward = self._pack_direction(values_in, masks, deltas_f, evals, eval_masks, forwards_f)

        # BACKWARD
        values_b = values[::-1].copy()
        masks_b = masks[::-1].copy()
        evals_b = evals[::-1].copy()
        eval_masks_b = eval_masks[::-1].copy()

        values_in_b = np.nan_to_num(values_b, nan=0.0).astype(np.float32)
        deltas_b = _compute_deltas(masks_b)
        if self.include_tod:
            deltas_b[:, -2:] = 0.0
        forwards_b = _make_forward_fills(values_b)

        backward = self._pack_direction(values_in_b, masks_b, deltas_b, evals_b, eval_masks_b, forwards_b)

        return {
            "forward": forward,
            "backward": backward,
            "labels": torch.tensor(0.0, dtype=torch.float32),
            "is_train": torch.tensor(self.is_train_flag, dtype=torch.float32),
            # meta is NOT moved to device; use for coverage reporting
            "meta": meta,
        }


# ======================================================================================
# DataLoader wrapper
# ======================================================================================

def collate_fn(recs):
    def stack_dir(key):
        return {
            "values": torch.stack([r[key]["values"] for r in recs], dim=0),
            "masks": torch.stack([r[key]["masks"] for r in recs], dim=0),
            "deltas": torch.stack([r[key]["deltas"] for r in recs], dim=0),
            "evals": torch.stack([r[key]["evals"] for r in recs], dim=0),
            "eval_masks": torch.stack([r[key]["eval_masks"] for r in recs], dim=0),
            "forwards": torch.stack([r[key]["forwards"] for r in recs], dim=0),
        }

    batch = {
        "forward": stack_dir("forward"),
        "backward": stack_dir("backward"),
        "labels": torch.stack([r["labels"] for r in recs], dim=0),
        "is_train": torch.stack([r["is_train"] for r in recs], dim=0),
        # keep meta as a list of dicts (no device move)
        "meta": [r.get("meta", {}) for r in recs],
    }
    return batch


def get_loader_from_df(
    df: pd.DataFrame,
    attributes: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int = 360,
    stride: int = 60,
    batch_size: int = 32,
    shuffle: bool = True,
    p_point_holdout: float = 0.10,
    seed: int = 0,
    is_train_flag: int = 1,
    deterministic_holdout: bool = True,
    num_workers: int = 0,

    # masking config
    holdout_mode: str = "random_point",
    holdout_target: str | None = None,
    cond_avail: dict | None = None,
    target_only_eval: bool = True,

    # realistic block config
    holdout_bucket: str = "typ",
    mask_library: dict | None = None,
    max_resample_tries: int = 25,

    # helper feasibility (global defaults)
    # DEPRECATED: retained for backward compatibility; no longer used in placement checks
    min_avg_helpers_observed: float | None = None,
    min_frac_timesteps_with_any_helper: float | None = None,

    # per-target feasibility overrides (optional)
    # DEPRECATED: retained for backward compatibility; no longer used in placement checks
    min_avg_helpers_observed_by_target: dict[str, float] | None = None,
    min_frac_timesteps_with_any_helper_by_target: dict[str, float] | None = None,

    # target gt fraction (global + per-target override)
    min_target_gt_frac: float = 0.90,
    min_target_gt_frac_by_target: dict[str, float] | None = None,

    # helper template consistency threshold for realistic_block
    min_helper_template_match_frac: float = 0.80,

    gap_len_sampler=None,
    stratify_starts: bool = True,
):
    ds = WearableWindowSet(
        df=df,
        attributes=attributes,
        mean=mean,
        std=std,
        seq_len=seq_len,
        stride=stride,
        p_point_holdout=p_point_holdout,
        seed=seed,
        is_train_flag=is_train_flag,
        deterministic_holdout=deterministic_holdout,

        holdout_mode=holdout_mode,
        holdout_target=holdout_target,
        cond_avail=cond_avail,
        target_only_eval=target_only_eval,

        holdout_bucket=holdout_bucket,
        mask_library=mask_library,
        max_resample_tries=max_resample_tries,

        min_avg_helpers_observed=min_avg_helpers_observed,
        min_frac_timesteps_with_any_helper=min_frac_timesteps_with_any_helper,
        min_avg_helpers_observed_by_target=min_avg_helpers_observed_by_target,
        min_frac_timesteps_with_any_helper_by_target=min_frac_timesteps_with_any_helper_by_target,

        min_target_gt_frac=min_target_gt_frac,
        min_target_gt_frac_by_target=min_target_gt_frac_by_target,
        min_helper_template_match_frac=min_helper_template_match_frac,

        gap_len_sampler=gap_len_sampler,
        stratify_starts=stratify_starts,
        include_tod=True,
    )

    pin_memory = torch.cuda.is_available()

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


# ======================================================================================
# Precomputed deterministic window pipeline
# ======================================================================================

class PrecomputedWindowSet(Dataset):
    def __init__(self, records: list[dict]):
        super().__init__()
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        return self.records[idx]


def _minute_of_day_from_index(index: pd.Index) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        return (index.hour * 60 + index.minute).to_numpy(dtype=np.float32)
    return (np.arange(len(index), dtype=np.float32) % 1440.0)


def _design_matrix_24h(minute_of_day: np.ndarray) -> np.ndarray:
    t = np.asarray(minute_of_day, dtype=np.float64)
    ang = 2.0 * np.pi * t / 1440.0
    return np.column_stack(
        [
            np.ones_like(t, dtype=np.float64),
            np.sin(ang),
            np.cos(ang),
        ]
    )


def _fit_24h_harmonic(minute_of_day: np.ndarray, values: np.ndarray) -> np.ndarray:
    X = _design_matrix_24h(minute_of_day)
    beta, *_ = np.linalg.lstsq(X, values.astype(np.float64), rcond=None)
    return beta.astype(np.float64)


def _predict_24h_harmonic(minute_of_day: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return (_design_matrix_24h(minute_of_day) @ beta).astype(np.float32)


def _materialize_record_infos(ds: WearableWindowSet) -> list[dict]:
    infos = []
    for idx, s in enumerate(ds.starts):
        rec = ds[idx]
        window_index = ds.df.iloc[s: s + ds.seq_len].index
        start_ts = window_index[0] if len(window_index) else None
        infos.append(
            {
                "record": rec,
                "timestamps": window_index,
                "start_timestamp": start_ts,
                "start_idx": int(s),
                "dataset_index": int(idx),
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
    feature_idx = {feat: attrs.index(feat) for feat in harmonic_features}

    week_groups: dict[tuple[int, int], list[dict]] = {}
    for info in record_infos:
        ts = info["start_timestamp"]
        if isinstance(ts, pd.Timestamp):
            iso = ts.isocalendar()
            key = (int(iso.year), int(iso.week))
        else:
            key = (0, 0)
        week_groups.setdefault(key, []).append(info)

    sorted_weeks = sorted(week_groups.keys())
    coeffs_by_week: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    last_good: dict[str, np.ndarray] = {}

    for week_key in sorted_weeks:
        coeffs_by_week[week_key] = {}
        infos = week_groups[week_key]
        for feat in harmonic_features:
            d = feature_idx[feat]
            t_list = []
            y_list = []
            for info in infos:
                rec = info["record"]
                forward = rec["forward"]
                evals = forward["evals"].cpu().numpy()[:, : len(attrs)]
                masks = forward["masks"].cpu().numpy()[:, : len(attrs)]
                timestamps = info["timestamps"]

                raw = evals[:, d] * (float(std[d]) + 1e-8) + float(mean[d])
                obs = masks[:, d] > 0.5
                if obs.any():
                    minute_of_day = _minute_of_day_from_index(timestamps)[obs]
                    t_list.append(minute_of_day)
                    y_list.append(raw[obs])

            n_obs = int(sum(len(x) for x in y_list))
            beta = None
            if n_obs >= int(min_obs):
                t_obs = np.concatenate(t_list, axis=0)
                y_obs = np.concatenate(y_list, axis=0)
                beta = _fit_24h_harmonic(t_obs, y_obs)
                last_good[feat] = beta
            elif feat in last_good:
                beta = last_good[feat]
            elif n_obs >= 3:
                t_obs = np.concatenate(t_list, axis=0)
                y_obs = np.concatenate(y_list, axis=0)
                beta = _fit_24h_harmonic(t_obs, y_obs)
            else:
                beta = np.zeros(3, dtype=np.float64)

            coeffs_by_week[week_key][feat] = beta

        for info in infos:
            timestamps = info["timestamps"]
            minute_of_day_all = _minute_of_day_from_index(timestamps)
            priors = np.zeros((len(timestamps), len(harmonic_features)), dtype=np.float32)
            for j, feat in enumerate(harmonic_features):
                beta = coeffs_by_week[week_key][feat]
                priors[:, j] = _predict_24h_harmonic(minute_of_day_all, beta)
            info["harmonic_raw"] = priors
            info["week_key"] = week_key

    return record_infos


def _compute_harmonic_stats(record_infos: list[dict]) -> dict[str, np.ndarray]:
    mats = [info["harmonic_raw"] for info in record_infos if "harmonic_raw" in info]
    if len(mats) == 0:
        raise ValueError("No harmonic priors found when computing harmonic stats.")
    X = np.concatenate(mats, axis=0).astype(np.float32)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std <= 1e-8, 1.0, std).astype(np.float32)
    return {"mean": mean, "std": std}


def _append_auxiliary_channels(
    record_infos: list[dict],
    include_tod: bool,
    include_harmonic: bool,
    harmonic_stats: dict[str, np.ndarray] | None,
) -> list[dict]:
    out = []
    for info in record_infos:
        rec = info["record"]
        timestamps = info["timestamps"]

        forward = rec["forward"]
        backward = rec["backward"]

        aux_f = []
        if include_tod:
            minute_of_day = _minute_of_day_from_index(timestamps)
            phase = (2.0 * np.pi * minute_of_day) / 1440.0
            tod = np.stack([np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)
            aux_f.append(tod)

        if include_harmonic:
            if harmonic_stats is None:
                raise ValueError("harmonic_stats must be provided when include_harmonic=True.")
            harm_raw = info["harmonic_raw"].astype(np.float32)
            harm = (harm_raw - harmonic_stats["mean"]) / (harmonic_stats["std"] + 1e-8)
            aux_f.append(harm.astype(np.float32))

        if len(aux_f) == 0:
            out.append(rec)
            continue

        aux_f = np.concatenate(aux_f, axis=1).astype(np.float32)
        aux_b = aux_f[::-1].copy()

        def _augment_direction(direction: dict, aux_arr: np.ndarray) -> dict:
            values = direction["values"].cpu().numpy()
            masks = direction["masks"].cpu().numpy()
            deltas = direction["deltas"].cpu().numpy()
            evals = direction["evals"].cpu().numpy()
            eval_masks = direction["eval_masks"].cpu().numpy()
            forwards = direction["forwards"].cpu().numpy()

            aux_masks = np.ones_like(aux_arr, dtype=np.float32)
            aux_deltas = np.zeros_like(aux_arr, dtype=np.float32)
            aux_eval_masks = np.zeros_like(aux_arr, dtype=np.float32)

            values = np.concatenate([values, aux_arr], axis=1)
            masks = np.concatenate([masks, aux_masks], axis=1)
            deltas = np.concatenate([deltas, aux_deltas], axis=1)
            evals = np.concatenate([evals, aux_arr], axis=1)
            eval_masks = np.concatenate([eval_masks, aux_eval_masks], axis=1)
            forwards = np.concatenate([forwards, aux_arr], axis=1)

            return {
                "values": torch.from_numpy(values.astype(np.float32)),
                "masks": torch.from_numpy(masks.astype(np.float32)),
                "deltas": torch.from_numpy(deltas.astype(np.float32)),
                "evals": torch.from_numpy(evals.astype(np.float32)),
                "eval_masks": torch.from_numpy(eval_masks.astype(np.float32)),
                "forwards": torch.from_numpy(forwards.astype(np.float32)),
            }

        out.append(
            {
                "forward": _augment_direction(forward, aux_f),
                "backward": _augment_direction(backward, aux_b),
                "labels": rec["labels"],
                "is_train": rec["is_train"],
                "meta": rec.get("meta", {}),
            }
        )
    return out


def prepare_precomputed_windows_from_df(
    df: pd.DataFrame,
    attributes: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int = 360,
    stride: int = 60,
    p_point_holdout: float = 0.10,
    seed: int = 0,
    is_train_flag: int = 1,
    deterministic_holdout: bool = True,
    holdout_mode: str = "random_point",
    holdout_target: str | None = None,
    cond_avail: dict | None = None,
    target_only_eval: bool = True,
    holdout_bucket: str = "typ",
    mask_library: dict | None = None,
    max_resample_tries: int = 25,
    min_target_gt_frac: float = 0.90,
    min_target_gt_frac_by_target: dict[str, float] | None = None,
    min_helper_template_match_frac: float = 0.80,
    gap_len_sampler=None,
    stratify_starts: bool = True,
    include_tod: bool = False,
    include_harmonic: bool = False,
    harmonic_features: list[str] | None = None,
    harmonic_min_obs: int = 100,
    harmonic_stats: dict[str, np.ndarray] | None = None,
):
    base_ds = WearableWindowSet(
        df=df,
        attributes=attributes,
        mean=mean,
        std=std,
        seq_len=seq_len,
        stride=stride,
        p_point_holdout=p_point_holdout,
        seed=seed,
        is_train_flag=is_train_flag,
        deterministic_holdout=deterministic_holdout,
        holdout_mode=holdout_mode,
        holdout_target=holdout_target,
        cond_avail=cond_avail,
        target_only_eval=target_only_eval,
        holdout_bucket=holdout_bucket,
        mask_library=mask_library,
        max_resample_tries=max_resample_tries,
        min_target_gt_frac=min_target_gt_frac,
        min_target_gt_frac_by_target=min_target_gt_frac_by_target,
        min_helper_template_match_frac=min_helper_template_match_frac,
        gap_len_sampler=gap_len_sampler,
        stratify_starts=stratify_starts,
        include_tod=False,
    )

    record_infos = _materialize_record_infos(base_ds)

    out_harmonic_stats = harmonic_stats
    if include_harmonic:
        harmonic_features = list(harmonic_features or [])
        if len(harmonic_features) == 0:
            raise ValueError("harmonic_features must be provided when include_harmonic=True.")
        record_infos = _fit_weekly_24h_harmonics(
            record_infos=record_infos,
            attributes=attributes,
            mean=np.asarray(mean, dtype=np.float32),
            std=np.asarray(std, dtype=np.float32),
            harmonic_features=harmonic_features,
            min_obs=harmonic_min_obs,
        )
        if out_harmonic_stats is None:
            out_harmonic_stats = _compute_harmonic_stats(record_infos)

    records = _append_auxiliary_channels(
        record_infos=record_infos,
        include_tod=include_tod,
        include_harmonic=include_harmonic,
        harmonic_stats=out_harmonic_stats,
    )
    return records, out_harmonic_stats


def get_loader_from_precomputed_records(
    records: list[dict],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
):
    ds = PrecomputedWindowSet(records)
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
