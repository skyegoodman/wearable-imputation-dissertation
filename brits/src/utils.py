from __future__ import annotations

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    for direct in ['forward', 'backward']:
        for key, value in batch[direct].items():
            if torch.is_tensor(value):
                batch[direct][key] = value.to(device)
    for key in ['labels', 'is_train']:
        if key in batch and torch.is_tensor(batch[key]):
            batch[key] = batch[key].to(device)
    return batch


def safe_nanmean(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float('nan')


def safe_nanstd(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanstd(arr, ddof=1)) if np.sum(~np.isnan(arr)) > 1 else float('nan')
