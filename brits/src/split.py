from __future__ import annotations

import pandas as pd


class TimeSplitTVT:
    """Time-respecting train/validation/test split container."""

    def __init__(self, train, val, test, split_time, val_start_time):
        self.train = train
        self.val = val
        self.test = test
        self.split_time = split_time
        self.val_start_time = val_start_time


def time_split_first_months_with_val(
    df: pd.DataFrame,
    months_train: int = 4,
    val_fraction: float = 0.1,
    min_train_rows: int = 1000,
    min_val_rows: int = 200,
) -> TimeSplitTVT:
    """
    Split a DatetimeIndex dataframe by time.

    The first ``months_train`` months are used for train+validation. The last
    ``val_fraction`` of that period is validation; everything after that period
    is test.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('df must have a DatetimeIndex')
    if not (0.0 < val_fraction < 0.5):
        raise ValueError('val_fraction should be between 0 and 0.5')

    df = df.sort_index()
    start_time = df.index.min()
    split_time = start_time + pd.DateOffset(months=int(months_train))

    trainval = df.loc[df.index < split_time].copy()
    test = df.loc[df.index >= split_time].copy()

    if len(trainval) < int(min_train_rows):
        raise ValueError(f'Train+val split too small: {len(trainval)} rows')

    cut = int((1.0 - float(val_fraction)) * len(trainval))
    if cut <= 0 or cut >= len(trainval):
        raise ValueError('val_fraction resulted in an empty train or validation split')

    train = trainval.iloc[:cut].copy()
    val = trainval.iloc[cut:].copy()

    if len(val) < int(min_val_rows):
        raise ValueError(f'Validation split too small: {len(val)} rows')

    return TimeSplitTVT(train, val, test, split_time, val.index.min())
