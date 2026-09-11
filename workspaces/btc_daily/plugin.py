"""BTC-specific on-chain and halving-cycle feature registrations."""

import pandas as pd
import torch

from alphaverify.domain import tensor_runtime, torch_features


def _torch_column(data: pd.DataFrame, column: str) -> torch.Tensor:
    return tensor_runtime.tensor(data[column].to_numpy(float))


def _torch_ratio(data: pd.DataFrame, column: str, period: int) -> torch.Tensor:
    values = _torch_column(data, column)
    logged = torch.where(values > 0, torch.log(values), torch.full_like(values, float('nan')))
    return logged / torch_features.rolling_mean(logged.unsqueeze(0), period).squeeze(0) - 1.0


# ── halving cycle helpers ─────────────────────────────────────────────────────

_HALVING_DATES = pd.to_datetime([
    '2012-11-28',
    '2016-07-09',
    '2020-05-11',
    '2024-04-20',
])
_NEXT_HALVING_EST = pd.Timestamp('2028-04-20')
_ALL_HALVINGS = list(_HALVING_DATES) + [_NEXT_HALVING_EST]


def _cycle_phase(data: pd.DataFrame) -> pd.Series:
    """Fractional position in the current halving cycle: 0 = just after halving, 1 = just before next."""
    dates = pd.to_datetime(data.index)
    phases = []
    for dt in dates:
        past   = [t for t in _ALL_HALVINGS if t <= dt]
        future = [t for t in _ALL_HALVINGS if t > dt]
        last   = max(past) if past else _ALL_HALVINGS[0]
        nxt    = min(future) if future else _NEXT_HALVING_EST
        phases.append((dt - last).days / max((nxt - last).days, 1))
    return pd.Series(phases, index=data.index, dtype=float)


def _days_to_halving(data: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(data.index)
    vals = []
    for dt in dates:
        future = [t for t in _ALL_HALVINGS if t > dt]
        nxt    = min(future) if future else _NEXT_HALVING_EST
        vals.append(max(0, (nxt - dt).days))
    return pd.Series(vals, index=data.index, dtype=float)


def _days_since_halving(data: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(data.index)
    days  = []
    for dt in dates:
        eligible = _HALVING_DATES[_HALVING_DATES <= dt]
        last     = eligible.max() if len(eligible) else _HALVING_DATES.min()
        days.append(max(0, (dt - last).days))
    return pd.Series(days, index=data.index, dtype=float)


def register(features) -> None:
    """Register BTC-only features for one pipeline run."""
    # The data-frame index remains the date-label boundary; all numeric feature
    # arithmetic below is on the pipeline's selected Torch device.
    features.register_torch('mvrv', lambda d, p: _torch_column(d, 'mvrv'))
    features.register_torch('hash_rate_ma_ratio', lambda d, p: _torch_ratio(d, 'hash_rate', p['period']))
    features.register_torch('adr_act_ma_ratio', lambda d, p: _torch_ratio(d, 'adr_act_cnt', p['period']))
    features.register_torch('cycle_phase', lambda d, p: tensor_runtime.tensor(_cycle_phase(d).to_numpy(float)))
    features.register_torch('days_to_halving', lambda d, p: tensor_runtime.tensor(_days_to_halving(d).to_numpy(float)))
    features.register_torch('days_since_halving', lambda d, p: tensor_runtime.tensor(_days_since_halving(d).to_numpy(float)))
