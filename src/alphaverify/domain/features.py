"""Feature registry: labelled pandas boundaries, one Torch numerical implementation."""
from __future__ import annotations

from typing import Callable

import pandas as pd
import torch

from alphaverify.domain import barrier, tensor_runtime, torch_features

_TORCH_OHLCV_FEATURES = {
    'constant', 'rsi', 'rsi_spread', 'stoch_k', 'stoch_d', 'williams_r',
    'wr_spread', 'macd', 'macd_histogram', 'ma_ratio', 'ma_cross',
    'realized_vol', 'vol_ratio', 'atr', 'bb_pct', 'bb_width',
    'volume_ratio', 'drawdown', 'drawdown_recovery', 'roc', 'roc_spread',
}
_TORCH_EXTERNAL_FEATURES = {
    'dxy_ret', 'dxy_ma_ratio', 'vix_level', 'vix_ma_ratio', 'vix_ret',
    'tnx_level', 'tnx_ma_ratio', 'tnx_ret',
}
_BUILTINS = _TORCH_OHLCV_FEATURES | _TORCH_EXTERNAL_FEATURES | {'day_of_week'}


def is_ohlcv_feature(name: str) -> bool:
    """Whether a feature can be recomputed from an unlabeled synthetic OHLCV path.

    Workspace extensions may be registered as Torch features too, but that only
    describes their numerical implementation on real, labelled data.  It does
    not mean they can be evaluated against a synthetic path, which has no
    calendar index or external columns.
    """
    return name in _TORCH_OHLCV_FEATURES


class FeatureRegistry:
    """Built-in features plus the workspace extensions registered for one pipeline run."""

    def __init__(self) -> None:
        self._torch_features: dict[str, Callable] = {}

    def register_torch(self, name: str, feature: Callable) -> None:
        """Register an extension returning a one-dimensional device tensor."""
        self._torch_features[name] = feature

    def compute(self, data: pd.DataFrame, feature: str, params: dict, cache=None) -> pd.Series:
        if feature in self._torch_features:
            values = self._torch_features[feature](data, params)
        elif feature in _TORCH_OHLCV_FEATURES:
            if cache is not None and ("ohlcv",) in cache:
                ohlcv = cache[("ohlcv",)]
            else:
                ohlcv = barrier.ohlcv_tensor(data)
                if cache is not None:
                    cache[("ohlcv",)] = ohlcv
            values = torch_features.compute(ohlcv, feature, params, cache).squeeze(0)
        elif feature in _TORCH_EXTERNAL_FEATURES:
            values = _external(data, feature, params)
        elif feature == 'day_of_week':
            values = tensor_runtime.tensor(pd.to_datetime(data.index).dayofweek.to_numpy(float))
        else:
            available = sorted(_BUILTINS | self._torch_features.keys())
            raise ValueError(f"Unknown feature: '{feature}'. Available: {available}")
        return pd.Series(values.cpu().numpy(), index=data.index, dtype=float)


def _forward_fill(values: torch.Tensor) -> torch.Tensor:
    out = values.clone()
    for index in range(1, len(out)):
        out[index] = torch.where(torch.isfinite(out[index]), out[index], out[index - 1])
    return out


def _external(data: pd.DataFrame, feature: str, params: dict) -> torch.Tensor:
    column = 'dxy' if feature.startswith('dxy') else 'vix' if feature.startswith('vix') else 'tnx'
    values = _forward_fill(tensor_runtime.tensor(data[column].to_numpy(float)))
    if feature.endswith('_level'):
        return values
    if feature.endswith('_ma_ratio'):
        return values / torch_features.rolling_mean(values.unsqueeze(0), params['period']).squeeze(0) - 1.0
    period = params['period']
    out = torch.full_like(values, float('nan'))
    out[period:] = values[period:] / values[:-period] - 1.0
    return out
