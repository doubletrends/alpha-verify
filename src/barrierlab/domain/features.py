"""Feature registry: labelled pandas boundaries, one Torch numerical implementation."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch

from barrierlab.domain import tensor_runtime, torch_features

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
    """Explicit feature registry scoped to one pipeline run."""

    def __init__(self) -> None:
        self._features: dict[str, Callable | None] = {}
        self._torch_features: dict[str, Callable] = {}

    def register(self, name: str, feature: Callable | None) -> None:
        self._features[name] = feature

    def register_torch(self, name: str, feature: Callable) -> None:
        """Register an extension returning a one-dimensional device tensor."""
        # A Torch extension is also a valid public feature.  Registering that
        # name here avoids maintaining an unreachable pandas fallback solely to
        # satisfy ``compute``'s name validation.
        self._features[name] = None
        self._torch_features[name] = feature

    def compute(self, data: pd.DataFrame, feature: str, params: dict, cache=None) -> pd.Series:
        if feature not in self._features:
            raise ValueError(f"Unknown feature: '{feature}'. Available: {sorted(self._features)}")
        if feature in self._torch_features:
            values = self._torch_features[feature](data, params)
        elif feature in _TORCH_OHLCV_FEATURES:
            if cache is not None and ("ohlcv",) in cache:
                ohlcv = cache[("ohlcv",)]
            else:
                close = data['close'].to_numpy(float)
                open_ = data['open'].to_numpy(float) if 'open' in data else close
                high = data['high'].to_numpy(float) if 'high' in data else np.maximum(open_, close)
                low = data['low'].to_numpy(float) if 'low' in data else np.minimum(open_, close)
                volume = data['volume'].to_numpy(float) if 'volume' in data else np.ones(len(data))
                ohlcv = torch.stack(tuple(tensor_runtime.tensor(v) for v in (open_, high, low, close, volume)), dim=-1).unsqueeze(0)
                if cache is not None:
                    cache[("ohlcv",)] = ohlcv
            values = torch_features.compute(ohlcv, feature, params, cache).squeeze(0)
        elif feature in _TORCH_EXTERNAL_FEATURES:
            values = _external(data, feature, params)
        elif feature == 'day_of_week':
            values = tensor_runtime.tensor(pd.to_datetime(data.index).dayofweek.to_numpy(float))
        else:
            handler = self._features[feature]
            if handler is None:
                raise ValueError(f"Feature '{feature}' has no implementation")
            return handler(data, params).reindex(data.index)
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


def register_builtin_features(registry: FeatureRegistry) -> None:
    """Register names only: all built-in numerical work dispatches to Torch above."""
    for name in _BUILTINS:
        registry.register(name, None)
