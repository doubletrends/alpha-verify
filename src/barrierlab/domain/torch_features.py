"""Torch feature kernels shared by CPU and CUDA numerical runs.

The input is ``[path, bar, open/high/low/close/volume]``.  A real history is
simply a one-path batch; validation uses all synthetic paths at once.
"""
from __future__ import annotations

import torch


def _rolling(x: torch.Tensor, window: int, op: str) -> torch.Tensor:
    out = torch.full_like(x, float("nan"))
    if window <= x.shape[1]:
        windows = x.unfold(1, window, 1)
        if op == "mean":
            values = windows.mean(-1)
        elif op == "max":
            values = windows.amax(-1)
        else:
            values = windows.std(-1, correction=1)
        out[:, window - 1:] = values
    return out


def rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    """Public rolling helper for non-OHLC numeric input columns."""
    return _rolling(x, window, "mean")


def _ema(x: torch.Tensor, span: int) -> torch.Tensor:
    """Pandas ``ewm(span=..., adjust=False)`` semantics, batched by path."""
    alpha = 2.0 / (span + 1.0)
    out = torch.empty_like(x)
    out[:, 0] = x[:, 0]
    for bar in range(1, x.shape[1]):
        out[:, bar] = alpha * x[:, bar] + (1.0 - alpha) * out[:, bar - 1]
    return out


def _returns(close: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(close, float("nan"))
    out[:, 1:] = torch.log(close[:, 1:] / close[:, :-1])
    return out


def compute(ohlcv: torch.Tensor, feature: str, params: dict, cache: dict | None = None) -> torch.Tensor:
    """Compute a feature, reusing shared rolling primitives when supplied."""
    cache = {} if cache is None else cache
    key = ("feature", feature, tuple(sorted(params.items())))
    if key not in cache:
        cache[key] = _compute(ohlcv, feature, params, cache)
    return cache[key]


def _compute(ohlcv: torch.Tensor, feature: str, params: dict, cache: dict) -> torch.Tensor:
    open_, high, low, close, volume = (ohlcv[:, :, index] for index in range(5))
    def remembered(key, build):
        if key not in cache:
            cache[key] = build()
        return cache[key]
    mean = lambda x, n: remembered(("mean", x.data_ptr(), n), lambda: _rolling(x, n, "mean"))
    maximum = lambda x, n: remembered(("max", x.data_ptr(), n), lambda: _rolling(x, n, "max"))
    std = lambda x, n: remembered(("std", x.data_ptr(), n), lambda: _rolling(x, n, "std"))
    if feature == "constant":
        return torch.zeros_like(close)
    if feature == "ma_ratio":
        return close / mean(close, params["period"]) - 1.0
    if feature == "ma_cross":
        return mean(close, params["fast"]) / mean(close, params["slow"]) - 1.0
    if feature == "roc":
        period = params["period"]
        def build_roc():
            out = torch.full_like(close, float("nan"))
            out[:, period:] = close[:, period:] / close[:, :-period] - 1.0
            return out
        return remembered(("roc", period), build_roc)
    if feature == "roc_spread":
        fast, slow = params["fast"], params["slow"]
        fast_roc = compute(ohlcv, "roc", {"period": fast}, cache)
        slow_roc = compute(ohlcv, "roc", {"period": slow}, cache)
        return fast_roc - slow_roc
    if feature in {"drawdown", "drawdown_recovery"}:
        first = close / maximum(close, params["period"] if feature == "drawdown" else params["short"]) - 1.0
        return first if feature == "drawdown" else first - (close / maximum(close, params["long"]) - 1.0)
    if feature in {"rsi", "rsi_spread"}:
        delta = remembered(("close_delta",), lambda: torch.diff(close, dim=1, prepend=close[:, :1]))
        gains = remembered(("close_gains",), lambda: delta.clamp_min(0))
        losses = remembered(("close_losses",), lambda: (-delta).clamp_min(0))
        def rsi(period: int) -> torch.Tensor:
            return 100.0 - 100.0 / (1.0 + mean(gains, period) / mean(losses, period))
        return rsi(params["period"]) if feature == "rsi" else rsi(params["fast"]) - rsi(params["slow"])
    if feature in {"macd", "macd_histogram"}:
        fast_ema = remembered(("ema_close", params["fast"]), lambda: _ema(close, params["fast"]))
        slow_ema = remembered(("ema_close", params["slow"]), lambda: _ema(close, params["slow"]))
        line = remembered(
            ("macd_line", params["fast"], params["slow"]),
            lambda: (fast_ema - slow_ema) / close,
        )
        return line if feature == "macd" else line - _ema(line, params.get("signal", 9))
    if feature in {"realized_vol", "vol_ratio"}:
        returns = remembered(("returns",), lambda: _returns(close))
        if feature == "realized_vol": return std(returns, params["period"]) * (252.0 ** 0.5) * 100.0
        return std(returns, params["fast"]) / std(returns, params["slow"])
    if feature == "atr":
        previous = torch.cat((close[:, :1], close[:, :-1]), dim=1)
        tr = remembered(("true_range",), lambda: torch.stack(
            (high - low, (high - previous).abs(), (low - previous).abs()), -1,
        ).amax(-1))
        return mean(tr, params["period"]) / close
    if feature in {"bb_pct", "bb_width"}:
        average, deviation = mean(close, params["period"]), std(close, params["period"]); multiple = params.get("std_dev", 2.0)
        return (close - (average - multiple * deviation)) / (2 * multiple * deviation) if feature == "bb_pct" else 2 * multiple * deviation / average
    if feature == "volume_ratio":
        return volume / mean(volume, params["period"])
    if feature in {"stoch_k", "stoch_d", "williams_r", "wr_spread"}:
        def bounds(period: int):
            highest = remembered(("rolling_high", period), lambda: maximum(high, period))
            lowest = remembered(("rolling_low", period), lambda: -maximum(-low, period))
            return highest, lowest
        def williams(period: int) -> torch.Tensor:
            highest, lowest = bounds(period)
            return (close - highest) / (highest - lowest) + 1.0
        def stoch(period: int) -> torch.Tensor:
            highest, lowest = bounds(period)
            return remembered(
                ("stoch", period), lambda: 100.0 * (close - lowest) / (highest - lowest),
            )
        if feature == "stoch_k": return stoch(params["k_period"])
        if feature == "stoch_d": return mean(stoch(params["k_period"]), params.get("d_period", 3))
        if feature == "williams_r": return williams(params["period"])
        return williams(params["fast"]) - williams(params["slow"])
    raise ValueError(f"Torch feature not implemented: {feature}")
