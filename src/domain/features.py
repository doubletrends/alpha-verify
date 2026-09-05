import numpy as np
import pandas as pd
from typing import Callable

class FeatureRegistry:
    """Explicit feature registry scoped to a pipeline run."""

    def __init__(self) -> None:
        self._features: dict[str, Callable] = {}

    def register(self, name: str, feature: Callable) -> None:
        self._features[name] = feature

    def compute(self, data: pd.DataFrame, feature: str, params: dict) -> pd.Series:
        if feature not in self._features:
            raise ValueError(f"Unknown feature: '{feature}'. Available: {sorted(self._features)}")
        return self._features[feature](data, params)


# ── private helpers ──────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    rs    = up.rolling(period).mean() / down.rolling(period).mean()
    return 100.0 - (100.0 / (1.0 + rs))


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    hh = high.rolling(period).max()
    ll  = low.rolling(period).min()
    return ((close - hh) / (hh - ll) + 1.0)   # normalized to [0, 1]; 1=oversold, 0=overbought


def _wr_spread(high: pd.Series, low: pd.Series, close: pd.Series, fast: int, slow: int) -> pd.Series:
    return _williams_r(high, low, close, fast) - _williams_r(high, low, close, slow)


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int) -> pd.Series:
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    return 100.0 * (close - ll) / (hh - ll)


def _macd(close: pd.Series, fast: int, slow: int) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow) / close   # normalized by price


def _macd_histogram(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    line = _macd(close, fast, slow)
    return line - line.ewm(span=signal, adjust=False).mean()


def _ma_ratio(close: pd.Series, period: int) -> pd.Series:
    return close / close.rolling(period).mean() - 1.0


def _ma_cross(close: pd.Series, fast: int, slow: int) -> pd.Series:
    return close.rolling(fast).mean() / close.rolling(slow).mean() - 1.0


def _realized_vol(close: pd.Series, period: int) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(252) * 100


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean() / close   # normalized by price


def _bb_pct(close: pd.Series, period: int, std_dev: float = 2.0) -> pd.Series:
    ma  = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (close - (ma - std_dev * std)) / (2 * std_dev * std)


def _bb_width(close: pd.Series, period: int, std_dev: float = 2.0) -> pd.Series:
    ma  = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (2 * std_dev * std) / ma


def _volume_ratio(volume: pd.Series, period: int) -> pd.Series:
    return volume / volume.rolling(period).mean()


def _vol_ratio(close: pd.Series, fast: int, slow: int) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    fast_vol = log_ret.rolling(fast).std()
    slow_vol = log_ret.rolling(slow).std()
    return fast_vol / slow_vol.replace(0, np.nan)


def _drawdown(close: pd.Series, period: int) -> pd.Series:
    return close / close.rolling(period).max() - 1.0


def _drawdown_recovery(close: pd.Series, short: int, long: int) -> pd.Series:
    return _drawdown(close, short) - _drawdown(close, long)


def _rsi_spread(close: pd.Series, fast: int, slow: int) -> pd.Series:
    return _rsi(close, fast) - _rsi(close, slow)


def _stoch_d(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, d_period: int = 3) -> pd.Series:
    return _stoch_k(high, low, close, k_period).rolling(d_period).mean()


def _roc(close: pd.Series, period: int) -> pd.Series:
    return close.pct_change(period)


def _roc_spread(close: pd.Series, fast: int, slow: int) -> pd.Series:
    return close.pct_change(fast) - close.pct_change(slow)


def _dxy_ret(dxy: pd.Series, period: int) -> pd.Series:
    return dxy.pct_change(period)


def _dxy_ma_ratio(dxy: pd.Series, period: int) -> pd.Series:
    dxy = dxy.ffill()  # fill weekend gaps so rolling window sees consecutive values
    return dxy / dxy.rolling(period).mean() - 1.0


def _day_of_week(data: pd.DataFrame) -> pd.Series:
    return pd.Series(pd.to_datetime(data.index).dayofweek, index=data.index, dtype=float)


def _vix_level(data: pd.DataFrame) -> pd.Series:
    return data['vix']


def _vix_ma_ratio(data: pd.DataFrame, period: int) -> pd.Series:
    v = data['vix'].ffill()
    return v / v.rolling(period).mean() - 1.0


def _vix_ret(data: pd.DataFrame, period: int) -> pd.Series:
    return data['vix'].ffill().pct_change(period)


def _tnx_level(data: pd.DataFrame) -> pd.Series:
    return data['tnx']


def _tnx_ma_ratio(data: pd.DataFrame, period: int) -> pd.Series:
    t = data['tnx'].ffill()
    return t / t.rolling(period).mean() - 1.0


def _tnx_ret(data: pd.DataFrame, period: int) -> pd.Series:
    return data['tnx'].ffill().pct_change(period)


def _constant(data: pd.DataFrame) -> pd.Series:
    """
    A feature with no information, so every bar falls in a single bin.

    This is what turns the unconditional base rate into an ordinary node: measured
    through exactly the same machinery as any condition, it yields P(touch Δ in t)
    with nothing conditioned on. The engine then needs no special case for the base.
    """
    return pd.Series(0.0, index=data.index)


# ── built-in registrations ────────────────────────────────────────────────────

def register_builtin_features(registry: FeatureRegistry) -> None:
    registry.register('constant',          lambda d, p: _constant(d))
    registry.register('rsi',               lambda d, p: _rsi(d['close'], p['period']))
    registry.register('rsi_spread',        lambda d, p: _rsi_spread(d['close'], p['fast'], p['slow']))
    registry.register('stoch_k',           lambda d, p: _stoch_k(d['high'], d['low'], d['close'], p['k_period']))
    registry.register('stoch_d',           lambda d, p: _stoch_d(d['high'], d['low'], d['close'], p['k_period'], p.get('d_period', 3)))
    registry.register('williams_r',        lambda d, p: _williams_r(d['high'], d['low'], d['close'], p['period']))
    registry.register('wr_spread',         lambda d, p: _wr_spread(d['high'], d['low'], d['close'], p['fast'], p['slow']))
    registry.register('macd',              lambda d, p: _macd(d['close'], p['fast'], p['slow']))
    registry.register('macd_histogram',    lambda d, p: _macd_histogram(d['close'], p['fast'], p['slow'], p.get('signal', 9)))
    registry.register('ma_ratio',          lambda d, p: _ma_ratio(d['close'], p['period']))
    registry.register('ma_cross',          lambda d, p: _ma_cross(d['close'], p['fast'], p['slow']))
    registry.register('realized_vol',      lambda d, p: _realized_vol(d['close'], p['period']))
    registry.register('vol_ratio',         lambda d, p: _vol_ratio(d['close'], p['fast'], p['slow']))
    registry.register('atr',               lambda d, p: _atr(d['high'], d['low'], d['close'], p['period']))
    registry.register('bb_pct',            lambda d, p: _bb_pct(d['close'], p['period'], p.get('std_dev', 2.0)))
    registry.register('bb_width',          lambda d, p: _bb_width(d['close'], p['period'], p.get('std_dev', 2.0)))
    registry.register('volume_ratio',      lambda d, p: _volume_ratio(d['volume'], p['period']))
    registry.register('drawdown',          lambda d, p: _drawdown(d['close'], p['period']))
    registry.register('drawdown_recovery', lambda d, p: _drawdown_recovery(d['close'], p['short'], p['long']))
    registry.register('roc',               lambda d, p: _roc(d['close'], p['period']))
    registry.register('roc_spread',        lambda d, p: _roc_spread(d['close'], p['fast'], p['slow']))
    registry.register('dxy_ret',           lambda d, p: _dxy_ret(d['dxy'], p['period']))
    registry.register('dxy_ma_ratio',      lambda d, p: _dxy_ma_ratio(d['dxy'], p['period']))
    registry.register('day_of_week',       lambda d, p: _day_of_week(d))
    registry.register('vix_level',         lambda d, p: _vix_level(d))
    registry.register('vix_ma_ratio',      lambda d, p: _vix_ma_ratio(d, p['period']))
    registry.register('vix_ret',           lambda d, p: _vix_ret(d, p['period']))
    registry.register('tnx_level',         lambda d, p: _tnx_level(d))
    registry.register('tnx_ma_ratio',      lambda d, p: _tnx_ma_ratio(d, p['period']))
    registry.register('tnx_ret',           lambda d, p: _tnx_ret(d, p['period']))
