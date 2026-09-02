from typing import Callable

import pandas as pd
import yfinance as yf

_registry: dict[str, Callable] = {}


def register_source(name: str, fn: Callable) -> None:
    """Register a data source. fn signature: (start: str, asset: dict) -> pd.DataFrame"""
    _registry[name] = fn


# Fetches are memoized for the life of the process. A full sweep runs one node at a
# time over the same handful of source sets, and without this a 65-node family means
# 65 identical downloads -- slow, and enough to get rate-limited.
_cache: dict[tuple, pd.DataFrame] = {}


def fetch(sources: list[str], start: str, asset: dict) -> pd.DataFrame:
    key = (tuple(sources), start, asset.get('ticker'), asset.get('interval'))
    if key in _cache:
        return _cache[key].copy()
    frames = [_registry[s](start=start, asset=asset) for s in sources]
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f.reindex(out.index, method='ffill'), how='left')
    _cache[key] = out
    return out.copy()


# ── helpers ───────────────────────────────────────────────────────────────────

def _clamp_intraday_start(start: str, interval: str) -> str:
    """Yahoo only serves hourly data for the trailing 730 days; requests reaching
    further back fail entirely, so clamp the start into the allowed window."""
    if not any(c in interval for c in ('h', 'm')):
        return start
    earliest = pd.Timestamp.now().normalize() - pd.Timedelta(days=728)
    return max(pd.Timestamp(start), earliest).strftime('%Y-%m-%d')


def _yfinance_ohlcv(ticker: str, interval: str, start: str) -> pd.DataFrame:
    start = _clamp_intraday_start(start, interval)
    df = yf.download(ticker, start=start, interval=interval, auto_adjust=False, progress=False)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    intraday = any(c in interval for c in ('h', 'm'))
    if not intraday:
        idx = idx.normalize()
    df.index = idx
    df.index.name = 'Date'
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)


def _yfinance_cross(ticker: str, col_name: str, interval: str, start: str) -> pd.DataFrame:
    """Fetch a single-column cross-asset series (VIX, yields, etc.)."""
    start = _clamp_intraday_start(start, interval)
    df = yf.download(ticker, start=start, interval=interval, auto_adjust=False, progress=False)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    # Fall back to daily if the requested interval returned nothing
    interval_used = interval
    if df.empty and interval not in ('1d', '1wk'):
        df = yf.download(ticker, start=start, interval='1d', auto_adjust=False, progress=False)
        if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        interval_used = '1d'
    if df.empty:
        return pd.DataFrame(columns=[col_name])
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    intraday = any(c in interval_used for c in ('h', 'm'))
    if not intraday:
        idx = idx.normalize()
    df.index = idx
    df.index.name = 'Date'
    return df[['Close']].rename(columns={'Close': col_name}).ffill()


# ── built-in sources ──────────────────────────────────────────────────────────

def _ohlcv(start: str, asset: dict) -> pd.DataFrame:
    return _yfinance_ohlcv(asset['ticker'], asset['interval'], start)

register_source('ohlcv', _ohlcv)


def _vix(start: str, asset: dict) -> pd.DataFrame:
    return _yfinance_cross('^VIX', 'vix', asset['interval'], start)

register_source('vix', _vix)


def _treasury(start: str, asset: dict) -> pd.DataFrame:
    return _yfinance_cross('^TNX', 'tnx', asset['interval'], start)

register_source('treasury', _treasury)


def _dxy(start: str, asset: dict) -> pd.DataFrame:
    df = yf.download('DX-Y.NYB', start=start, interval='1d', auto_adjust=False, progress=False)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df.index.name = 'Date'
    return df[['Close']].rename(columns={'Close': 'dxy'}).ffill()

register_source('dxy', _dxy)
