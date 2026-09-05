from pathlib import Path
from typing import Callable

import pandas as pd
import yfinance as yf


def _configure_yfinance_cache() -> None:
    """Keep yfinance's writable databases with the current pipeline workspace."""
    cache_path = Path.cwd() / ".yfinance-cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_path))

class SourceRegistry:
    """Explicit source registry and per-run market-data cache."""

    def __init__(self) -> None:
        self._sources: dict[str, Callable] = {}
        self._cache: dict[tuple, pd.DataFrame] = {}

    def register(self, name: str, source: Callable) -> None:
        self._sources[name] = source

    def fetch(self, sources: list[str], start: str, asset: dict) -> pd.DataFrame:
        key = (tuple(sources), start, asset.get("ticker"), asset.get("interval"))
        if key in self._cache:
            return self._cache[key].copy()
        missing = sorted(set(sources) - self._sources.keys())
        if missing:
            raise ValueError(f"unknown data sources: {', '.join(missing)}")
        frames = [self._sources[source](start=start, asset=asset) for source in sources]
        if not frames:
            raise ValueError("at least one data source is required")
        data = frames[0]
        for frame in frames[1:]:
            data = data.join(frame.reindex(data.index, method="ffill"), how="left")
        self._cache[key] = data
        return data.copy()


# ── helpers ───────────────────────────────────────────────────────────────────

def _clamp_intraday_start(start: str, interval: str) -> str:
    """Yahoo only serves hourly data for the trailing 730 days; requests reaching
    further back fail entirely, so clamp the start into the allowed window."""
    if not any(c in interval for c in ('h', 'm')):
        return start
    earliest = pd.Timestamp.now().normalize() - pd.Timedelta(days=728)
    return max(pd.Timestamp(start), earliest).strftime('%Y-%m-%d')


def _yfinance_ohlcv(ticker: str, interval: str, start: str) -> pd.DataFrame:
    _configure_yfinance_cache()
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

def _vix(start: str, asset: dict) -> pd.DataFrame:
    return _yfinance_cross('^VIX', 'vix', asset['interval'], start)

def _treasury(start: str, asset: dict) -> pd.DataFrame:
    return _yfinance_cross('^TNX', 'tnx', asset['interval'], start)

def _dxy(start: str, asset: dict) -> pd.DataFrame:
    df = yf.download('DX-Y.NYB', start=start, interval='1d', auto_adjust=False, progress=False)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df.index.name = 'Date'
    return df[['Close']].rename(columns={'Close': 'dxy'}).ffill()



def register_builtin_sources(registry: SourceRegistry) -> None:
    registry.register("ohlcv", _ohlcv)
    registry.register("vix", _vix)
    registry.register("treasury", _treasury)
    registry.register("dxy", _dxy)
