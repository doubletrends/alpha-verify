"""Provider-independent boundary for workspace-prepared market data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaverify.infrastructure.workspace_plugins import load_workspace_module


def validate_market_data(data: pd.DataFrame, *, require_full_ohlcv: bool = True) -> None:
    """Reject malformed input without sorting, filling, dropping, or repairing it."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("workspace data must be a nonempty DataFrame")
    if (not isinstance(data.index, pd.DatetimeIndex) or data.index.hasnans
            or not data.index.is_unique or not data.index.is_monotonic_increasing):
        raise ValueError("workspace data needs unique, increasing, nonmissing timestamps")
    if data.index.tz is not None:
        raise ValueError("workspace must normalize timestamps to its declared naive time basis")
    required = ["open", "high", "low", "close", "volume"] if require_full_ohlcv else ["high", "low", "close"]
    if not data.columns.is_unique or any(column not in data for column in required):
        raise ValueError("workspace data needs unique columns including " + ", ".join(required))
    for column in data:
        if (not pd.api.types.is_numeric_dtype(data[column])
                or pd.api.types.is_bool_dtype(data[column])
                or pd.api.types.is_complex_dtype(data[column])):
            raise ValueError(f"workspace column {column!r} must be real numeric data")
    market_columns = [column for column in ("open", "high", "low", "close", "volume") if column in data]
    values = data[market_columns].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError("workspace OHLCV must be finite; missing-bar policy belongs in data.py")
    prices = [column for column in ("open", "high", "low", "close") if column in data]
    if (data[prices].to_numpy(dtype=float) <= 0).any() or ("volume" in data and (data["volume"] < 0).any()):
        raise ValueError("workspace prices must be positive and volume nonnegative")
    endpoints = data[[column for column in ("open", "close") if column in data]]
    if ((data["high"] < endpoints.max(axis=1)) | (data["low"] > endpoints.min(axis=1))).any():
        raise ValueError("workspace OHLC ordering must satisfy low <= open/close <= high")
    if np.isinf(data.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("workspace auxiliary values may be missing, but not infinite")


class WorkspaceData:
    """Load workspace preparation lazily and cache complete panels per run."""

    def __init__(self, workspace):
        self.workspace = workspace
        self._loader = None
        self._panels = {}

    def fetch(self, sources: list[str]) -> pd.DataFrame:
        if not sources or len(set(sources)) != len(sources):
            raise ValueError("request at least one distinct workspace source")
        key = tuple(sources)
        if key not in self._panels:
            if self._loader is None:
                module = load_workspace_module(self.workspace.dir, "data", required=True)
                factory = getattr(module, "create_loader", None)
                if not callable(factory):
                    raise ValueError("workspace data.py must define create_loader(*, start, asset, cache_dir)")
                self._loader = factory(start=self.workspace.start_date,
                                       asset=dict(self.workspace.asset),
                                       cache_dir=self.workspace.dir / "00_data")
                if not callable(self._loader):
                    raise ValueError("workspace create_loader must return a callable accepting sources")
            data = self._loader(list(sources))
            validate_market_data(data)
            self._panels[key] = data.copy(deep=True)
        return self._panels[key].copy(deep=True)
