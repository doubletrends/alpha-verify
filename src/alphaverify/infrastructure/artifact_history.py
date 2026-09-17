"""Market and feature histories restored from stage artifacts, and their identity."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from alphaverify.infrastructure.market_data import validate_market_data


_MARKET_COLUMNS = ("open", "high", "low", "close", "volume")
_REQUIRED_HISTORY_COLUMNS = ("index", "high", "low", "close")


def market_data_from_artifact(artifact: dict) -> pd.DataFrame:
    """Restore the ordered OHLCV history retained in a pipeline artifact."""
    missing = [key for key in _REQUIRED_HISTORY_COLUMNS if key not in artifact]
    if missing:
        raise ValueError(f"artifact lacks ordered history ({', '.join(missing)})")
    index = pd.to_datetime(artifact["index"])
    data = pd.DataFrame(
        {key: artifact[key].astype(float) for key in _MARKET_COLUMNS if key in artifact},
        index=index,
    )
    data.index.name = "Date"
    # Legacy artifacts may omit open/volume, but no stored row is silently dropped.
    validate_market_data(data, require_full_ohlcv=False)
    return data


def feature_from_artifact(artifact: dict, index: pd.Index) -> pd.Series:
    """Restore a feature series and align it to reconstructed market history."""
    if "feature_values" not in artifact or "index" not in artifact:
        raise ValueError("artifact lacks ordered feature values")
    values = pd.Series(
        artifact["feature_values"].astype(float),
        index=pd.to_datetime(artifact["index"]),
        name="feature",
    )
    return values.reindex(index)


def market_history_key(data: pd.DataFrame) -> str:
    """Content identity for ordered market data shared by multiple nodes."""
    digest = hashlib.sha256()
    digest.update(str(tuple(data.columns)).encode())
    digest.update(data.index.asi8.tobytes())
    digest.update(np.ascontiguousarray(data.to_numpy(dtype=float)).tobytes())
    return digest.hexdigest()
