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


def feature_bins_from_artifact(artifact: dict) -> np.ndarray:
    """Each stored bar's condition bin under the artifact's stored edges, or -1 when missing.

    Uses the Stage 1 left-edge convention: a value equal to an edge enters the lower bin.
    """
    values = np.asarray(artifact["feature_values"], dtype=float)
    bins = np.searchsorted(np.asarray(artifact["bin_edges"], dtype=float), values, side="left")
    return np.where(np.isfinite(values), bins, -1)


def market_history_key(data: pd.DataFrame) -> str:
    """Content identity for ordered market data shared by multiple nodes."""
    digest = hashlib.sha256()
    digest.update(str(tuple(data.columns)).encode())
    digest.update(data.index.asi8.tobytes())
    digest.update(np.ascontiguousarray(data.to_numpy(dtype=float)).tobytes())
    return digest.hexdigest()
