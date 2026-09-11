"""BTC hourly raw-bar preparation, with UTC timestamps and no resampling."""

import numpy as np
import pandas as pd

from .._shared.snapshots import snapshot

# The media endpoint dereferences the publisher's Git LFS object.
DATASET_URL = (
    "https://media.githubusercontent.com/media/mouadja02/"
    "bitcoin-technical-indicators-dataset/main/bitcoin-hourly-ohlcv.csv"
)
COLUMNS = ["DATETIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME_BTC"]
CLEANING_VERSION = "btc-hourly-v1"


def create_loader(*, start, asset, cache_dir):
    if (asset.get("provider") != "mouadja02/bitcoin-technical-indicators-dataset"
            or asset.get("interval") != "1h" or asset.get("ticker") != "BTC-USD"):
        raise ValueError("BTC hourly data.py requires the declared hourly BTC dataset")
    prepared = None

    def load(sources):
        nonlocal prepared
        if sources != ["ohlcv"]:
            raise ValueError("BTC hourly declares only the ohlcv source")
        if prepared is None:
            raw = pd.read_csv(DATASET_URL, usecols=COLUMNS)
            info = snapshot(raw, cache_dir, "ohlcv")
            frame = raw.copy()
            index = pd.to_datetime(frame.pop("DATETIME"), utc=True, errors="raise").dt.tz_localize(None)
            frame = frame.rename(columns={"VOLUME_BTC": "VOLUME"}).rename(columns=str.lower)
            frame.index = pd.DatetimeIndex(index, name="Date")
            frame = frame.sort_index(kind="stable")
            frame = frame[~frame.index.duplicated(keep="last")]
            frame = frame.loc[frame.index >= pd.Timestamp(start)]
            frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            frame = frame.dropna()  # Missing bars are removed, never filled or resampled.
            frame.attrs["provenance"] = {
                "requested_start": start, "asset": dict(asset),
                "cleaning_version": CLEANING_VERSION, "time_basis": "UTC, timezone-naive",
                "alignment": "raw timestamps; no resampling or filling; keep last duplicate",
                "sources": {"ohlcv": {**info, "url": DATASET_URL,
                                       "raw_rows": len(raw), "prepared_rows": len(frame)}},
            }
            prepared = frame
        return prepared.copy()

    return load
