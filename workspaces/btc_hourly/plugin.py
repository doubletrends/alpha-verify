"""BTC hourly test workspace — raw OHLCV from the public GitHub dataset."""

from __future__ import annotations

import pandas as pd


# The repository stores this CSV in Git LFS. The usual raw.githubusercontent URL
# returns an LFS pointer, while this media URL dereferences the actual CSV bytes.
DATASET_URL = (
    "https://media.githubusercontent.com/media/mouadja02/"
    "bitcoin-technical-indicators-dataset/main/bitcoin-hourly-ohlcv.csv"
)
_COLUMNS = ["DATETIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME_BTC"]


def _hourly_ohlcv(start: str, asset: dict) -> pd.DataFrame:
    """Load the publisher's raw hourly bars, not its derived indicator columns."""
    frame = pd.read_csv(DATASET_URL, usecols=_COLUMNS)
    index = pd.to_datetime(frame.pop("DATETIME"), utc=True).dt.tz_localize(None)
    data = frame.rename(columns={"VOLUME_BTC": "VOLUME"}).rename(columns=str.lower)
    data = data.set_axis(index, axis="index")
    data.index.name = "Date"
    data = data.sort_index().loc[lambda value: value.index >= pd.Timestamp(start)]
    data = data[~data.index.duplicated(keep="last")]
    if data.empty:
        raise ValueError(f"BTC hourly dataset has no bars on or after {start}")
    return data


def register(sources, features) -> None:
    """Replace Yahoo's limited intraday source for this workspace only."""
    sources.register("ohlcv", _hourly_ohlcv)
