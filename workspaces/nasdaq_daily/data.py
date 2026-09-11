"""NASDAQ daily inputs: unadjusted Yahoo bars and same-date auxiliary closes.

Dates are exchange session labels, timezone-naive. Auxiliary values use the
same labelled day's close, then forward-fill on primary sessions, never backfill.
This retains the experiment's end-of-day convention; it is not a statement
that all feeds were published simultaneously or are point-in-time vintages.
"""

import numpy as np
import pandas as pd

from .._shared.yahoo import download
from .._shared.snapshots import snapshot

SOURCES = {"vix": ("^VIX", "vix"), "treasury": ("^TNX", "tnx"),
           "dxy": ("DX-Y.NYB", "dxy")}
CLEANING_VERSION = "nasdaq-daily-v1"


def create_loader(*, start, asset, cache_dir):
    if asset.get("provider") != "yfinance" or asset.get("interval") != "1d":
        raise ValueError("NASDAQ data.py requires the declared yfinance daily source")
    feeds, provenance = {}, {}

    def feed(name):
        if name not in feeds:
            ticker = asset["ticker"] if name == "ohlcv" else SOURCES[name][0]
            raw = download(ticker, "1d", start, cache_dir)
            provenance[name] = {**snapshot(raw, cache_dir, name), "ticker": ticker,
                                "transport": raw.attrs.get("transport", "yfinance")}
            frame = raw.rename(columns=str.lower).copy()
            frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
            frame.index.name = "Date"
            frame = frame.sort_index(kind="stable")
            frame = frame[~frame.index.duplicated(keep="last")]
            frame = frame.loc[frame.index >= pd.Timestamp(start)]
            columns = ["open", "high", "low", "close", "volume"] if name == "ohlcv" else ["close"]
            frame = frame[columns].apply(pd.to_numeric, errors="coerce")
            frame = frame.replace([np.inf, -np.inf], np.nan)
            if name == "ohlcv":
                # Drop incomplete bars; never invent prices or repair invalid ordering.
                frame = frame.dropna(subset=columns)
            else:
                frame = frame.rename(columns={"close": SOURCES[name][1]}).ffill()
            provenance[name]["raw_rows"] = len(raw)
            provenance[name]["prepared_rows"] = len(frame)
            feeds[name] = frame
        return feeds[name]

    def load(sources):
        if not sources or sources[0] != "ohlcv" or set(sources) - {"ohlcv", *SOURCES}:
            raise ValueError("NASDAQ sources must start with ohlcv and use declared source names")
        panel = feed("ohlcv").copy()
        for name in sources[1:]:
            panel = panel.join(feed(name).reindex(panel.index, method="ffill"))
        panel.attrs["provenance"] = {
            "requested_start": start, "asset": dict(asset),
            "cleaning_version": CLEANING_VERSION, "time_basis": "exchange session date",
            "alignment": "same-date auxiliary close; forward fill; no backfill",
            "sources": {name: provenance[name] for name in sources},
        }
        return panel

    return load
