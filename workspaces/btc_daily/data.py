"""BTC daily source and cleaning policy.

Yahoo bars use daily session labels; CoinMetrics uses UTC daily labels.
Auxiliary values are aligned by labelled date, forward-filled, never backfilled.
This preserves the existing end-of-day research convention; downloaded values
are not point-in-time vintages and their labels do not establish release times.
"""

import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .._shared.yahoo import download
from .._shared.snapshots import snapshot

SOURCES = {"vix": ("^VIX", "vix"), "treasury": ("^TNX", "tnx"),
           "dxy": ("DX-Y.NYB", "dxy")}
METRICS = {"CapMVRVCur": "mvrv", "HashRate": "hash_rate",
           "AdrActCnt": "adr_act_cnt", "TxCnt": "tx_cnt"}
CLEANING_VERSION = "btc-daily-v1"


def _coinmetrics(start):
    query = urllib.parse.urlencode({
        "assets": "btc", "metrics": ",".join(METRICS), "frequency": "1d",
        "format": "json", "page_size": "10000", "start_time": start,
    })
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?" + query
    rows, seen = [], set()
    while url:
        if url in seen:
            raise ValueError("CoinMetrics repeated a pagination URL")
        seen.add(url)
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.load(response)
        rows.extend(payload["data"])
        url = payload.get("next_page_url")
    if not rows:
        raise ValueError("CoinMetrics returned no history")
    return pd.DataFrame(rows)


def create_loader(*, start, asset, cache_dir):
    if (asset.get("provider") != "yfinance" or asset.get("interval") != "1d"
            or asset.get("ticker") != "BTC-USD"):
        raise ValueError("BTC daily data.py requires the declared BTC-USD daily source")
    feeds, provenance = {}, {}

    def feed(name):
        if name not in feeds:
            if name == "coinmetrics":
                raw = _coinmetrics(start)
                info = {"provider": "coinmetrics", "asset": "btc", "metrics": list(METRICS)}
                frame = raw.set_index(pd.to_datetime(raw["time"], utc=True).dt.tz_localize(None))
                frame = frame[list(METRICS)].rename(columns=METRICS)
            else:
                ticker = asset["ticker"] if name == "ohlcv" else SOURCES[name][0]
                raw = download(ticker, "1d", start, cache_dir)
                info = {"ticker": ticker, "transport": raw.attrs.get("transport", "yfinance")}
                frame = raw.rename(columns=str.lower).copy()
                frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
                if name == "ohlcv":
                    frame = frame[["open", "high", "low", "close", "volume"]]
                else:
                    frame = frame[["close"]].rename(columns={"close": SOURCES[name][1]})
            provenance[name] = {**snapshot(raw, cache_dir, name), **info, "raw_rows": len(raw)}
            frame.index.name = "Date"
            frame = frame.sort_index(kind="stable")
            frame = frame[~frame.index.duplicated(keep="last")]
            frame = frame.loc[frame.index >= pd.Timestamp(start)]
            frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if name == "ohlcv":
                frame = frame.dropna()  # No invented prices or OHLC-order repairs.
            elif name != "coinmetrics":
                frame = frame.ffill()
            # CoinMetrics missing values remain missing on actual source dates.
            provenance[name]["prepared_rows"] = len(frame)
            feeds[name] = frame
        return feeds[name]

    def load(sources):
        if not sources or sources[0] != "ohlcv" or set(sources) - {"ohlcv", "coinmetrics", *SOURCES}:
            raise ValueError("BTC daily sources must start with ohlcv and use declared source names")
        panel = feed("ohlcv").copy()
        for name in sources[1:]:
            panel = panel.join(feed(name).reindex(panel.index, method="ffill"))
        panel.attrs["provenance"] = {
            "requested_start": start, "asset": dict(asset),
            "cleaning_version": CLEANING_VERSION, "time_basis": "daily source date",
            "alignment": "same-date auxiliary value; forward fill missing dates; no backfill",
            "sources": {name: provenance[name] for name in sources},
        }
        return panel

    return load
