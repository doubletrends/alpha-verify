"""Optional Yahoo transport shared by workspaces; no cleaning or alignment policy."""

import json
import urllib.parse
import urllib.request

import pandas as pd
import yfinance as yf


def download(ticker, interval, start, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir / "yfinance"))
    frame = yf.download(ticker, start=start, interval=interval,
                        auto_adjust=False, progress=False)
    if not frame.empty:
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.attrs["transport"] = "yfinance"
        return frame
    query = urllib.parse.urlencode({
        "period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
        "period2": 2147483647, "interval": interval,
    })
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + urllib.parse.quote(
        ticker, safe=""
    ) + "?" + query
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    results = payload.get("chart", {}).get("result") or []
    if not results:
        raise ValueError(f"Yahoo returned no history for {ticker}")
    result = results[0]
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    # Decode the provider's timestamps; retain timezone information for the caller.
    zone = result.get("meta", {}).get("exchangeTimezoneName", "UTC")
    index = pd.to_datetime(result.get("timestamp", []), unit="s", utc=True).tz_convert(zone)
    frame = pd.DataFrame({key.title(): quote.get(key, [])
                          for key in ("open", "high", "low", "close", "volume")}, index=index)
    frame.attrs["transport"] = "yahoo-chart"
    return frame
