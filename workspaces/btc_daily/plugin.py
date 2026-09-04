"""BTC daily workspace plugin — registers coinmetrics source and BTC-specific features."""

import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from data import features, fetcher

# ── coinmetrics source ────────────────────────────────────────────────────────

def _coinmetrics(start: str, asset: dict) -> pd.DataFrame:
    params = {
        'assets':    'btc',
        'metrics':   'CapMVRVCur,HashRate,AdrActCnt,TxCnt',
        'frequency': '1d',
        'format':    'json',
        'page_size': '10000',
    }
    url = (
        'https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?'
        + urllib.parse.urlencode(params)
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.load(r)
    df = pd.DataFrame(payload['data'])
    df['Date'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
    df = df.set_index('Date').sort_index()
    rename = {
        'CapMVRVCur': 'mvrv',
        'HashRate':   'hash_rate',
        'AdrActCnt':  'adr_act_cnt',
        'TxCnt':      'tx_cnt',
    }
    df = df[list(rename)].rename(columns=rename).apply(pd.to_numeric, errors='coerce')
    return df[df.index >= pd.Timestamp(start)]

fetcher.register_source('coinmetrics', _coinmetrics)


# ── on-chain feature helpers ──────────────────────────────────────────────────

def _hash_rate_ma_ratio(data: pd.DataFrame, period: int) -> pd.Series:
    hr = np.log(data['hash_rate'].replace(0, np.nan))
    return hr / hr.rolling(period).mean() - 1.0


def _adr_act_ma_ratio(data: pd.DataFrame, period: int) -> pd.Series:
    aa = np.log(data['adr_act_cnt'].replace(0, np.nan))
    return aa / aa.rolling(period).mean() - 1.0


# ── halving cycle helpers ─────────────────────────────────────────────────────

_HALVING_DATES = pd.to_datetime([
    '2012-11-28',
    '2016-07-09',
    '2020-05-11',
    '2024-04-20',
])
_NEXT_HALVING_EST = pd.Timestamp('2028-04-20')
_ALL_HALVINGS = list(_HALVING_DATES) + [_NEXT_HALVING_EST]


def _cycle_phase(data: pd.DataFrame) -> pd.Series:
    """Fractional position in the current halving cycle: 0 = just after halving, 1 = just before next."""
    dates = pd.to_datetime(data.index)
    phases = []
    for dt in dates:
        past   = [h for h in _ALL_HALVINGS if h <= dt]
        future = [h for h in _ALL_HALVINGS if h > dt]
        last   = max(past) if past else _ALL_HALVINGS[0]
        nxt    = min(future) if future else _NEXT_HALVING_EST
        phases.append((dt - last).days / max((nxt - last).days, 1))
    return pd.Series(phases, index=data.index, dtype=float)


def _days_to_halving(data: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(data.index)
    vals = []
    for dt in dates:
        future = [h for h in _ALL_HALVINGS if h > dt]
        nxt    = min(future) if future else _NEXT_HALVING_EST
        vals.append(max(0, (nxt - dt).days))
    return pd.Series(vals, index=data.index, dtype=float)


def _days_since_halving(data: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(data.index)
    days  = []
    for dt in dates:
        eligible = _HALVING_DATES[_HALVING_DATES <= dt]
        last     = eligible.max() if len(eligible) else _HALVING_DATES.min()
        days.append(max(0, (dt - last).days))
    return pd.Series(days, index=data.index, dtype=float)


# ── register features ─────────────────────────────────────────────────────────

features.register('mvrv',               lambda d, p: d['mvrv'])
features.register('hash_rate_ma_ratio', lambda d, p: _hash_rate_ma_ratio(d, p['period']))
features.register('adr_act_ma_ratio',   lambda d, p: _adr_act_ma_ratio(d, p['period']))
features.register('cycle_phase',        lambda d, p: _cycle_phase(d))
features.register('days_to_halving',    lambda d, p: _days_to_halving(d))
features.register('days_since_halving', lambda d, p: _days_since_halving(d))
