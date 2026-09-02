"""
Outcome registry — the barrier-touch event whose conditional probability the matrix measures.

The engine measures  P(outcome | feature condition) − P(outcome).  The question it is
built around is *first passage*: will price reach a level L at any point before t+h?
That is what a stop, a liquidation, an option and a margin call all actually care
about, and it is a different question from "where does price close at t+h" — a spike
that round-trips touches the barrier but ends flat.

Two barriers, one primitive:

    drawdown   price touches −θ  (relative to close_t) at some point in (t, t+h]
    runup      price touches +θ  at some point in (t, t+h]

They are mirror images, so both are `_touch` with a sign. Measuring both on the same
bars is what makes the *skew* readout possible — whether a condition raises downside
touch risk specifically, or simply raises both barriers (a volatility proxy).

An outcome fn has signature  (data: pd.DataFrame, h: int, params: dict) -> pd.Series
of 1.0 / 0.0 / NaN aligned to data.index.  NaN marks bars whose outcome is not yet
realized (the trailing h bars); those rows are dropped from every count.

Registration mirrors data/features.py, so a workspace plugin can add its own outcome
at import time with no root edits.
"""

from typing import Callable

import pandas as pd

# Below this the event is too rare for conditional counting to say anything: the
# threshold slices carry a handful of events each and every deviation is noise.
# θ has to be scaled to the asset and horizon (10% over 14 BTC days is a 21.7%
# event; over 24 NASDAQ hours it is a 0.4% one), so this guards against a θ that
# was sensible for one workspace being carried into another.
MIN_BASE_RATE = 0.05

_registry: dict[str, dict] = {}


def register(
    name:  str,
    fn:    Callable,
    event: str,
    title: str,
    expr:  str,
    sign:  int = -1,
) -> None:
    """
    Register an outcome.

      name  — key used in universe.json / --outcome
      fn    — (data, h, params) -> Series of 1.0/0.0/NaN
      event — very short symbol for sheet names, e.g. 'dd', 'run'  (Excel caps
              sheet names at 31 chars, so keep this under ~6 characters)
      title — human title for the xlsx header block, e.g. 'Drawdown Risk'
      expr  — formula-style description of the event for the header block
      sign  — how to colour a positive deviation: -1 when more of this event is
              bad news (drawdown), +1 when it is good news (runup). The writer's
              colour scale reads this, so "worse" is red for every outcome.
    """
    _registry[name] = {'fn': fn, 'event': event, 'title': title, 'expr': expr, 'sign': sign}


def get(name: str) -> dict:
    if name not in _registry:
        raise KeyError(f"unknown outcome '{name}' (registered: {', '.join(sorted(_registry))})")
    return _registry[name]


def available() -> list[str]:
    return sorted(_registry)


def compute(data: pd.DataFrame, name: str, h: int, params: dict | None = None) -> pd.Series:
    """Evaluate a registered outcome at horizon h. Returns 1.0/0.0/NaN on data.index."""
    return get(name)['fn'](data, h, params or {})


def mirror(name: str) -> str | None:
    """
    The opposite barrier of a touch outcome, or None if it has no mirror.

    Used by the skew readout, which measures the same condition against both
    barriers on identical bars.
    """
    return {'drawdown': 'runup', 'runup': 'drawdown'}.get(name)


def event_label(name: str, params: dict | None = None) -> str:
    """
    Short symbol for the event, used in xlsx sheet names, output paths and filenames.

    The threshold is folded into the label so that a 10% barrier and a 20% barrier
    never produce identically-named sheets or overwrite each other's directories.
    """
    spec  = get(name)
    ev    = spec['event']
    thr   = (params or {}).get('threshold')
    if thr is not None:
        ev = f'{ev}{abs(float(thr)) * 100:g}'
    return ev


def describe(name: str, params: dict | None = None) -> str:
    """One-line description of the event with its parameters substituted in."""
    spec = get(name)
    expr = spec['expr']
    for k, v in (params or {}).items():
        expr = expr.replace(k, f'{v:g}' if isinstance(v, (int, float)) else str(v))
    return expr


# ── built-in outcomes ─────────────────────────────────────────────────────────

def _forward_extreme(close: pd.Series, h: int, direction: int) -> pd.Series:
    """
    Best (direction=+1) or worst (direction=-1) close over (t, t+h], as a return
    relative to close_t. NaN where the full window is not realized.

    Note the window starts at t+1: the barrier can only be touched *after* the bar
    the condition is read on, so today's own close never counts as a touch.
    """
    fwd = pd.concat([close.shift(-k) for k in range(1, h + 1)], axis=1)
    ext = fwd.max(axis=1) if direction > 0 else fwd.min(axis=1)
    return (ext / close - 1.0).where(close.shift(-h).notna())


def _touch(data: pd.DataFrame, h: int, params: dict, direction: int) -> pd.Series:
    """
    Did price touch the barrier at ±θ within h bars?

    direction = -1 → a peak-to-trough loss exceeding θ  (drawdown)
    direction = +1 → a gain exceeding θ at any point    (runup)
    """
    thr  = abs(float(params.get('threshold', 0.10)))
    move = _forward_extreme(data['close'], h, direction)
    hit  = move < -thr if direction < 0 else move > thr
    return hit.where(move.notna()).astype(float)


def _drawdown(data: pd.DataFrame, h: int, params: dict) -> pd.Series:
    """Price fell more than `threshold` below close_t at some point within h bars."""
    return _touch(data, h, params, direction=-1)


def _runup(data: pd.DataFrame, h: int, params: dict) -> pd.Series:
    """Price rose more than `threshold` above close_t at some point within h bars."""
    return _touch(data, h, params, direction=+1)


register('drawdown', _drawdown,
         event='dd',  title='Drawdown Risk',
         expr='min(price_h0+1..h) / price_h0 - 1 < -threshold',
         sign=-1)

register('runup',    _runup,
         event='run', title='Runup Chance',
         expr='max(price_h0+1..h) / price_h0 - 1 > +threshold',
         sign=+1)
