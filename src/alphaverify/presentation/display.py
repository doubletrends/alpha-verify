"""Display conventions shared by workbook and figure views."""

from __future__ import annotations

import numpy as np

# Shift color scales saturate at this probability difference (±30 percentage points).
SHIFT_DISPLAY_LIMIT = 0.30

_ABBREV = {'rsi', 'ma', 'atr', 'dxy', 'macd', 'bb', 'mvrv', 'wr', 'roc', 'vol'}


def barrier_decimals(barriers) -> int:
    """Fewest percent decimals that show every barrier of a grid exactly, at most four."""
    percents = np.abs(np.asarray(barriers, dtype=float)) * 100
    for decimals in range(4):
        if np.allclose(np.round(percents, decimals), percents, rtol=0, atol=1e-9):
            return decimals
    return 4


def format_barrier(value: float, barriers) -> str:
    """A signed barrier as a percentage, precise enough for its grid."""
    return f"{value:+.{barrier_decimals(barriers)}%}"


def barrier_number_format(barriers) -> str:
    """Excel number format matching ``format_barrier`` for a grid."""
    decimals = barrier_decimals(barriers)
    fraction = "." + "0" * decimals if decimals else ""
    return f"+0{fraction}%;-0{fraction}%;0%"


def feature_label(node_id: str) -> str:
    words = []
    for p in node_id.split('_'):
        if p.isdigit():
            if words:
                words[-1] += '-' + p
        elif p.lower() in _ABBREV:
            words.append(p.upper())
        else:
            words.append(p.title())
    return ' '.join(words)
