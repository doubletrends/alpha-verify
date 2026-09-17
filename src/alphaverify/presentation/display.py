"""Display conventions shared by workbook and figure views."""

from __future__ import annotations

# Shift color scales saturate at this probability difference (±30 percentage points).
SHIFT_DISPLAY_LIMIT = 0.30

_ABBREV = {'rsi', 'ma', 'atr', 'dxy', 'macd', 'bb', 'mvrv', 'wr', 'roc', 'vol'}


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
