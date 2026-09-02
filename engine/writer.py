"""
The pipeline's deliverable.

write_barrier_xlsx — per node, one sheet per horizon. Rows are barrier levels theta,
columns are the feature's condition bins, cells are P(price touches theta within h |
feature in bin) with the event count beside them.

Nothing else is written.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

_PCT_FMT = '0.0%'

_RED, _WHITE, _AMBER = 'C00000', 'FFFFFF', 'FFEB9C'
_FILL_ROW1 = PatternFill('solid', start_color='666666', end_color='666666')
_FILL_ROW2 = PatternFill('solid', start_color='B2B2B2', end_color='B2B2B2')
_FILL_ROW3 = PatternFill('solid', start_color='CCCCCC', end_color='CCCCCC')
_FILL_NBAND = PatternFill('solid', start_color='EFEFEF', end_color='EFEFEF')
_CENTER = Alignment(horizontal='center', vertical='center', wrap_text=True)

_HDR_ROWS = 3
_COL_ROW  = 5   # bin labels
_N_ROW    = 6   # observations per bin
_DATA_ROW = 7   # first theta row

_ABBREV = {'rsi', 'ma', 'atr', 'dxy', 'macd', 'bb', 'mvrv', 'wr', 'roc', 'vol'}


def _write_headers(ws, row1: str, row2: str, row3: str, merge_end: str) -> None:
    for r, text, font, fill in [
        (1, row1, Font(bold=True, size=14, color='FFFFFF'), _FILL_ROW1),
        (2, row2, Font(bold=True, size=12, color='000000'), _FILL_ROW2),
        (3, row3, Font(size=11, color='000000'), _FILL_ROW3),
    ]:
        ws.merge_cells(f'A{r}:{merge_end}{r}')
        c = ws[f'A{r}']
        c.value, c.font, c.fill, c.alignment = text, font, fill, _CENTER
        ws.row_dimensions[r].height = 20


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


# ── Deliverable A ─────────────────────────────────────────────────────────────

def write_barrier_xlsx(
    surfaces: dict,          # {horizon label: engine.barrier.touch_matrix result}
    labels:   list[str],     # bin labels, len == n_bins
    path:     Path,
    node_id:  str,
    feature:  str,
    params:   dict,
) -> None:
    """
    One sheet per horizon: will price reach theta, given the condition?

    Each bin gets a pair of columns — the conditional probability, and the event count
    behind it. Keeping the count in its own numeric cell rather than folding it into
    the probability string is what lets the sheet stay sortable and conditionally
    formatted; a cell reading 40% off three events and one reading 40% off ninety are
    the same number and very different evidence, and the pair makes that visible.

    Column B is the unconditional P(touch theta), so every conditional cell can be read
    against the base rate on the same row rather than in isolation.
    """
    wb = Workbook()
    wb.remove(wb.active)
    label  = feature_label(node_id)
    n_bins = len(labels)
    end    = get_column_letter(2 + 2 * n_bins)

    for hz, s in surfaces.items():
        ws = wb.create_sheet(hz.replace('+', ''))
        _write_headers(
            ws,
            f'{label} — P(price touches theta within {hz}) by condition',
            f'rows: barrier theta (theta<0 = the low reaches it, theta>0 = the high)   |   '
            f'columns: {feature} quantile bins   |   cells: probability, then event count',
            f'node {node_id}   params={params}   n={s["n_obs"]}   '
            f'intraday high/low; the window opens at t+1',
            merge_end=end,
        )

        ws.cell(row=_COL_ROW, column=1, value='theta').font = Font(bold=True)
        ws.cell(row=_COL_ROW, column=2, value='base').font = Font(bold=True)
        for b, lab in enumerate(labels):
            c = ws.cell(row=_COL_ROW, column=3 + 2 * b, value=lab)
            c.font, c.alignment = Font(bold=True), _CENTER
            ws.cell(row=_COL_ROW, column=4 + 2 * b, value='hits').font = Font(bold=True, size=9)

        ws.cell(row=_N_ROW, column=1, value='n =').font = Font(bold=True, italic=True)
        nb = ws.cell(row=_N_ROW, column=2, value=int(s['n_obs']))
        nb.font, nb.fill = Font(italic=True), _FILL_NBAND
        for b in range(n_bins):
            c = ws.cell(row=_N_ROW, column=3 + 2 * b, value=int(s['bin_n'][b]))
            c.font, c.fill = Font(italic=True), _FILL_NBAND
            ws.cell(row=_N_ROW, column=4 + 2 * b).fill = _FILL_NBAND

        thetas = s['thetas']
        for i, th in enumerate(thetas):
            r = _DATA_ROW + i
            tc = ws.cell(row=r, column=1, value=float(th))
            tc.number_format, tc.font = '+0%;-0%', Font(bold=True)
            bc = ws.cell(row=r, column=2, value=float(s['base'][i]))
            bc.number_format, bc.font = _PCT_FMT, Font(italic=True)
            for b in range(n_bins):
                p = s['prob'][i, b]
                pc = ws.cell(row=r, column=3 + 2 * b,
                             value=None if np.isnan(p) else float(p))
                pc.number_format = _PCT_FMT
                hc = ws.cell(row=r, column=4 + 2 * b, value=int(s['hits'][i, b]))
                hc.font = Font(size=9, color='808080')

        n_rows = len(thetas)
        # One absolute scale across every column, so columns are comparable to each
        # other and not merely internally ranked.
        for b in range(n_bins):
            col = get_column_letter(3 + 2 * b)
            ws.conditional_formatting.add(
                f'{col}{_DATA_ROW}:{col}{_DATA_ROW + n_rows - 1}',
                ColorScaleRule(start_type='num', start_value=0,    start_color=_WHITE,
                               mid_type='num',   mid_value=0.35,   mid_color=_AMBER,
                               end_type='num',   end_value=0.85,   end_color=_RED))
            ws.column_dimensions[col].width = 11
            ws.column_dimensions[get_column_letter(4 + 2 * b)].width = 7
        ws.column_dimensions['A'].width = 9
        ws.column_dimensions['B'].width = 9
        ws.freeze_panes = f'C{_DATA_ROW}'

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
