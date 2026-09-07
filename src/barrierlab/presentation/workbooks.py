"""
The pipeline's deliverable.

write_barrier_xlsx renders a cube as one tab per condition bin. Each tab is that bin's
whole (Δ x horizon) face, so the workbook holds every value the cube holds -- it is
a faithful view of the measurement, not a summary of it.

Stage 1 workbooks show raw conditional probabilities. Stage 2 shift workbooks show the
same full grid after subtracting the unconditional baseline.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from barrierlab.domain.barrier import MIN_BIN_N

_PCT_FMT = '0.0%'
_PP_FMT = '+0.0;-0.0;0.0'
_SHIFT_PCT_FMT = '+0.0%;-0.0%;0.0%'
_PVAL_FMT = '0.0000'

_WHITE, _AMBER, _RED = 'FFFFFF', 'FFD166', 'C00000'
_BLUE = '2A78D6'
_GREEN = '00875A'
_SHIFT_SCALE_LIMIT = 0.30
_FILL_ROW1 = PatternFill('solid', start_color='666666', end_color='666666')
_FILL_ROW2 = PatternFill('solid', start_color='B2B2B2', end_color='B2B2B2')
_FILL_NBAND = PatternFill('solid', start_color='EFEFEF', end_color='EFEFEF')
_FILL_NA = PatternFill('solid', start_color='F7F7F7', end_color='F7F7F7')
_FILL_MID = PatternFill('solid', start_color='DDDDDD', end_color='DDDDDD')
_CENTER = Alignment(horizontal='center', vertical='center', wrap_text=True)

_COL_ROW  = 4   # horizon labels
_N_ROW    = 3   # observations behind this bin at each horizon
_DATA_ROW = 5   # first Δ row
_VALID_COL_ROW = 3

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


def _sheet_names(labels: list[str]) -> list[str]:
    """
    Tab names are the conditions themselves -- 'x < 0.12', '0.12 < x < 0.34' -- so the
    tab strip reads as the ladder of bins with nothing to decode.

    Excel caps a sheet name at 31 chars and forbids : \\ / ? * [ ]. Truncation could in
    principle collide two long labels, so a numeric suffix is appended only when that
    actually happens rather than pre-emptively numbering every tab.
    """
    out: list[str] = []
    for label in labels:
        name = re.sub(r'[:\\/?*\[\]]', '-', label)[:31]
        if name in out:
            base = name[:28]
            k = 2
            while f'{base}~{k}' in out:
                k += 1
            name = f'{base}~{k}'
        out.append(name)
    return out


def _write_headers(ws, row1: str, row2: str, row3: str | None, merge_end: str) -> None:
    rows = [
        (1, row1, Font(bold=True, size=14, color='FFFFFF'), _FILL_ROW1),
        (2, row2, Font(bold=True, size=11, color='000000'), _FILL_ROW2),
    ]
    for r, text, font, fill in rows:
        ws.merge_cells(f'A{r}:{merge_end}{r}')
        c = ws[f'A{r}']
        c.value, c.font, c.fill, c.alignment = text, font, fill, _CENTER
        ws.row_dimensions[r].height = 18


def _condition_title(node_id: str, feature: str, labels: list[str], edges: np.ndarray,
                     bin_index: int, unconditional: bool) -> str:
    display_feature = feature_label(node_id)
    if unconditional:
        condition = 'all'
    elif len(edges) == 0:
        condition = labels[bin_index]
    elif bin_index == 0:
        condition = f'{display_feature} < {float(edges[0]):.3f}'
    elif bin_index == len(edges):
        condition = f'{float(edges[-1]):.3f} < {display_feature}'
    else:
        condition = f'{float(edges[bin_index - 1]):.3f} < {display_feature} < {float(edges[bin_index]):.3f}'
    return f'Condition —— {condition}'


def write_barrier_xlsx(
    cube:    dict,
    path:    Path,
    node_id: str,
    feature: str,
    params:  dict,
    unit:    str = 'd',
) -> None:
    """
    One tab per condition bin; each tab is that bin's full Δ x horizon face.

    Rows run from the highest barrier at the top to the lowest at the bottom, the way a
    price ladder reads: up the sheet is up in price. Δ = 0 sits in the middle and is
    shaded, marking the boundary between two different questions -- above it a cell asks
    whether the *high* reached that level, below it whether the *low* did.

    The colour scale is a fixed 0..100% on every tab, so flipping between them shows the
    probability band moving with the condition instead of each tab being rescaled to
    look alike.

    The n band under the header is the observation count behind that bin at each
    horizon; it falls as t grows, because the last t bars have no realized forward
    window.
    """
    prob     = cube['prob']
    Δs   = cube['Δs']
    horizons = cube['horizons']
    bin_n    = cube['bin_n']
    labels   = cube['meta']['bin_labels']
    edges    = cube['edges']
    n_th, n_bins, n_t = prob.shape

    order = np.argsort(Δs)[::-1]        # highest barrier on the top row
    end   = get_column_letter(1 + n_t)

    wb = Workbook()
    wb.remove(wb.active)

    unconditional = n_bins == 1 and labels[0] == 'all'
    names = _sheet_names(list(labels))

    for b in range(n_bins):
        ws = wb.create_sheet(names[b])
        title = _condition_title(node_id, feature, labels, edges, b, unconditional)
        _write_headers(
            ws,
            title,
            'P (Price touches Δ within t | Condition)',
            None,
            merge_end=end,
        )

        c = ws.cell(row=_COL_ROW, column=1, value='Δ')
        c.font, c.alignment = Font(bold=True), _CENTER
        for j, t in enumerate(horizons):
            c = ws.cell(row=_COL_ROW, column=2 + j, value=f'+{int(t)}{unit}')
            c.font, c.alignment = Font(bold=True), _CENTER

        c = ws.cell(row=_N_ROW, column=1, value='n =')
        c.font, c.fill = Font(bold=True, italic=True, size=9), _FILL_NBAND
        for j in range(n_t):
            c = ws.cell(row=_N_ROW, column=2 + j, value=int(bin_n[b, j]))
            c.font, c.fill, c.alignment = Font(italic=True, size=9), _FILL_NBAND, _CENTER

        for r_off, i in enumerate(order):
            r = _DATA_ROW + r_off
            th = float(Δs[i])
            is_zero = abs(th) < 1e-12
            tc = ws.cell(row=r, column=1, value=th)
            tc.number_format = '+0%;-0%;0%'
            tc.font, tc.alignment = Font(bold=True), _CENTER
            if is_zero:
                tc.fill = _FILL_MID
            for j in range(n_t):
                v = prob[i, b, j]
                cell = ws.cell(row=r, column=2 + j,
                               value=None if not np.isfinite(v) else round(float(v), 4))
                cell.number_format = _PCT_FMT
                if not np.isfinite(v):
                    cell.fill = _FILL_NA

        ws.conditional_formatting.add(
            f'B{_DATA_ROW}:{end}{_DATA_ROW + n_th - 1}',
            ColorScaleRule(start_type='num', start_value=0,   start_color=_WHITE,
                           mid_type='num',   mid_value=0.5,   mid_color=_AMBER,
                           end_type='num',   end_value=1.0,   end_color=_RED))

        ws.column_dimensions['A'].width = 8
        for j in range(n_t):
            ws.column_dimensions[get_column_letter(2 + j)].width = 6.5
        ws.freeze_panes = f'B{_DATA_ROW}'

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def write_shift_xlsx(
    cube:    dict,
    path:    Path,
    node_id: str,
    feature: str,
    params:  dict,
    unit:    str = 'd',
) -> None:
    """
    One tab per condition bin; each tab is the full baseline-subtracted surface.

    Values are percentage-point deviations from the baseline node. The color scale is
    centered at zero: blue means the barrier is touched less often than unconditional,
    red means more often.
    """
    values   = cube['shift']
    Δs   = cube['Δs']
    horizons = cube['horizons']
    bin_n    = cube['bin_n']
    labels   = cube['meta']['bin_labels']
    edges    = cube['edges']
    n_th, n_bins, n_t = values.shape

    order = np.argsort(Δs)[::-1]
    end = get_column_letter(1 + n_t)
    lim = _SHIFT_SCALE_LIMIT

    wb = Workbook()
    wb.remove(wb.active)

    unconditional = n_bins == 1 and labels[0] == 'all'
    names = _sheet_names(list(labels))

    for b in range(n_bins):
        ws = wb.create_sheet(names[b])
        title = _condition_title(node_id, feature, labels, edges, b, unconditional)
        _write_headers(
            ws,
            title,
            'P (Price touches Δ within t | Condition) - P (Price touches Δ within t)',
            None,
            merge_end=end,
        )

        c = ws.cell(row=_COL_ROW, column=1, value='Δ')
        c.font, c.alignment = Font(bold=True), _CENTER
        for j, t in enumerate(horizons):
            c = ws.cell(row=_COL_ROW, column=2 + j, value=f'+{int(t)}{unit}')
            c.font, c.alignment = Font(bold=True), _CENTER

        c = ws.cell(row=_N_ROW, column=1, value='n =')
        c.font, c.fill = Font(bold=True, italic=True, size=9), _FILL_NBAND
        for j in range(n_t):
            c = ws.cell(row=_N_ROW, column=2 + j, value=int(bin_n[b, j]))
            c.font, c.fill, c.alignment = Font(italic=True, size=9), _FILL_NBAND, _CENTER

        for r_off, i in enumerate(order):
            r = _DATA_ROW + r_off
            th = float(Δs[i])
            is_zero = abs(th) < 1e-12
            tc = ws.cell(row=r, column=1, value=th)
            tc.number_format = '+0%;-0%;0%'
            tc.font, tc.alignment = Font(bold=True), _CENTER
            if is_zero:
                tc.fill = _FILL_MID
            for j in range(n_t):
                v = values[i, b, j]
                cell = ws.cell(row=r, column=2 + j,
                               value=None if not np.isfinite(v) else round(float(v) / 100.0, 4))
                cell.number_format = _SHIFT_PCT_FMT
                if not np.isfinite(v):
                    cell.fill = _FILL_NA

        rng = f'B{_DATA_ROW}:{end}{_DATA_ROW + n_th - 1}'
        ws.conditional_formatting.add(
            rng,
            ColorScaleRule(start_type='num', start_value=-lim, start_color=_BLUE,
                           mid_type='num', mid_value=0, mid_color=_WHITE,
                           end_type='num', end_value=lim, end_color=_RED))

        ws.column_dimensions['A'].width = 8
        for j in range(n_t):
            ws.column_dimensions[get_column_letter(2 + j)].width = 8.5
        ws.freeze_panes = f'B{_DATA_ROW}'

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def _safe_sheet_name(name: str, used: set[str]) -> str:
    name = re.sub(r'[:\\/?*\[\]]', '-', name)[:31]
    if name not in used:
        used.add(name)
        return name
    base = name[:28]
    k = 2
    while f'{base}~{k}' in used:
        k += 1
    name = f'{base}~{k}'
    used.add(name)
    return name


def _write_validation_matrix(
    wb: Workbook,
    used: set[str],
    sheet_name: str,
    values: np.ndarray,
    Δs: np.ndarray,
    horizons: np.ndarray,
    title: str,
    subtitle: str,
    row3: str,
    number_format: str,
    unit: str,
    pvalue_scale: bool,
) -> None:
    n_th, n_t = values.shape
    order = np.argsort(Δs)[::-1]
    end = get_column_letter(1 + n_t)
    ws = wb.create_sheet(_safe_sheet_name(sheet_name, used))
    _write_headers(ws, title, subtitle, row3, merge_end=end)

    c = ws.cell(row=_VALID_COL_ROW, column=1, value='Δ')
    c.font, c.alignment = Font(bold=True), _CENTER
    for j, t in enumerate(horizons):
        c = ws.cell(row=_VALID_COL_ROW, column=2 + j, value=f'+{int(t)}{unit}')
        c.font, c.alignment = Font(bold=True), _CENTER

    for r_off, i in enumerate(order):
        r = _DATA_ROW + r_off
        th = float(Δs[i])
        tc = ws.cell(row=r, column=1, value=th)
        tc.number_format = '+0%;-0%;0%'
        tc.font, tc.alignment = Font(bold=True), _CENTER
        if abs(th) < 1e-12:
            tc.fill = _FILL_MID
        for j in range(n_t):
            v = values[i, j]
            cell = ws.cell(row=r, column=2 + j,
                           value=None if not np.isfinite(v) else float(v))
            cell.number_format = number_format
            if not np.isfinite(v):
                cell.fill = _FILL_NA

    rng = f'B{_DATA_ROW}:{end}{_DATA_ROW + n_th - 1}'
    if pvalue_scale:
        ws.conditional_formatting.add(
            rng,
            ColorScaleRule(start_type='num', start_value=0, start_color=_GREEN,
                           mid_type='num', mid_value=0.05, mid_color=_AMBER,
                           end_type='num', end_value=1.0, end_color=_RED))
    else:
        ws.conditional_formatting.add(
            rng,
            ColorScaleRule(start_type='num', start_value=-20, start_color=_GREEN,
                           mid_type='num', mid_value=0, mid_color=_WHITE,
                           end_type='num', end_value=20, end_color=_RED))

    ws.column_dimensions['A'].width = 8
    for j in range(n_t):
        ws.column_dimensions[get_column_letter(2 + j)].width = 8.5
    ws.freeze_panes = f'B{_DATA_ROW}'


def write_validation_xlsx(
    result:  dict,
    path:    Path,
    node_id: str,
    feature: str,
    params:  dict,
    unit:    str = 'd',
) -> None:
    """
    Human-readable validation artifact.

    The first sheet is the horizon-level node search statistic. The
    remaining sheets render the per-cell signed deviation and pointwise p-value for
    each condition bin. Pointwise p-values are for reading the surface; discoveries are
    assigned by validation across the selected node/horizon sweep.
    """
    Δs = result['Δs']
    horizons = result['horizons']
    labels = result['meta']['bin_labels']
    n_bins = result['cell_real'].shape[1]

    wb = Workbook()
    ws = wb.active
    ws.title = 'summary'
    end = 'F'
    _write_headers(
        ws,
        f'{feature_label(node_id)} — validation summary',
        'Peak null test by horizon; this is the statistic validation corrects',
        f'node {node_id}   params={params}   p floor = 1/(usable shifts + 1)',
        merge_end=end,
    )
    headers = ['horizon', 'peak dev pp', 'null p95 pp', 'peak p', 'usable shifts', 'p floor']
    for col, header in enumerate(headers, 1):
        c = ws.cell(row=_VALID_COL_ROW, column=col, value=header)
        c.font, c.alignment = Font(bold=True), _CENTER
    for r_off, t in enumerate(horizons):
        r = _VALID_COL_ROW + 1 + r_off
        n = int(result['n_shifts'][r_off])
        floor = None if n <= 0 else 1.0 / (1.0 + n)
        row = [
            f'+{int(t)}{unit}',
            result['peak_real'][r_off],
            result['peak_p95'][r_off],
            result['peak_p'][r_off],
            n,
            floor,
        ]
        for col, v in enumerate(row, 1):
            c = ws.cell(row=r, column=col,
                        value=None if isinstance(v, float) and not np.isfinite(v) else v)
            if col in (2, 3):
                c.number_format = _PP_FMT
            if col in (4, 6):
                c.number_format = _PVAL_FMT
            c.alignment = _CENTER
    for col, width in enumerate([10, 13, 13, 10, 14, 10], 1):
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.freeze_panes = f'A{_VALID_COL_ROW + 1}'

    used = {'summary'}
    names = _sheet_names(list(labels))
    for b in range(n_bins):
        label = labels[b]
        suffix = names[b]
        common = (f'node {node_id}   bin {b + 1} of {n_bins}: {label}   '
                  'rows: barrier Δ   columns: horizon')
        _write_validation_matrix(
            wb, used, f'dev {suffix}',
            result['cell_real'][:, b, :], Δs, horizons,
            f'{feature_label(node_id)} — signed validation deviation',
            'Deviation from this node bin sample rate, percentage points',
            common,
            _PP_FMT,
            unit,
            pvalue_scale=False,
        )
        _write_validation_matrix(
            wb, used, f'p {suffix}',
            result['cell_p'][:, b, :], Δs, horizons,
            f'{feature_label(node_id)} — pointwise validation p-values',
            'Per-cell null p-values; useful for reading, not for discovery claims',
            common,
            _PVAL_FMT,
            unit,
            pvalue_scale=True,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
