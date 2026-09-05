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


def write_bayes_xlsx(
    path: Path,
    probability: np.ndarray,
    deltas: np.ndarray,
    horizons: np.ndarray,
    n_observations: np.ndarray,
    unit: str = "d",
) -> None:
    """Render one weighted-Bayes probability face with Stage 1's exact formatting."""
    surface = np.asarray(probability, dtype=float)
    if surface.shape != (len(deltas), len(horizons)):
        raise ValueError("Bayes probability surface must be shaped delta x horizon")
    counts = np.asarray(n_observations, dtype=int)
    if counts.shape != (len(horizons),):
        raise ValueError("Bayes observation counts must have one value per horizon")
    write_barrier_xlsx(
        {
            "prob": surface[:, None, :],
            "Δs": np.asarray(deltas, dtype=float),
            "horizons": np.asarray(horizons, dtype=int),
            "bin_n": counts[None, :],
            "meta": {"bin_labels": ["Weighted Bayes"]},
            "edges": np.array([], dtype=float),
        },
        path,
        node_id="weighted_bayes",
        feature="weighted_bayes",
        params={},
        unit=unit,
    )


def write_bayes_shift_xlsx(
    path: Path,
    shift_pp: np.ndarray,
    probability: np.ndarray,
    baseline: np.ndarray,
    deltas: np.ndarray,
    horizons: np.ndarray,
    n_observations: np.ndarray,
    unit: str = "d",
) -> None:
    """Render one weighted-Bayes shift face with Stage 2's exact formatting."""
    shifts = np.asarray(shift_pp, dtype=float)
    probabilities = np.asarray(probability, dtype=float)
    baseline_surface = np.asarray(baseline, dtype=float)
    expected_shape = (len(deltas), len(horizons))
    if shifts.shape != expected_shape:
        raise ValueError("Bayes shift surface must be shaped delta x horizon")
    if probabilities.shape != expected_shape or baseline_surface.shape != expected_shape:
        raise ValueError("Bayes probability and baseline surfaces must be shaped delta x horizon")
    counts = np.asarray(n_observations, dtype=int)
    if counts.shape != (len(horizons),):
        raise ValueError("Bayes observation counts must have one value per horizon")
    write_shift_xlsx(
        {
            "shift": shifts[:, None, :],
            "prob": probabilities[:, None, :],
            "base": baseline_surface,
            "Δs": np.asarray(deltas, dtype=float),
            "horizons": np.asarray(horizons, dtype=int),
            "bin_n": counts[None, :],
            "meta": {"bin_labels": ["Weighted Bayes"]},
            "edges": np.array([], dtype=float),
        },
        path,
        node_id="weighted_bayes",
        feature="weighted_bayes",
        params={},
        unit=unit,
    )


def write_redundancy_xlsx(
    path: Path,
    names: list[str],
    families: list[str],
    information: np.ndarray,
    similarity: np.ndarray,
    cluster_ids: np.ndarray,
    representatives: np.ndarray,
    clusters: list[dict],
    pairs: list[dict],
    validation_by_node: dict[str, dict],
    meta: dict,
) -> None:
    """Render the Stage 5 numerical artifact as one research-facing workbook."""
    wb = Workbook()
    overview = wb.active
    overview.title = 'Overview'

    overview.merge_cells('A1:H1')
    title = overview['A1']
    title.value = 'Stage 5 — conditional redundancy map'
    title.font = Font(size=16, bold=True, color='FFFFFF')
    title.fill = _FILL_ROW1
    title.alignment = Alignment(horizontal='left', vertical='center')
    overview.row_dimensions[1].height = 25
    overview_rows = [
        ('workspace', meta['workspace']),
        ('generated UTC', meta['generated']),
        ('target', f"P(touch {meta['target']['Δ']:+.0%} within {meta['target']['horizon']}{meta['target']['unit']})"),
        ('cleared nodes', len(names)),
        ('clusters', len(clusters)),
        ('conditional-NMI threshold', meta['method']['threshold']),
        ('complete observations', meta['n_observations']),
        ('outcome rate', meta['outcome_rate']),
        ('validation fingerprint', meta['source_validation_fingerprint']),
        ('interpretation', 'Research diagnostic only; Stage 6 recomputes selection and weights inside each training fold.'),
    ]
    for row_index, (label, value) in enumerate(overview_rows, 3):
        overview.cell(row=row_index, column=1, value=label).font = Font(bold=True)
        cell = overview.cell(row=row_index, column=2, value=value)
        cell.alignment = Alignment(wrap_text=True, vertical='top')
        if label == 'outcome rate':
            cell.number_format = _PCT_FMT
        if label == 'conditional-NMI threshold':
            cell.number_format = '0.00'
    overview.column_dimensions['A'].width = 30
    overview.column_dimensions['B'].width = 92

    order = sorted(
        range(len(names)),
        key=lambda i: (int(cluster_ids[i]), not bool(representatives[i]), -float(information[i]), names[i]),
    )
    matrix = wb.create_sheet('Matrix')
    matrix.cell(row=1, column=1, value='node / conditional NMI').font = Font(bold=True)
    for position, index in enumerate(order, 2):
        header = matrix.cell(row=1, column=position, value=names[index])
        header.font = Font(bold=True, color='FFFFFF' if representatives[index] else '000000')
        header.fill = PatternFill(
            'solid',
            start_color=_GREEN if representatives[index] else 'D9EAF7',
            end_color=_GREEN if representatives[index] else 'D9EAF7',
        )
        header.alignment = Alignment(text_rotation=90, horizontal='center', vertical='bottom')
        row_header = matrix.cell(row=position, column=1, value=names[index])
        row_header.font = Font(bold=bool(representatives[index]))
        if representatives[index]:
            row_header.fill = PatternFill('solid', start_color='D9EAD3', end_color='D9EAD3')
        for column_position, column_index in enumerate(order, 2):
            cell = matrix.cell(
                row=position,
                column=column_position,
                value=float(similarity[index, column_index]),
            )
            cell.number_format = '0.000'
            cell.alignment = _CENTER
    matrix_range = f"B2:{get_column_letter(len(names) + 1)}{len(names) + 1}"
    matrix.conditional_formatting.add(
        matrix_range,
        ColorScaleRule(
            start_type='num', start_value=0, start_color='FFFFFF',
            mid_type='num', mid_value=float(meta['method']['threshold']), mid_color='FFD966',
            end_type='num', end_value=1, end_color='C00000',
        ),
    )
    matrix.freeze_panes = 'B2'
    matrix.column_dimensions['A'].width = 28
    matrix.row_dimensions[1].height = 110
    for column in range(2, len(names) + 2):
        matrix.column_dimensions[get_column_letter(column)].width = 5

    cluster_sheet = wb.create_sheet('Clusters')
    cluster_headers = [
        'cluster', 'representative', 'members', 'families', 'size',
        'representative information', 'mean internal NMI', 'maximum internal NMI',
    ]
    cluster_sheet.append(cluster_headers)
    for row in clusters:
        cluster_sheet.append([
            row['cluster'], row['representative'], ', '.join(row['members']),
            ', '.join(row['families']), row['size'], row['representative_information'],
            row['mean_internal_nmi'], row['max_internal_nmi'],
        ])

    node_sheet = wb.create_sheet('Nodes')
    node_headers = [
        'node', 'family', 'cluster', 'representative', 'information',
        'validation rank', 'best horizon', 'raw p', 'BH q',
        'economic deviation pp', 'economic barrier',
    ]
    node_sheet.append(node_headers)
    for index in order:
        validation = validation_by_node[names[index]]
        best = validation.get('economic', {}).get('best') or validation.get('best_cell') or {}
        node_sheet.append([
            names[index], families[index], int(cluster_ids[index]), bool(representatives[index]),
            float(information[index]), validation.get('selection_rank'), validation.get('horizon'),
            validation.get('peak_p'), validation.get('q_value'), best.get('dev'), best.get('Δ'),
        ])

    pair_sheet = wb.create_sheet('Pairs')
    pair_headers = ['left', 'right', 'conditional NMI', 'above threshold', 'same cluster']
    pair_sheet.append(pair_headers)
    for row in pairs:
        pair_sheet.append([
            row['left'], row['right'], row['conditional_nmi'], row['above_threshold'],
            row['same_cluster'],
        ])

    definitions = wb.create_sheet('Definitions')
    definition_rows = [
        ('conditional NMI', 'Normalized I(bin_i; bin_j | touch outcome); larger values indicate more duplicated conditional evidence.'),
        ('threshold', 'Pairs at or above this value are linked; connected components form clusters.'),
        ('representative', 'The cluster member with the greatest individual target information.'),
        ('cluster', 'An interpretive dependence group, not a hard feature-deletion instruction.'),
        ('Stage 6 boundary', 'Walk-forward Bayes does not use this full-history map; it learns fold-local selection and weights.'),
    ]
    definitions.append(['term', 'meaning'])
    for row in definition_rows:
        definitions.append(row)

    for sheet, widths in (
        (cluster_sheet, [10, 28, 78, 30, 8, 27, 20, 23]),
        (node_sheet, [28, 18, 10, 16, 15, 16, 14, 12, 12, 24, 18]),
        (pair_sheet, [28, 28, 18, 18, 15]),
        (definitions, [24, 100]),
    ):
        for cell in sheet[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = _FILL_ROW1
            cell.alignment = _CENTER
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
        for column, width in enumerate(widths, 1):
            sheet.column_dimensions[get_column_letter(column)].width = width
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical='top', wrap_text=True)

    for row in cluster_sheet.iter_rows(min_row=2, min_col=6, max_col=8):
        for cell in row:
            cell.number_format = '0.0000'
    for row in node_sheet.iter_rows(min_row=2, min_col=5, max_col=5):
        row[0].number_format = '0.0000'
    for row in node_sheet.iter_rows(min_row=2, min_col=8, max_col=9):
        for cell in row:
            cell.number_format = _PVAL_FMT
    for row in node_sheet.iter_rows(min_row=2, min_col=10, max_col=10):
        row[0].number_format = _PP_FMT
    for row in node_sheet.iter_rows(min_row=2, min_col=11, max_col=11):
        row[0].number_format = _PCT_FMT
    for row in pair_sheet.iter_rows(min_row=2, min_col=3, max_col=3):
        row[0].number_format = '0.0000'

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    wb.close()
