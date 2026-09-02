import numpy as np
from pathlib import Path

import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

_DEV_FMT  = '+0.0"pp";-0.0"pp";"0pp"'

# A deviation is bounded by the base rate it is measured against: at a 21.7% base
# rate the surface cannot go below -21.7pp, but has 78pp of headroom above. A fixed
# ±20pp scale (calibrated when every outcome was a ~50% coin flip) therefore saturates
# one side and wastes the other. The scale is derived from the base rate instead.
_SCALE_MIN  = 8.0    # floor, so a very rare event still gets a readable spread
_SCALE_FRAC = 0.75   # of the distance to the nearer bound

_RED   = 'C00000'
_WHITE = 'FFFFFF'
_GREEN = '00B050'

# Sheet names carry the event symbol so two barriers (or two thresholds) never
# collide in one workbook. Readers match on the comparison mark, not the full name.
_MARKS = {'pdf': '≈', 'above': '>', 'below': '<'}


def sheet_names(event: str = 'dd') -> dict[str, str]:
    """The three sheet names for an event symbol, keyed 'pdf' / 'above' / 'below'."""
    return {k: f'P({event} | X {m} x) - P({event})' for k, m in _MARKS.items()}


def find_sheet(wb, kind: str):
    """
    Locate a sheet by kind ('pdf' / 'above' / 'below') regardless of which outcome
    wrote it. Readers match on the comparison mark rather than the full name, so a
    workbook produced under any outcome stays readable.
    """
    mark = _MARKS[kind]
    for name in wb.sheetnames:
        if name.startswith('P(') and f' X {mark} x)' in name:
            return wb[name]
    raise KeyError(f"no '{kind}' sheet in workbook (sheets: {wb.sheetnames})")

# Excel row layout:
#   rows 1-3  : header block (title, formula, description)
#   row 4     : empty (spacer, written by to_excel startrow offset)
#   row 5     : pandas column headers
#   row 6+    : data
_HDR_ROWS  = 3
_START_ROW = _HDR_ROWS + 1  # to_excel startrow → pandas writes headers at row 5
_DATA_ROW  = _HDR_ROWS + 3  # first data row = row 6


_FILL_ROW1 = PatternFill(start_color='666666', end_color='666666', fill_type='solid')
_FILL_ROW2 = PatternFill(start_color='B2B2B2', end_color='B2B2B2', fill_type='solid')
_FILL_ROW3 = PatternFill(start_color='CCCCCC', end_color='CCCCCC', fill_type='solid')
_CENTER    = Alignment(horizontal='center', vertical='center', wrap_text=True)

_ABBREVS = {'rsi', 'ma', 'atr', 'dxy', 'macd', 'bb', 'mvrv', 'wr', 'roc', 'vol'}


def _feature_label(node_id: str) -> str:
    words = []
    for p in node_id.split('_'):
        if p.isdigit():
            if words:
                words[-1] = words[-1] + '-' + p
        elif p.lower() in _ABBREVS:
            words.append(p.upper())
        else:
            words.append(p.title())
    return ' '.join(words)


def _subtract_base(df: pd.DataFrame, base_rate: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in df.columns:
        if col == 'n':
            continue
        result[col] = df[col] - base_rate.loc[col, 'base_rate']
    return result


def _local_function(p_above: pd.DataFrame, base_rate: pd.DataFrame) -> pd.DataFrame:
    """
    Recover the local (PDF) win rate by finite-differencing the CDF-above.

    p_above rows are ordered by ascending threshold, so p_above.iloc[i] covers
    the larger population (feature > lower_threshold) and p_above.iloc[i+1] the
    smaller one (feature > higher_threshold).  Naming convention:
      n_hi / v_hi  → higher observation count (lower threshold)
      n_lo / v_lo  → lower observation count  (higher threshold)

    Wins in the slice between two adjacent thresholds:
        wins_slice = n_hi · P_hi − n_lo · P_lo

    where P is the raw CDF win rate (percent).  Dividing by n_slice gives the
    local win rate for observations whose feature value falls in that interval.

    Note: n_hi and n_lo are the longest-horizon counts (n_last from _row), used
    as a consistent proxy across all horizons.  This slightly understates n for
    shorter horizons but is conservative and uniformly applied.
    """
    horizon_cols = [c for c in p_above.columns if c != 'n']
    rows, idx = [], []

    for i in range(len(p_above) - 1):
        n_hi    = int(p_above.iloc[i]['n'])
        n_lo    = int(p_above.iloc[i + 1]['n'])
        n_slice = n_hi - n_lo
        if n_slice < 1:
            continue

        t_hi = float(p_above.index[i].split()[-1])
        t_lo = float(p_above.index[i + 1].split()[-1])
        mid  = (t_hi + t_lo) / 2

        row = {'n': n_slice}
        for col in horizon_cols:
            v_hi = p_above.iloc[i][col]
            v_lo = p_above.iloc[i + 1][col]
            if pd.notna(v_hi) and pd.notna(v_lo):
                wins_slice = n_hi * v_hi / 100.0 - n_lo * v_lo / 100.0
                row[col]   = wins_slice / n_slice * 100.0 - base_rate.loc[col, 'base_rate']
            else:
                row[col]   = np.nan

        rows.append(row)
        idx.append(f'X ≈ {mid:.4g}')

    result = pd.DataFrame(rows, index=pd.Index(idx, name='condition'))
    return result[['n'] + horizon_cols]


def _scale_limit(base_rate: pd.DataFrame) -> float:
    """
    Symmetric ±limit for the colour scale, derived from the base rate.

    A conditional rate lives in [0, 100], so a deviation from a base rate p0 is
    bounded by [-p0, 100-p0]. The scale is set to a fraction of the *nearer* bound
    so neither tail saturates immediately, with a floor for very rare events.
    """
    rates = base_rate['base_rate'].dropna()
    if rates.empty:
        return 20.0
    p0    = float(rates.mean())
    room  = min(p0, 100.0 - p0)
    return max(_SCALE_MIN, round(room * _SCALE_FRAC, 1))


def _color_scale(ws, n_rows: int, n_horizons: int, data_row: int,
                 limit: float = 20.0, sign: int = -1) -> None:
    """
    Colour a deviation surface so that red always means "worse".

    `sign` comes from the outcome registry: -1 when more of the event is bad news
    (a drawdown barrier being touched more often), +1 when it is good news (a runup).
    Without this the drawdown surfaces render backwards — elevated crash risk in
    green — because the original scale hard-coded the directional convention where
    a higher probability was always favourable.
    """
    rng = f'{get_column_letter(3)}{data_row}:{get_column_letter(2 + n_horizons)}{data_row - 1 + n_rows}'
    lo_color, hi_color = (_GREEN, _RED) if sign < 0 else (_RED, _GREEN)
    ws.conditional_formatting.add(rng, ColorScaleRule(
        start_type='num', start_value=-limit, start_color=lo_color,
        mid_type='num',   mid_value=0,        mid_color=_WHITE,
        end_type='num',   end_value=+limit,   end_color=hi_color,
    ))


def _format_cells(ws, n_rows: int, n_horizons: int, data_row: int) -> None:
    for col in range(3, 3 + n_horizons):
        for row in range(data_row, data_row + n_rows):
            ws.cell(row=row, column=col).number_format = _DEV_FMT


def _write_headers(ws, row1: str, row2: str, row3: str, merge_end: str = 'P') -> None:
    specs = [
        (1, row1, Font(bold=True,   size=14, color='FFFFFF'), _FILL_ROW1),
        (2, row2, Font(bold=True,   size=13, color='000000'), _FILL_ROW2),
        (3, row3, Font(           size=12, color='000000'), _FILL_ROW3),
    ]
    for r, text, font, fill in specs:
        ws.merge_cells(f'A{r}:{merge_end}{r}')
        cell           = ws[f'A{r}']
        cell.value     = text
        cell.font      = font
        cell.fill      = fill
        cell.alignment = _CENTER
        ws.row_dimensions[r].height = 22


def write_xlsx(
    p_below:   pd.DataFrame,
    p_above:   pd.DataFrame,
    base_rate: pd.DataFrame,
    path:      Path,
    node_id:   str,
    event:     str = 'dd10',
    title:     str = 'Drawdown Risk',
    expr:      str = 'min(price_h0+1..h) / price_h0 - 1 < -0.1',
    sign:      int = -1,
) -> None:
    label     = _feature_label(node_id)
    sheets    = sheet_names(event)
    SHEET_PDF, SHEET_ABOVE, SHEET_BELOW = sheets['pdf'], sheets['above'], sheets['below']
    dev_above = _subtract_base(p_above, base_rate)
    dev_below = _subtract_base(p_below, base_rate)
    fn_df     = _local_function(p_above, base_rate)
    limit     = _scale_limit(base_rate)

    n_cdf        = len(dev_above)
    n_pdf        = len(fn_df)
    horizon_cols = [c for c in dev_above.columns if c != 'n']
    n_horizons   = len(horizon_cols)
    merge_end    = get_column_letter(2 + n_horizons)  # condition + n + horizons

    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        fn_df.to_excel(    writer, sheet_name=SHEET_PDF,   startrow=_START_ROW)
        dev_above.to_excel(writer, sheet_name=SHEET_ABOVE, startrow=_START_ROW)
        dev_below.to_excel(writer, sheet_name=SHEET_BELOW, startrow=_START_ROW)

        wb = writer.book

        _write_headers(wb[SHEET_PDF],
            f'Conditional {title} on {label} — Probability Density Function (PDF)',
            f'P({expr} | {label} ≈ x) - P({expr})',
            f'Measures P({expr}) at horizon h, given that {label} is approximately at value x, minus the unconditional base rate.',
            merge_end=merge_end,
        )
        _format_cells(wb[SHEET_PDF], n_pdf, n_horizons, data_row=_DATA_ROW)
        _color_scale( wb[SHEET_PDF], n_pdf, n_horizons, data_row=_DATA_ROW, limit=limit, sign=sign)

        _write_headers(wb[SHEET_ABOVE],
            f'Conditional {title} on {label} — Cumulative Distribution Function (CDF)',
            f'P({expr} | {label} > x) - P({expr})',
            f'Measures P({expr}) at horizon h, given that {label} is above threshold x, minus the unconditional base rate.',
            merge_end=merge_end,
        )
        _format_cells(wb[SHEET_ABOVE], n_cdf, n_horizons, data_row=_DATA_ROW)
        _color_scale( wb[SHEET_ABOVE], n_cdf, n_horizons, data_row=_DATA_ROW, limit=limit, sign=sign)

        _write_headers(wb[SHEET_BELOW],
            f'Conditional {title} on {label} — Cumulative Distribution Function (CDF)',
            f'P({expr} | {label} < x) - P({expr})',
            f'Measures P({expr}) at horizon h, given that {label} is below threshold x, minus the unconditional base rate.',
            merge_end=merge_end,
        )
        _format_cells(wb[SHEET_BELOW], n_cdf, n_horizons, data_row=_DATA_ROW)
        _color_scale( wb[SHEET_BELOW], n_cdf, n_horizons, data_row=_DATA_ROW, limit=limit, sign=sign)

    print(f'Saved -> {path}')
