"""Stage 7: render report figures from artifacts on disk."""

from __future__ import annotations

from pipeline.step_04_validation import validation_summary_is_current
from presentation.reports.diagnostics import fig_null, fig_null_gap_ranking
from presentation.reports.single_node import fig_band, fig_shift_all


def build(ws) -> None:
    """Render every figure this workspace has the artifacts for."""
    out = ws.result_dir
    universe = ws.catalog.raw
    cleared = ws.read_json(ws.validation_summary_path)

    if not validation_summary_is_current(ws, cleared):
        print('No current complete 04_validation/validation.json - run --validation first.')
        return

    print(f"\n=== 7. Report [{ws.dir.name}] ===")
    print(f"  rendering from artifacts on disk; nothing here re-measures\n")

    jobs = [
        ('A band', 'A_band.png', lambda: fig_band(ws, out)),
        ('B null', 'B_null.png', lambda: fig_null(ws, universe, cleared, out)),
        ('C shifts', None, lambda: fig_shift_all(ws, universe, cleared, out)),
        ('D null gap ranking', 'D_null_gap_ranking.png',
         lambda: fig_null_gap_ranking(ws, cleared, out)),
    ]

    expected = {filename for _, filename, _ in jobs if filename}
    # Remove report views that no longer exist in the product, while preserving retained
    # views if a live-data renderer cannot rebuild one of them on this run.
    for stale in out.glob('*.png'):
        if stale.name not in expected and not stale.name.startswith('C_shift_'):
            stale.unlink()

    written = []
    for name, _, fn in jobs:
        try:
            result = fn()
        except Exception as e:
            print(f"  {name:<20} [skip] {type(e).__name__}: {e}")
            continue
        if result is None or result == []:
            print(f"  {name:<20} [skip] artifact missing")
            continue
        paths = result if isinstance(result, list) else [result]
        written.extend(paths)
        if len(paths) == 1:
            print(f"  {name:<20} {paths[0].name}")
        else:
            print(f"  {name:<20} {len(paths)} files")

    print(f"\n  wrote {len(written)} figures to workspaces/{ws.dir.name}/result/")
