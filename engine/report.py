"""Stage 7: render report figures from artifacts on disk."""

from __future__ import annotations

from engine.report_diagnostics import fig_null, fig_null_gap_ranking
from engine.report_single_node import fig_band, fig_shift_all
from universe import all_nodes, load_universe


_CAPTIONS = {
    'A_band.png': 'A. The measured forward envelope starts at the last close.',
    'B_null.png': 'B. The exact circular-shift null threshold and its resolution floor.',
    'D_null_gap_ranking.png': 'D. The strongest discoveries sit far beyond their own '
                               'null thresholds.',
}


def _caption(path) -> str:
    if path.name.startswith('C_shift_'):
        node = path.stem.removeprefix('C_shift_')
        return f'C. {node}: strongest cleared conditional deviation from the baseline.'
    return _CAPTIONS.get(path.name, path.stem)


def build(ws) -> None:
    """Render every figure this workspace has the artifacts for."""
    out = ws.result_dir
    universe = load_universe(ws.universe_path)
    cleared = ws.read_json(ws.cleared_path)

    if not cleared:
        print('No 05_gate.json - run --gate first.')
        return

    print(f"\n=== 7. Report [{ws.dir.name}] ===")
    print(f"  rendering from artifacts on disk; nothing here re-measures\n")

    jobs = [
        ('A band', 'A_band.png', lambda: fig_band(ws, universe, cleared, out)),
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

    index = [f'# {ws.dir.name}', '',
             f"`{ws.asset['ticker']}` · {ws.asset.get('interval', '1d')} · "
             f"from {ws.start_date} · {len(all_nodes(universe))} nodes", '']
    for p in written:
        index += [f'### {_caption(p)}', '',
                  f'![{p.stem}]({p.name})', '']
    (out / 'README.md').write_text('\n'.join(index), encoding='utf-8')

    print(f"\n  wrote {len(written)} figures to workspaces/{ws.dir.name}/result/")
