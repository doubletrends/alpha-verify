"""Stage seven: render report figures from artifacts on disk."""

from __future__ import annotations

from engine.report_composition import fig_calibration, fig_composition_grid
from engine.report_diagnostics import (
    fig_families,
    fig_funnel,
    fig_horizons,
    fig_landscape,
    fig_null,
    fig_null_gap_ranking,
)
from engine.report_single_node import fig_atr_ladder, fig_band, fig_baseline, fig_shift
from tree.tree import all_nodes, load_tree


_CAPTIONS = {
    '01_band.png': 'The measured forward envelope starts at the last close.',
    '02_landscape.png': 'Magnitude clears the null; direction mostly does not.',
    '03_where_the_edge_is.png': 'Where the strongest cleared effects sit by horizon.',
    '04_families.png': 'Which feature families carry discoveries.',
    '05_null.png': 'The exact circular-shift null as a distribution, and its resolution '
                   'floor.',
    '06_funnel.png': 'The economic and statistical filters both matter.',
    '07_baseline_surface.png': 'The unconditional surface is the reference every '
                               'condition must beat.',
    '08_conditional_shift.png': 'One condition\'s strongest actionable deviation from '
                                'that baseline.',
    '09_composition.png': 'Composing conditions out of sample exposes overconfidence.',
    '10_composition_grid.png': 'Where the composed ranking survives out of sample.',
    '11_atr_regime_ladder.png': 'ATR turns one barrier into a visible risk ladder.',
    '12_null_gap_ranking.png': 'The strongest discoveries sit far beyond their own '
                               'null thresholds.',
}


def build(ws) -> None:
    """Render every figure this workspace has the artifacts for."""
    out = ws.result_dir
    tree = load_tree(ws.tree_path)
    evaluation = ws.read_json(ws.eval_path)
    cleared = ws.read_json(ws.cleared_path)

    if not cleared:
        print('No 05_gate.json - run --gate first.')
        return

    print(f"\n=== 7. Report [{ws.dir.name}] ===")
    print(f"  rendering from artifacts on disk; nothing here re-measures\n")

    # stale numbering from an earlier layout would otherwise linger beside the new files
    for stale in out.glob('*.png'):
        stale.unlink()

    jobs = [
        ('band',             lambda: fig_band(ws, tree, cleared, out)),
        ('landscape',        lambda: fig_landscape(ws, tree, evaluation, cleared, out)),
        ('where the edge is', lambda: fig_horizons(ws, cleared, out)),
        ('families',         lambda: fig_families(ws, tree, cleared, out)),
        ('the null',         lambda: fig_null(ws, tree, cleared, out)),
        ('funnel',           lambda: fig_funnel(ws, tree, evaluation, cleared, out)),
        ('baseline surface', lambda: fig_baseline(ws, out)),
        ('conditional shift', lambda: fig_shift(ws, tree, cleared, out)),
        ('composition',      lambda: fig_calibration(ws, out)),
        ('composition grid', lambda: fig_composition_grid(ws, out)),
        ('ATR ladder',       lambda: fig_atr_ladder(ws, cleared, out)),
        ('null gap ranking', lambda: fig_null_gap_ranking(ws, cleared, out)),
    ]

    written = []
    for name, fn in jobs:
        try:
            path = fn()
        except Exception as e:
            print(f"  {name:<20} [skip] {type(e).__name__}: {e}")
            continue
        if path is None:
            print(f"  {name:<20} [skip] artifact missing")
            continue
        written.append(path)
        print(f"  {name:<20} {path.name}")

    index = [f'# {ws.dir.name}', '',
             f"`{ws.asset['ticker']}` · {ws.asset.get('interval', '1d')} · "
             f"from {ws.start_date} · {len(all_nodes(tree))} nodes", '']
    for p in written:
        index += [f'### {_CAPTIONS.get(p.name, p.stem)}', '',
                  f'![{p.stem}]({p.name})', '']
    (out / 'README.md').write_text('\n'.join(index), encoding='utf-8')

    print(f"\n  wrote {len(written)} figures to workspaces/{ws.dir.name}/result/")
