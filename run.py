"""
Barrier-touch pipeline.

    --cubes             1. measure     cubes_full/       P(touch θ in h | bin)
    --surfaces          2. render      surfaces_full/    the same values, readable
    --cubes-summary     3. reduce      cubes_summary/    the coarse grid, + economic filter
    --surfaces-summary  4. render      surfaces_summary/
    --validate          5. falsify     validations/      exact shuffled null
                                    validations_xlsx/ readable validation sheets
    --gate              6. correct     cleared.json      Benjamini-Hochberg + economic

The full cube is the faithful record and is never judged. The summary is a strict
subset of it -- index selection, no interpolation -- chosen so adjacent cells are
genuinely different measurements. Everything that judges runs there: a peak over
1,190 cells has a far lower noise ceiling than one over 12,300, and counting those
cells as tests is an honest measure of how wide the search actually was.

Every node passes through every stage, the baseline included. Shuffling a constant
cannot change anything, so its validation is degenerate by construction -- one
meaningless artifact is cheaper than a branch in the pipeline.

Usage:
    python run.py --workspace btc_daily_14days --cubes
    python run.py --workspace btc_daily_14days --surfaces
    python run.py --workspace btc_daily_14days --cubes-summary
    python run.py --workspace btc_daily_14days --surfaces-summary
    python run.py --workspace btc_daily_14days --validate
    python run.py --workspace btc_daily_14days --gate
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data import features, fetcher
from engine import barrier, validate as val, writer
from tree.tree import load_tree, all_nodes, find_node, all_in_family
from workspace import Workspace, BASELINE_NODE


def _loader(ws: Workspace):
    cache: dict[tuple, pd.DataFrame] = {}

    def load(sources: list) -> pd.DataFrame:
        key = tuple(sorted(sources))
        if key not in cache:
            cache[key] = fetcher.fetch(list(sources), start=ws.start_date,
                                       asset=ws.asset).dropna(subset=['close'])
        return cache[key]
    return load


def _node_feature(ws: Workspace, node: dict, get_data):
    """(data, feature series) for a node; raises if it cannot be computed."""
    data = get_data(node['data'])
    if data.empty:
        raise ValueError('empty data')
    feat = features.compute(data, node['feature'], node['params']).reindex(data.index)
    n_valid = int(feat.notna().sum())
    if n_valid < ws.min_obs:
        raise ValueError(f'only {n_valid} valid observations')
    return data, feat


def _h_num(label: str) -> int:
    return int(''.join(c for c in label if c.isdigit()))


# ── Deliverable A ─────────────────────────────────────────────────────────────

def _baseline_surface(ws: Workspace) -> "np.ndarray | None":
    """
    The baseline node's probability surface, shaped (n_theta, n_horizons).

    This is the unconditional P(touch θ in h). It is measured as an ordinary node,
    so it arrives here the same way every condition does -- there is no separate base
    array anywhere in the pipeline.
    """
    path = ws.baseline_cube
    if not path.exists():
        return None
    return barrier.load_cube(path)['prob'][:, 0, :]


def build_cube(ws: Workspace, node: dict, get_data, quiet: bool = False) -> None:
    """Measure one node across every horizon and barrier level; write the full cube."""
    data, feat = _node_feature(ws, node, get_data)
    edges = barrier.bin_edges(feat, ws.n_bins)
    cube  = barrier.touch_tensor(data, feat, ws.horizons, ws.thetas, edges)

    barrier.save_cube(cube, ws.cube_path(node['family'], node['id']), {
        'node': node['id'], 'family': node['family'],
        'feature': node['feature'], 'params': node['params'],
        'workspace': ws.dir.name,
        'bin_labels': barrier.bin_labels(edges, feat),
        'grid': 'full',
        'generated': datetime.now(timezone.utc).isoformat(),
    })
    if not quiet:
        print(f"  {node['id']:<26} {cube['prob'].shape}  n={int(cube['n_obs'][0])}")


def cmd_cubes(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """Stage 1. The faithful record: every barrier level, every horizon."""
    tree  = load_tree(ws.tree_path)
    nodes = all_in_family(tree, family) if family else all_nodes(tree)
    if not rerun:
        nodes = [n for n in nodes if not ws.has_cube(n['family'], n['id'])]
    if not nodes:
        print('Nothing to do (use --rerun to rebuild existing cubes).')
        return

    get_data = _loader(ws)
    hz = ws.horizons
    print(f"\n=== 1. Full cubes [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(f"  {len(ws.thetas)} θ x {ws.n_bins} bins x {len(hz)} horizons "
          f"(+{hz[0]}{ws.horizon_unit}..+{hz[-1]}{ws.horizon_unit})   "
          f"θ {ws.thetas[0]:+.0%}..{ws.thetas[-1]:+.0%} step {ws.theta_step:.0%}")
    print(f"  value = P(touch θ in h | bin); intraday high/low\n")

    skipped = {}
    for node in nodes:
        try:
            build_cube(ws, node, get_data)
        except Exception as e:
            skipped[node['id']] = str(e)
            print(f"  {node['id']:<26} [skip] {e}")
    print(f"\n  wrote to {ws.dir.relative_to(ROOT)}/cubes_full/"
          + (f"   ({len(skipped)} skipped)" if skipped else ""))


def cmd_surfaces(ws: Workspace, family: str | None = None) -> None:
    """Stage 2. The full cube, made readable. Renders only; never re-measures."""
    _render(ws, family, summary=False)


def cmd_cubes_summary(ws: Workspace, family: str | None = None) -> None:
    """
    Stage 3. The coarse grid everything is judged on, selected from the full cube.

    Pure index selection -- no interpolation, no second measurement -- so a summary cell
    is bit-identical to the full cell it came from. Adjacent cells here are genuinely
    different measurements, which is what makes a peak over them meaningful and a count
    of them an honest test count.

    The economic filter runs here rather than on the full cube, so that what is judged
    and what is validated sit on the same grid.
    """
    tree  = load_tree(ws.tree_path)
    nodes = [n for n in (all_in_family(tree, family) if family else all_nodes(tree))
             if ws.has_cube(n['family'], n['id'])]
    if not nodes:
        print('No full cubes to summarize - run --cubes first.')
        return

    # the baseline is the reference the filter reads, so its summary must exist first
    nodes = ([n for n in nodes if n['id'] == BASELINE_NODE]
             + [n for n in nodes if n['id'] != BASELINE_NODE])

    th, hz = ws.summary_thetas, ws.summary_horizons
    print(f"\n=== 3. Summary cubes [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(f"  {len(th)} θ x {ws.n_bins} bins x {len(hz)} horizons = "
          f"{len(th) * ws.n_bins * len(hz)} cells  "
          f"(full: {len(ws.thetas) * ws.n_bins * len(ws.horizons)})")
    print(f"  θ {', '.join(f'{v:+.0%}' for v in th[len(th)//2:])}  mirrored")
    print(f"  h {', '.join(f'+{h}{ws.horizon_unit}' for h in hz)}\n")

    evals, skipped = {}, {}
    for node in nodes:
        try:
            full = barrier.load_cube(ws.cube_path(node['family'], node['id']))
            sub  = barrier.summarize(full, th, hz)
            barrier.save_cube(sub, ws.summary_cube_path(node['family'], node['id']),
                              {**full['meta'], 'grid': 'summary'})
        except Exception as e:
            skipped[node['id']] = str(e)
            print(f"  {node['id']:<26} [skip] {e}")
            continue

        baseline = _baseline_surface(ws)
        if baseline is None:
            print('  no baseline summary yet - cannot evaluate')
            return
        ev = barrier.evaluate(sub, baseline, ws.min_dev, ws.min_bin_n, ws.min_run)
        evals[node['id']] = ev
        b = ev['best']
        note = (f"best {b['dev']:+5.1f}pp  θ={b['theta']:+.0%}  "
                f"+{b['horizon']}{ws.horizon_unit:<3} bin {b['bin']}  n={b['bin_n']}  "
                f"run={b['run']}") if b else 'no qualifying cell'
        print(f"  {node['id']:<26} {note}")

    ws.write_json(ws.eval_path, {
        'workspace': ws.dir.name,
        'generated': datetime.now(timezone.utc).isoformat(),
        'grid': {'thetas': th.tolist(), 'horizons': hz.tolist()},
        'criteria': {'min_dev': ws.min_dev, 'min_bin_n': ws.min_bin_n,
                     'min_run': ws.min_run},
        'passed':  sorted(n for n, e in evals.items() if e['passed']),
        'nodes':   {n: {'passed': e['passed'], 'best': e['best']}
                    for n, e in evals.items()},
        'skipped': skipped,
    })
    n_pass = sum(1 for e in evals.values() if e['passed'])
    print(f"\n  economic filter: {n_pass} / {len(evals)} pass "
          f"(|dev| >= {ws.min_dev}pp across >= {ws.min_run} adjacent θ rows, "
          f"bin n >= {ws.min_bin_n})")
    print(f"  wrote {ws.eval_path.relative_to(ROOT)}")


def cmd_surfaces_summary(ws: Workspace, family: str | None = None) -> None:
    """Stage 4. The summary cube, made readable. Same renderer, smaller grid."""
    _render(ws, family, summary=True)


def _render(ws: Workspace, family: str | None, summary: bool) -> None:
    tree = load_tree(ws.tree_path)
    have = ws.has_summary_cube if summary else ws.has_cube
    src  = ws.summary_cube_path if summary else ws.cube_path
    dst  = ws.summary_surface_path if summary else ws.surface_path
    label = 'summary' if summary else 'full'
    stage = '4. Summary surfaces' if summary else '2. Full surfaces'

    nodes = [n for n in (all_in_family(tree, family) if family else all_nodes(tree))
             if have(n['family'], n['id'])]
    if not nodes:
        print(f'No {label} cubes to render - run --cubes'
              f'{"-summary" if summary else ""} first.')
        return

    print(f"\n=== {stage} [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(f"  {ws.n_bins} tabs per node, one per condition bin; "
          f"each tab is that bin's full θ x horizon face\n")
    n_ok = 0
    for node in nodes:
        cube = barrier.load_cube(src(node['family'], node['id']))
        try:
            writer.write_barrier_xlsx(cube, dst(node['family'], node['id']),
                                      node['id'], node['feature'], node['params'],
                                      ws.horizon_unit)
        except PermissionError:
            print(f"  {node['id']:<26} [locked] close it in Excel and re-run")
            continue
        n_ok += 1
    print(f"  wrote {n_ok} workbooks under {ws.dir.relative_to(ROOT)}/"
          f"surfaces_{label}/")


def cmd_validate(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """
    Stage 5, per node: null-test the *summary* cube and write an artifact beside it.

    Every node gets one, the baseline included. Shuffling a constant feature cannot
    change anything, so its artifact comes out degenerate -- peak deviation 0, p = 1 --
    and that is deliberate: a uniform pipeline with one meaningless file beats a
    pipeline with a special case in it.

    Per-node artifacts also mean two runs never write the same path, which is the
    failure that made the previous shared validation.json untrustworthy.
    """
    tree  = load_tree(ws.tree_path)
    nodes = [n for n in (all_in_family(tree, family) if family else all_nodes(tree))
             if ws.has_summary_cube(n['family'], n['id'])]
    if not rerun:
        nodes = [n for n in nodes if (
            not ws.has_validation(n['family'], n['id'])
            or not ws.has_validation_sheet(n['family'], n['id'])
        )]
    if not nodes:
        print('Nothing to validate or render (use --rerun to redo, or --cubes-summary first).')
        return

    get_data = _loader(ws)
    th, hz = ws.summary_thetas, ws.summary_horizons
    print(f"\n=== 5. Validation [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(f"  on the summary grid: {len(th)} θ x {ws.n_bins} bins x {len(hz)} horizons "
          f"= {len(th) * ws.n_bins * len(hz)} cells per node")
    print(f"  a peak over that has a far lower noise ceiling than one over the full "
          f"{len(ws.thetas) * ws.n_bins * len(ws.horizons)}")
    print(f"  guard {val.EDGE_GUARD} bars either side, so the p-value floor is 1/(usable+1)")
    print(f"  writes .npz machine artifacts and .xlsx readable validation sheets\n")

    done_npz = 0
    done_xlsx = 0
    for node in nodes:
        try:
            cube = barrier.load_cube(ws.summary_cube_path(node['family'], node['id']))
            if ws.has_validation(node['family'], node['id']) and not rerun:
                r = val.load(ws.validation_path(node['family'], node['id']))
            else:
                data, feat = _node_feature(ws, node, get_data)
                # the summary cube's own coordinates and edges, so the test and the
                # thing it judges are the same grid down to the last cell
                r = val.validate_node(data, feat, cube['horizons'], cube['thetas'], cube['edges'])
                val.save(r, ws.validation_path(node['family'], node['id']), {
                    'node': node['id'], 'family': node['family'],
                    'feature': node['feature'], 'params': node['params'],
                    'workspace': ws.dir.name,
                    'bin_labels': cube['meta']['bin_labels'],
                    'generated': datetime.now(timezone.utc).isoformat(),
                })
                done_npz += 1
            writer.write_validation_xlsx(
                r, ws.validation_sheet_path(node['family'], node['id']),
                node['id'], node['feature'], node['params'], ws.horizon_unit)
            done_xlsx += 1
        except Exception as e:
            print(f"  {node['id']:<26} [skip] {e}")
            continue

        finite = np.isfinite(r['peak_p'])
        if finite.any():
            k = int(np.nanargmin(np.where(finite, r['peak_p'], np.nan)))
            floor = 1.0 / (1.0 + r['n_shifts'][k])
            tag = '  [at the floor]' if r['peak_p'][k] <= floor + 1e-12 else ''
            print(f"  {node['id']:<26} best p={r['peak_p'][k]:.5f} at +{r['horizons'][k]}"
                  f"{ws.horizon_unit}  peak {r['peak_real'][k]:5.1f} vs p95 "
                  f"{r['peak_p95'][k]:5.1f}{tag}")
        else:
            print(f"  {node['id']:<26} insufficient data at every horizon")

    print(f"\n  wrote {done_npz} .npz artifacts under {ws.dir.relative_to(ROOT)}/validations/")
    print(f"  wrote {done_xlsx} workbooks under {ws.dir.relative_to(ROOT)}/validations_xlsx/")
    print(f"  verdicts are assigned by --gate, which needs the whole sweep to correct across")


def cmd_gate(ws: Workspace, q: float = 0.05) -> None:
    """
    Stage three's conclusion: correct across the sweep, then intersect with the
    economic filter.

    Verdicts are assigned here rather than stored per node, because a multiple-testing
    correction is a property of the collection. Baking a verdict into a node's artifact
    would silently invalidate it the moment another node joined the sweep.

    The correction is Benjamini-Hochberg. Bonferroni is not available: it needs a
    p-value below q/m, and with m in the hundreds that threshold falls under the
    1/(n+1) floor the circular-shift null can physically reach, so nothing could ever
    clear it however strong the signal.
    """
    ev = ws.read_json(ws.eval_path)
    if not ev:
        print('No evaluation.json - run --cubes first.')
        return

    tree = load_tree(ws.tree_path)
    rows = []
    for node in all_nodes(tree):
        path = ws.validation_path(node['family'], node['id'])
        if not path.exists():
            continue
        r = val.load(path)
        for j, h in enumerate(r['horizons']):
            rows.append({'node': node['id'], 'family': node['family'],
                         'horizon': int(h), 'peak_p': float(r['peak_p'][j]),
                         'peak_real': float(r['peak_real'][j]),
                         'peak_p95': float(r['peak_p95'][j]),
                         'n_shifts': int(r['n_shifts'][j])})
    if not rows:
        print('No validation artifacts - run --validate first.')
        return

    pvals = np.array([r['peak_p'] for r in rows])
    rejected, qvals = val.bh(pvals, q)
    for r, rej, qv in zip(rows, rejected, qvals):
        floor = 1.0 / (1.0 + r['n_shifts']) if r['n_shifts'] else np.nan
        r['q_value'] = None if not np.isfinite(qv) else round(float(qv), 6)
        r['at_floor'] = bool(np.isfinite(r['peak_p']) and np.isfinite(floor)
                             and r['peak_p'] <= floor + 1e-12)
        r['verdict'] = val.verdict(bool(rej), r['peak_p'], r['at_floor'])

    m = int(np.isfinite(pvals).sum())
    n_disc = sum(1 for r in rows if r['verdict'] == 'discovery')
    n_nom  = sum(1 for r in rows if r['verdict'] == 'nominal')
    floors = sorted({r['n_shifts'] for r in rows if r['n_shifts']})
    floor_lo = 1.0 / (1.0 + max(floors)) if floors else float('nan')

    print(f"\n=== Gate [{ws.dir.name}] ===")
    print(f"  {m} tests over {len({r['node'] for r in rows})} nodes x "
          f"{len({r['horizon'] for r in rows})} horizons")
    print(f"  Benjamini-Hochberg at q = {q}   |   p-value floor ~ {floor_lo:.2e} "
          f"(exact null has only n distinct shifts)")
    print(f"  Bonferroni would need p <= {0.05 / max(m, 1):.2e}, "
          f"{'reachable' if 0.05 / max(m, 1) >= floor_lo else 'BELOW the floor - unusable'}\n")
    print(f"  BH discoveries    : {n_disc:>4} / {m}")
    print(f"  nominal (p<=0.05) : {n_nom:>4} / {m}")

    # a node clears when it is economically usable and statistically survives
    passed_econ = set(ev.get('passed', []))
    by_node: dict[str, dict] = {}
    for r in rows:
        if r['verdict'] != 'discovery':
            continue
        cur = by_node.get(r['node'])
        if cur is None or r['q_value'] < cur['q_value']:
            by_node[r['node']] = r

    cleared = []
    for nid, r in by_node.items():
        if nid not in passed_econ:
            continue
        best = ev['nodes'].get(nid, {}).get('best')
        cleared.append({'node': nid, 'family': r['family'],
                        'horizon': r['horizon'], 'q_value': r['q_value'],
                        'peak_p': r['peak_p'], 'peak_real': r['peak_real'],
                        'at_floor': r['at_floor'], 'best_cell': best})
    cleared.sort(key=lambda c: (c['q_value'], -abs(c['best_cell']['dev'] if c['best_cell'] else 0)))

    print(f"\n  economic filter passed : {len(passed_econ)}")
    print(f"  and a BH discovery     : {len(cleared)}\n")
    if cleared:
        print(f"  {'node':<26}{'family':<13}{'h':>5}  {'q':<10}{'peak':>7}  best cell")
        print(f"  {'-'*26}{'-'*13}{'-'*5}  {'-'*10}{'-'*7}  {'-'*36}")
        for c in cleared:
            b = c['best_cell'] or {}
            cell = (f"{b.get('dev', 0):+.1f}pp @ θ={b.get('theta', 0):+.0%} "
                    f"bin {b.get('bin')} n={b.get('bin_n')}") if b else ''
            print(f"  {c['node']:<26}{c['family']:<13}{c['horizon']:>5}  "
                  f"{c['q_value']:<10.5f}{c['peak_real']:>7.1f}  {cell}")
    else:
        print('  Nothing clears both filters.')

    ws.write_json(ws.cleared_path, {
        'workspace': ws.dir.name,
        'generated': datetime.now(timezone.utc).isoformat(),
        'method': {'correction': 'benjamini-hochberg', 'q': q,
                   'n_tests': m, 'p_floor': floor_lo,
                   'note': 'circular-shift null is exact over all n shifts; '
                           'the floor 1/(n+1) is below no Bonferroni threshold at this m, '
                           'so FDR is used instead of FWER'},
        'summary': {'discovery': n_disc, 'nominal': n_nom, 'cleared': len(cleared)},
        'cleared': cleared,
        'tests': rows,
    })
    print(f"\n  wrote {ws.cleared_path.relative_to(ROOT)}")


# ── inventory ─────────────────────────────────────────────────────────────────

def cmd_status(ws: Workspace) -> None:
    tree = load_tree(ws.tree_path)
    ev   = ws.read_json(ws.eval_path).get('nodes', {})
    cl   = ws.read_json(ws.cleared_path)
    cleared = {c['node'] for c in cl.get('cleared', [])}
    disc = {t['node'] for t in cl.get('tests', []) if t.get('verdict') == 'discovery'}

    by_fam = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0])
    for n in all_nodes(tree):
        c = by_fam[n['family']]
        c[0] += 1
        c[1] += bool(ws.has_cube(n['family'], n['id']))
        c[2] += bool(ws.has_surface(n['family'], n['id']))
        c[3] += bool(ws.has_summary_cube(n['family'], n['id']))
        c[4] += bool(ws.has_summary_surface(n['family'], n['id']))
        c[5] += bool(ws.has_validation(n['family'], n['id']))
        c[6] += bool(ws.has_validation_sheet(n['family'], n['id']))
        c[7] += bool(ev.get(n['id'], {}).get('passed'))
        c[8] += n['id'] in cleared

    print(f"\n=== Status [{ws.dir.name}] ===")
    print(f"  θ {ws.thetas[0]:+.0%}..{ws.thetas[-1]:+.0%}   "
          f"horizons +{ws.horizons[0]}{ws.horizon_unit}..+{ws.horizons[-1]}{ws.horizon_unit}   "
          f"{ws.n_bins} bins")
    if cl:
        meth = cl.get('method', {})
        print(f"  gate: BH q={meth.get('q')} over {meth.get('n_tests')} tests, "
              f"{len(disc)} nodes with a discovery")
    print()
    hdr = (f"  {'family':<15}{'nodes':>6}{'cube':>6}{'sheet':>6}"
           f"{'sum':>6}{'sheet':>6}{'valid':>7}{'v-sheet':>8}{'econ':>6}{'cleared':>9}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for fam in sorted(by_fam):
        c = by_fam[fam]
        print(f"  {fam:<15}" + ''.join(f'{v:>6}' for v in c[:5])
              + f"{c[5]:>7}{c[6]:>8}{c[7]:>6}{c[8]:>9}")
    tot = [sum(by_fam[f][i] for f in by_fam) for i in range(9)]
    print('  ' + '-' * (len(hdr) - 2))
    print(f"  {'TOTAL':<15}" + ''.join(f'{v:>6}' for v in tot[:5])
          + f"{tot[5]:>7}{tot[6]:>8}{tot[7]:>6}{tot[8]:>9}")
    print(f"\n  full grid {len(ws.thetas)}θ x {len(ws.horizons)}h   |   "
          f"summary {len(ws.summary_thetas)}θ x {len(ws.summary_horizons)}h "
          f"(judged and validated here)")


def cmd_read(ws: Workspace, node_id: str) -> None:
    """
    Print one node's cube in the terminal: the θ rows carrying a qualifying cell, for
    whichever bin is strongest, with the null test beside them.

    Reads the stored artifacts rather than recomputing, so what prints is exactly what
    the workbook and the validation hold.
    """
    tree = load_tree(ws.tree_path)
    node = find_node(tree, node_id)
    path = ws.cube_path(node['family'], node_id)
    if not path.exists():
        print(f"No cube for {node_id} - run --cubes first.")
        return

    cube   = barrier.load_cube(path)
    prob   = cube['prob']
    thetas = cube['thetas']
    hz     = cube['horizons']
    labels = cube['meta']['bin_labels']

    print()
    print(f"{node_id}  [{node['family']}]  {node['feature']}  params={node['params']}")

    baseline = _baseline_surface(ws)
    if baseline is None:
        print('  no baseline cube - run --cubes first')
        return
    ev   = barrier.evaluate(cube, baseline, ws.min_dev, ws.min_bin_n, ws.min_run)
    best = ev['best']
    if not best:
        print(f"  no cell reaches {ws.min_dev}pp across {ws.min_run} adjacent θ rows")
        return

    b = best['bin']
    print(f"  strongest bin : {b + 1} of {len(labels)}   ({labels[b]})")
    print(f"  strongest cell: P={best['prob']:.1%} vs {best['base']:.1%} unconditional "
          f"at θ={best['theta']:+.0%}, +{best['horizon']}{ws.horizon_unit} "
          f"({best['dev']:+.1f}pp, run={best['run']}, n={best['bin_n']})")

    vpath = ws.validation_path(node['family'], node_id)
    if vpath.exists():
        vr = val.load(vpath)
        j  = int(np.flatnonzero(vr['horizons'] == best['horizon'])[0])
        floor = 1.0 / (1.0 + vr['n_shifts'][j])
        cp = vr['cell_p'][:, b, j]
        tag = ' (at the floor)' if vr['peak_p'][j] <= floor + 1e-12 else ''
        print(f"  null          : peak {vr['peak_real'][j]:.1f} vs p95 {vr['peak_p95'][j]:.1f}, "
              f"p={vr['peak_p'][j]:.5f}{tag}   floor {floor:.2e}")
        print(f"                  {int(np.nansum(cp <= 0.05))} of {int(np.isfinite(cp).sum())} "
              f"cells in this bin are pointwise p<=0.05")
    else:
        print('  null          : not validated yet')

    avail = {int(x) for x in hz}
    show  = [h for h in (1, 2, 3, 5, 7, 10, 14, 21, 30) if h in avail]
    cols  = [int(np.flatnonzero(hz == h)[0]) for h in show]
    dev   = (prob - baseline[:, None, :]) * 100.0

    print()
    header = ''.join(f"{'+' + str(h) + ws.horizon_unit:>8}" for h in show)
    print(f"  {'θ':>6}{header}")
    print('  ' + '-' * (6 + 8 * len(show)))
    shown = 0
    for i in sorted(range(len(thetas)), key=lambda k: -thetas[k]):
        d = dev[i, b, cols]
        finite = d[np.isfinite(d)]
        if finite.size == 0 or np.max(np.abs(finite)) < ws.min_dev:
            continue
        cells = ''.join('       -' if not np.isfinite(v) else f'{v:>8.1%}'
                        for v in prob[i, b, cols])
        print(f"  {thetas[i]:>+6.0%}{cells}")
        shown += 1
    if not shown:
        print(f"  no θ row deviates by {ws.min_dev}pp at these horizons")
    print()
    print(f"  values are P(touch θ | condition); rows shown are those deviating "
          f"at least {ws.min_dev}pp")
    print(f"  the workbook carries all {len(thetas)} θ levels, "
          f"{len(hz)} horizons and {len(labels)} bins")


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Barrier-touch pipeline: P(price reaches θ within h | condition), '
                    'measured on a full grid and judged on a coarse one.')
    parser.add_argument('--workspace', metavar='NAME', default='btc_daily_14days')
    parser.add_argument('--family', metavar='NAME', help='Restrict to one family')
    parser.add_argument('--rerun', action='store_true', help='Rebuild cubes that already exist')
    parser.add_argument('--fdr', type=float, default=0.05, metavar='Q',
                        help='Benjamini-Hochberg false-discovery rate for --gate (default 0.05)')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--cubes',           action='store_true', help='1. measure: full θ x bin x horizon cube per node')
    g.add_argument('--surfaces',        action='store_true', help='2. render the full cubes')
    g.add_argument('--cubes-summary',   action='store_true', dest='cubes_summary',
                   help='3. select the coarse grid everything is judged on; runs the economic filter')
    g.add_argument('--surfaces-summary', action='store_true', dest='surfaces_summary',
                   help='4. render the summary cubes')
    g.add_argument('--validate', action='store_true',
                   help='5. null-test every summary cube, writing .npz and readable .xlsx artifacts')
    g.add_argument('--gate',     action='store_true', help='6. correct across the sweep (BH) and intersect with the economic filter')
    g.add_argument('--status',   action='store_true', help='Inventory by family')
    g.add_argument('--read',     metavar='ID', help='Print a node surface summary')

    args = parser.parse_args()
    ws = Workspace(args.workspace)

    if args.cubes:
        cmd_cubes(ws, args.family, args.rerun)
    elif args.surfaces:
        cmd_surfaces(ws, args.family)
    elif args.cubes_summary:
        cmd_cubes_summary(ws, args.family)
    elif args.surfaces_summary:
        cmd_surfaces_summary(ws, args.family)
    elif args.validate:
        cmd_validate(ws, args.family, args.rerun)
    elif args.gate:
        cmd_gate(ws, args.fdr)
    elif args.status:
        cmd_status(ws)
    elif args.read:
        cmd_read(ws, args.read)
    else:
        parser.print_help()
