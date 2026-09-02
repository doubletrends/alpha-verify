"""
Barrier-touch pipeline.

            --cubes      the measurement: per node, a (θ x bin x horizon) cube of
                         P(touch θ in h | bin), saved as .npz.
            --surfaces   Deliverable A: projects a few horizon slices of each cube into
                         a workbook. Renders only; it never re-measures.
            --validate   stage three: an exact circular-shift null per node, by FFT,
                         written as one artifact per node beside its cube.
            --gate       correct across the sweep with Benjamini-Hochberg, intersect
                         with the economic filter, write what clears both.

The gate is the conclusion, not a hand-off: a node clears it only by passing both the
economic filter and the null test. The economic one asks whether the edge is large and
steady enough to be worth anything; the statistical one asks whether it is there at
all. Either alone is a way to be confidently wrong.

Usage:
    python run.py --workspace btc_daily_14days --cubes
    python run.py --workspace btc_daily_14days --surfaces
    python run.py --workspace btc_daily_14days --validate
    python run.py --workspace btc_daily_14days --gate
    python run.py --workspace btc_daily_14days --status
    python run.py --workspace btc_daily_14days --read bb_pct_20
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


def build_cube(ws: Workspace, node: dict, get_data, quiet: bool = False) -> dict:
    """Measure one node across every horizon and barrier level; write the .npz."""
    data, feat = _node_feature(ws, node, get_data)
    edges = barrier.bin_edges(feat, ws.n_bins)
    cube  = barrier.touch_tensor(data, feat, ws.horizons, ws.thetas, edges)

    barrier.save_cube(cube, ws.cube_path(node['family'], node['id']), {
        'node': node['id'], 'family': node['family'],
        'feature': node['feature'], 'params': node['params'],
        'workspace': ws.dir.name,
        'bin_labels': barrier.bin_labels(edges, feat),
        'generated': datetime.now(timezone.utc).isoformat(),
    })
    baseline = _baseline_surface(ws)
    if baseline is None:
        return {'passed': False, 'best': None, 'per_horizon': {}}
    ev = barrier.evaluate(cube, baseline, ws.min_dev, ws.min_bin_n, ws.min_run)
    if not quiet:
        b = ev['best']
        note = (f"best {b['dev']:+5.1f}pp  θ={b['theta']:+.0%}  "
                f"+{b['horizon']}{ws.horizon_unit:<3} bin {b['bin']}  "
                f"n={b['bin_n']}  run={b['run']}") if b else 'no qualifying cell'
        print(f"  {node['id']:<26} {note}")
    return ev


def cmd_cubes(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    tree  = load_tree(ws.tree_path)
    nodes = all_in_family(tree, family) if family else all_nodes(tree)
    if not rerun:
        nodes = [n for n in nodes if not ws.has_cube(n['family'], n['id'])]
    if not nodes:
        print('Nothing to do (use --rerun to rebuild existing cubes).')
        return

    # The baseline is the reference every other node is judged against, so it is built
    # first -- and rebuilt whenever anything else is, since a stale one would silently
    # shift every comparison.
    nodes = ([n for n in nodes if n['id'] == BASELINE_NODE]
             + [n for n in nodes if n['id'] != BASELINE_NODE])
    if not any(n['id'] == BASELINE_NODE for n in nodes) and not ws.baseline_cube.exists():
        print(f"No baseline cube yet. Run --cubes without --family, or add a "
              f"'{BASELINE_NODE}' node, before evaluating conditions.")
        return

    get_data = _loader(ws)
    hz = ws.horizons
    print(f"\n=== Cubes [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(f"  {len(ws.thetas)} θ x {ws.n_bins} bins x {len(hz)} horizons "
          f"(+{hz[0]}{ws.horizon_unit}..+{hz[-1]}{ws.horizon_unit})   "
          f"θ {ws.thetas[0]:+.0%}..{ws.thetas[-1]:+.0%} step {ws.theta_step:.0%}")
    print(f"  value = P(touch θ in h | bin); intraday high/low")
    print(f"  the '{BASELINE_NODE}' node is the unconditional reference, built first\n")

    evals, skipped = {}, {}
    for node in nodes:
        try:
            evals[node['id']] = build_cube(ws, node, get_data)
        except Exception as e:
            skipped[node['id']] = str(e)
            print(f"  {node['id']:<26} [skip] {e}")

    ws.write_json(ws.eval_path, {
        'workspace': ws.dir.name,
        'generated': datetime.now(timezone.utc).isoformat(),
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


# -- Deliverable A: a projection of the cube ----------------------------------

def cmd_surfaces(ws: Workspace, family: str | None = None) -> None:
    """
    Render the workbook from the cube: one tab per condition bin, each holding that
    bin's whole (θ x horizon) face.

    Every value in the cube appears in the workbook, so this is a faithful view of the
    measurement rather than a summary of it. No measurement happens here.
    """
    tree  = load_tree(ws.tree_path)
    nodes = [n for n in (all_in_family(tree, family) if family else all_nodes(tree))
             if ws.has_cube(n['family'], n['id'])]
    if not nodes:
        print('No cubes to project - run --cubes first.')
        return

    print(f"\n=== Deliverable A [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(f"  {ws.n_bins} tabs per node, one per condition bin; each tab is that bin's "
          f"full θ x horizon face\n")
    n_ok = 0
    for node in nodes:
        cube = barrier.load_cube(ws.cube_path(node['family'], node['id']))
        try:
            writer.write_barrier_xlsx(
                cube, ws.surface_path(node['family'], node['id']),
                node['id'], node['feature'], node['params'], ws.horizon_unit)
        except PermissionError:
            # Windows locks a workbook that is open in Excel. Skip it rather than
            # abandoning the run -- the cube is already written, so re-running the
            # projection once the file is closed costs nothing.
            print(f"  {node['id']:<26} [locked] close it in Excel and re-run --surfaces")
            continue
        n_ok += 1
    print(f"  wrote {n_ok} workbooks under {ws.dir.relative_to(ROOT)}/surfaces/")


# ── Validation ────────────────────────────────────────────────────────────────

def cmd_validate(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """
    Stage three, per node: null-test the cube and write an artifact beside it.

    Every node gets one, the baseline included. Shuffling a constant feature cannot
    change anything, so its artifact comes out degenerate -- peak deviation 0, p = 1 --
    and that is deliberate: a uniform pipeline with one meaningless file beats a
    pipeline with a special case in it.

    Per-node artifacts also mean two runs never write the same path, which is the
    failure that made the previous shared validation.json untrustworthy.
    """
    tree  = load_tree(ws.tree_path)
    nodes = [n for n in (all_in_family(tree, family) if family else all_nodes(tree))
             if ws.has_cube(n['family'], n['id'])]
    if not rerun:
        nodes = [n for n in nodes if not ws.has_validation(n['family'], n['id'])]
    if not nodes:
        print('Nothing to validate (use --rerun to redo existing artifacts).')
        return

    get_data = _loader(ws)
    print(f"\n=== Validation [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(f"  exact circular-shift null over every shift, by FFT; "
          f"{len(ws.thetas)} θ x {ws.n_bins} bins x {len(ws.horizons)} horizons per node")
    print(f"  guard {val.EDGE_GUARD} bars either side, so the p-value floor is 1/(usable+1)\n")

    done = 0
    for node in nodes:
        try:
            data, feat = _node_feature(ws, node, get_data)
            cube = barrier.load_cube(ws.cube_path(node['family'], node['id']))
        except Exception as e:
            print(f"  {node['id']:<26} [skip] {e}")
            continue

        # the cube's own edges, so validation partitions the sample identically
        r = val.validate_node(data, feat, ws.horizons, ws.thetas, cube['edges'])
        val.save(r, ws.validation_path(node['family'], node['id']), {
            'node': node['id'], 'family': node['family'],
            'feature': node['feature'], 'params': node['params'],
            'workspace': ws.dir.name,
            'bin_labels': cube['meta']['bin_labels'],
            'generated': datetime.now(timezone.utc).isoformat(),
        })
        done += 1

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

    print(f"\n  wrote {done} artifacts under {ws.dir.relative_to(ROOT)}/validations/")
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

    by_fam = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    for n in all_nodes(tree):
        c = by_fam[n['family']]
        c[0] += 1
        c[1] += bool(ws.has_cube(n['family'], n['id']))
        c[2] += bool(ws.has_surface(n['family'], n['id']))
        c[3] += bool(ws.has_validation(n['family'], n['id']))
        c[4] += bool(ev.get(n['id'], {}).get('passed'))
        c[5] += n['id'] in cleared

    print(f"\n=== Status [{ws.dir.name}] ===")
    print(f"  θ {ws.thetas[0]:+.0%}..{ws.thetas[-1]:+.0%}   "
          f"horizons +{ws.horizons[0]}{ws.horizon_unit}..+{ws.horizons[-1]}{ws.horizon_unit}   "
          f"{ws.n_bins} bins")
    if cl:
        meth = cl.get('method', {})
        print(f"  gate: BH q={meth.get('q')} over {meth.get('n_tests')} tests, "
              f"{len(disc)} nodes with a discovery")
    print()
    print(f"  {'family':<16}{'nodes':>7}{'cube':>7}{'sheet':>7}{'valid':>7}{'econ':>7}{'cleared':>9}")
    print('  ' + '-' * 60)
    for fam in sorted(by_fam):
        c = by_fam[fam]
        print(f"  {fam:<16}{c[0]:>7}{c[1]:>7}{c[2]:>7}{c[3]:>7}{c[4]:>7}{c[5]:>9}")
    tot = [sum(by_fam[f][i] for f in by_fam) for i in range(6)]
    print('  ' + '-' * 60)
    print(f"  {'TOTAL':<16}{tot[0]:>7}{tot[1]:>7}{tot[2]:>7}{tot[3]:>7}{tot[4]:>7}{tot[5]:>9}")


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
                    'then what that is worth as a trade.')
    parser.add_argument('--workspace', metavar='NAME', default='btc_daily_14days')
    parser.add_argument('--family', metavar='NAME', help='Restrict to one family')
    parser.add_argument('--rerun', action='store_true', help='Rebuild cubes that already exist')
    parser.add_argument('--fdr', type=float, default=0.05, metavar='Q',
                        help='Benjamini-Hochberg false-discovery rate for --gate (default 0.05)')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--cubes',    action='store_true', help='Measure every node: θ x bin x horizon (pre-A)')
    g.add_argument('--surfaces', action='store_true', help='Project cubes into Deliverable A workbooks')
    g.add_argument('--validate', action='store_true', help='Stage 3: null-test every cube, one artifact per node')
    g.add_argument('--gate',     action='store_true', help='Correct across the sweep (BH) and intersect with the economic filter')
    g.add_argument('--status',   action='store_true', help='Inventory by family')
    g.add_argument('--read',     metavar='ID', help='Print a node surface summary')

    args = parser.parse_args()
    ws = Workspace(args.workspace)

    if args.cubes:
        cmd_cubes(ws, args.family, args.rerun)
    elif args.surfaces:
        cmd_surfaces(ws, args.family)
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
