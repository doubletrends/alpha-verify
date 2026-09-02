"""
Barrier-touch pipeline.

            --cubes      the measurement: per node, a (θ x bin x horizon) cube of
                         P(touch θ in h | bin), saved as .npz.
            --surfaces   Deliverable A: projects a few horizon slices of each cube into
                         a workbook. Renders only; it never re-measures.
            --validate   null-test every node's surface.
            --gate       apply the economic filter and the null test together; write
                         the nodes that clear both.

The gate is the conclusion, not a hand-off: a node clears it only by passing both the
economic filter and the null test. The economic one asks whether the edge is large and
steady enough to be worth anything; the statistical one asks whether it is there at
all. Either alone is a way to be confidently wrong.

Usage:
    python run.py --workspace btc_daily_14days --cubes
    python run.py --workspace btc_daily_14days --surfaces
    python run.py --workspace btc_daily_14days --validate --shifts 20000
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
from engine import barrier, writer
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

def cmd_validate(ws: Workspace, family: str | None = None, n_shifts: int = 20000) -> None:
    tree  = load_tree(ws.tree_path)
    # The baseline carries no condition, so shuffling its feature changes nothing and
    # a null test on it is vacuous. Excluding it also keeps it out of the Bonferroni
    # denominator, where it would only make every real test harder to pass.
    nodes = [n for n in (all_in_family(tree, family) if family else all_nodes(tree))
             if ws.has_cube(n['family'], n['id']) and n['id'] != BASELINE_NODE]
    if not nodes:
        print('No cubes to validate — run --surfaces first.')
        return

    get_data = _loader(ws)
    n_tests  = len(nodes) * len(ws.barrier_horizons)
    print(f"\n=== Validation [{ws.dir.name}] ===")
    print(f"  sweep : {len(nodes)} nodes x {len(ws.barrier_horizons)} horizons = {n_tests} tests, "
          f"{n_shifts:,} feature shuffles each")
    print(f"  alpha : 0.05 / {n_tests} = {0.05 / n_tests:.2e} (Bonferroni)\n")
    hdr = ''.join(f'{f"+{h}{ws.horizon_unit}":>24}' for h in ws.barrier_horizons)
    print(f"  {'node':<26}{hdr}")
    print(f"  {'-' * 26}{''.join('-' * 24 for _ in ws.barrier_horizons)}")

    results = {}
    for node in nodes:
        try:
            data, feat = _node_feature(ws, node, get_data)
        except Exception as e:
            print(f"  {node['id']:<26}  [skip] {e}")
            continue
        cells, per_h = [], {}
        for h in ws.barrier_horizons:
            r = barrier.null_test(data, feat, h, ws.thetas, ws.n_bins, n_shifts=n_shifts)
            r['verdict'] = barrier.verdict(r['p_value'], n_tests=n_tests,
                                           p_floor=r.get('p_floor', 0.0))
            per_h[f'+{h}{ws.horizon_unit}'] = r
            if pd.isna(r['real']):
                cells.append(f"{'insufficient':>24}")
            else:
                mark = {'structure': '**', 'nominal': ' *', 'underpowered': ' ?',
                        'noise': '  ', 'insufficient': '  '}[r['verdict']]
                cells.append(f"{r['real']:>7.1f} vs{r['null_p95']:>5.1f} p={r['p_value']:<6.4f}{mark}")
        results[node['id']] = {'family': node['family'], 'horizons': per_h}
        print(f"  {node['id']:<26}{''.join(cells)}")

    # Merge into any prior sweep, so validating one family never discards the rest.
    # Bonferroni is then applied over every test the merged file holds -- narrowing
    # --family must not quietly shrink the correction for a search already run wide.
    prior = ws.read_json(ws.validation_path)
    merged = prior.get('nodes', {}) if prior else {}
    merged.update(results)
    results = merged

    flat = [r for n in results.values() for r in n['horizons'].values()]
    if len(flat) != n_tests:
        for n in results.values():
            for r in n['horizons'].values():
                r['verdict'] = barrier.verdict(r['p_value'], n_tests=len(flat),
                                               p_floor=r.get('p_floor', 0.0))
        print(f"\n  (merged with earlier runs: Bonferroni now over {len(flat)} tests)")

    n_st = sum(1 for r in flat if r['verdict'] == 'structure')
    n_no = sum(1 for r in flat if r['verdict'] == 'nominal')
    print()
    print('  columns: real peak  vs  null p95   p-value')
    print('  ** clears Bonferroni   * nominal only   ? at the resolution floor\n')
    print(f"  clears Bonferroni : {n_st:>4} / {len(flat)}")
    print(f"  nominal (p<0.05)  : {n_no:>4} / {len(flat)}   (expected by chance ~{0.05 * len(flat):.1f})")

    ws.write_json(ws.validation_path, {
        'workspace': ws.dir.name,
        'generated': datetime.now(timezone.utc).isoformat(),
        'n_shifts': n_shifts,
        'n_tests': len(flat),
        'alpha_bonf': 0.05 / max(len(flat), 1),
        'summary': {'structure': n_st, 'nominal': n_no, 'total': len(flat)},
        'nodes': results,
    })
    print(f"\n  wrote {ws.validation_path.relative_to(ROOT)}")


# ── The gate ──────────────────────────────────────────────────────────────────

_RANK = {'structure': 3, 'nominal': 2, 'underpowered': 1, 'noise': 0, 'insufficient': 0}


def cmd_gate(ws: Workspace, require: str = 'structure') -> None:
    """
    Intersect the economic and statistical filters; write the cleared.

    Both are required. A node with a large, steady edge that does not clear its null is
    a shape found by searching; a node that clears its null on a two-point edge is real
    and unusable. Round 2 is only worth running on nodes that are both.
    """
    ev  = ws.read_json(ws.eval_path)
    val = ws.read_json(ws.validation_path)
    if not ev or not val:
        print('Need both --surfaces and --validate first.')
        return
    need = _RANK[require]

    cleared = []
    for nid, rec in ev.get('nodes', {}).items():
        if not rec.get('passed'):
            continue
        hz = val.get('nodes', {}).get(nid, {}).get('horizons', {})
        best = None
        for h, r in hz.items():
            k = _RANK.get(r.get('verdict', 'noise'), 0)
            if best is None or k > best[1] or (k == best[1] and r.get('p_value', 1) < best[2]):
                best = (h, k, r.get('p_value', 1.0), r.get('verdict', 'noise'))
        if best and best[1] >= need:
            cleared.append({'node': nid, 'family': val['nodes'][nid]['family'],
                              'best_cell': rec['best'], 'validated_horizon': best[0],
                              'verdict': best[3], 'p_value': best[2]})

    cleared.sort(key=lambda s: s['p_value'])
    print(f"\n=== Gate [{ws.dir.name}] ===")
    print(f"  economic filter passed  : {len(ev.get('passed', []))}")
    print(f"  and reaching '{require}' : {len(cleared)}\n")
    if cleared:
        print(f"  {'node':<26}{'family':<13}{'horizon':<9}{'verdict':<11}{'p':<10}best cell")
        print(f"  {'-'*26}{'-'*13}{'-'*9}{'-'*11}{'-'*10}{'-'*34}")
        for s in cleared:
            b = s['best_cell']
            print(f"  {s['node']:<26}{s['family']:<13}{s['validated_horizon']:<9}"
                  f"{s['verdict']:<11}{s['p_value']:<10.5f}"
                  f"{b['dev']:+.1f}pp @ θ={b['theta']:+.0%} bin {b['bin']} n={b['bin_n']}")
    else:
        print('  Nothing survives both gates. Round 2 has nothing to evaluate.')

    ws.write_json(ws.cleared_path, {
        'workspace': ws.dir.name,
        'generated': datetime.now(timezone.utc).isoformat(),
        'require': require, 'cleared': cleared,
    })
    print(f"\n  wrote {ws.cleared_path.relative_to(ROOT)}")


# ── inventory ─────────────────────────────────────────────────────────────────

def cmd_status(ws: Workspace) -> None:
    tree = load_tree(ws.tree_path)
    ev   = ws.read_json(ws.eval_path).get('nodes', {})
    val  = ws.read_json(ws.validation_path).get('nodes', {})
    surv = {s['node'] for s in ws.read_json(ws.cleared_path).get('cleared', [])}

    by_fam = defaultdict(lambda: [0, 0, 0, 0, 0])
    for n in all_nodes(tree):
        c = by_fam[n['family']]
        c[0] += 1
        c[1] += bool(ws.has_cube(n['family'], n['id']))
        c[2] += bool(ev.get(n['id'], {}).get('passed'))
        c[3] += any(r.get('verdict') in ('structure', 'nominal')
                    for r in val.get(n['id'], {}).get('horizons', {}).values())
        c[4] += n['id'] in surv

    print(f"\n=== Status [{ws.dir.name}] ===")
    print(f"  θ {ws.thetas[0]:+.0%}..{ws.thetas[-1]:+.0%}   "
          f"horizons {', '.join(f'+{h}{ws.horizon_unit}' for h in ws.barrier_horizons)}   "
          f"{ws.n_bins} bins\n")
    print(f"  {'family':<16}{'nodes':>7}{'cube':>7}{'econ':>7}{'signif':>8}{'cleared':>10}")
    print('  ' + '-' * 57)
    for fam in sorted(by_fam):
        c = by_fam[fam]
        print(f"  {fam:<16}{c[0]:>7}{c[1]:>7}{c[2]:>7}{c[3]:>8}{c[4]:>10}")
    tot = [sum(by_fam[f][i] for f in by_fam) for i in range(5)]
    print('  ' + '-' * 57)
    print(f"  {'TOTAL':<16}{tot[0]:>7}{tot[1]:>7}{tot[2]:>7}{tot[3]:>8}{tot[4]:>10}")


def cmd_read(ws: Workspace, node_id: str) -> None:
    """
    Print one node's cube in the terminal: the θ rows carrying a qualifying cell,
    for whichever bin is strongest.

    Reads the stored cube rather than recomputing, so what prints is exactly what the
    workbook shows.
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
    val    = ws.read_json(ws.validation_path).get('nodes', {}).get(node_id, {}).get('horizons', {})

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
    if val:
        for h, r in val.items():
            print(f"  null {h:<5}    : {r.get('verdict')}  p={r.get('p_value')}  "
                  f"real={r.get('real')} vs p95={r.get('null_p95')}")

    avail = {int(x) for x in hz}
    show  = [h for h in (1, 2, 3, 5, 7, 10, 14, 21, 30) if h in avail]
    cols  = [int(np.flatnonzero(hz == h)[0]) for h in show]

    dev = (prob - baseline[:, None, :]) * 100.0
    print()
    header = ''.join(f"{'+' + str(h) + ws.horizon_unit:>8}" for h in show)
    print(f"  {'θ':>6}{header}")
    print('  ' + '-' * (6 + 8 * len(show)))
    shown = 0
    # highest barrier first, matching the workbook
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
    parser.add_argument('--shifts', type=int, default=20000, metavar='N',
                        help='Shuffles per --validate test (default 20000)')
    parser.add_argument('--require', choices=['structure', 'nominal'], default='structure',
                        help='Verdict a node must reach at --gate (default structure)')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--cubes',    action='store_true', help='Measure every node: θ x bin x horizon (pre-A)')
    g.add_argument('--surfaces', action='store_true', help='Project cubes into Deliverable A workbooks')
    g.add_argument('--validate', action='store_true', help='Null-test every surface')
    g.add_argument('--gate',     action='store_true', help='Intersect the economic and null filters')
    g.add_argument('--status',   action='store_true', help='Inventory by family')
    g.add_argument('--read',     metavar='ID', help='Print a node surface summary')

    args = parser.parse_args()
    ws = Workspace(args.workspace)

    if args.cubes:
        cmd_cubes(ws, args.family, args.rerun)
    elif args.surfaces:
        cmd_surfaces(ws, args.family)
    elif args.validate:
        cmd_validate(ws, args.family, n_shifts=args.shifts)
    elif args.gate:
        cmd_gate(ws, args.require)
    elif args.status:
        cmd_status(ws)
    elif args.read:
        cmd_read(ws, args.read)
    else:
        parser.print_help()
