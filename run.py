"""
Barrier-touch pipeline.

            --surfaces   Deliverable A, one workbook per node: for each horizon, the
                         probability that price touches each barrier level theta, given
                         each condition bin.
            --validate   null-test every node's surface.
            --gate       apply the economic filter and the null test together; write
                         the nodes that clear both.

The gate is the conclusion, not a hand-off: a node clears it only by passing both the
economic filter and the null test. The economic one asks whether the edge is large and
steady enough to be worth anything; the statistical one asks whether it is there at
all. Either alone is a way to be confidently wrong.

Usage:
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
from workspace import Workspace


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

def build_surface(ws: Workspace, node: dict, get_data, quiet: bool = False) -> dict:
    """Compute and write one node's barrier workbook; return its economic evaluation."""
    data, feat = _node_feature(ws, node, get_data)
    edges  = barrier.bin_edges(feat, ws.n_bins)
    labels = barrier.bin_labels(edges, feat)

    surfaces = {f'+{h}{ws.horizon_unit}': barrier.touch_matrix(data, feat, h, ws.thetas, edges)
                for h in ws.barrier_horizons}

    writer.write_barrier_xlsx(surfaces, labels, ws.surface_path(node['family'], node['id']),
                              node['id'], node['feature'], node['params'])
    ev = barrier.evaluate(surfaces, ws.min_dev, ws.min_bin_n, ws.min_run)
    if not quiet:
        b = ev['best']
        note = (f"best {b['dev']:+5.1f}pp  theta={b['theta']:+.0%}  {b['horizon']:<5} "
                f"bin {b['bin']}  n={b['bin_n']}  run={b['run']}") if b else 'no qualifying cell'
        print(f"  {node['id']:<26} {note}")
    return ev


def cmd_surfaces(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    tree  = load_tree(ws.tree_path)
    nodes = all_in_family(tree, family) if family else all_nodes(tree)
    if not rerun:
        nodes = [n for n in nodes if not ws.has_surface(n['family'], n['id'])]
    if not nodes:
        print('Nothing to do (use --rerun to rebuild existing surfaces).')
        return

    get_data = _loader(ws)
    hz = ', '.join(f'+{h}{ws.horizon_unit}' for h in ws.barrier_horizons)
    print(f"\n=== Deliverable A [{ws.dir.name}] — {len(nodes)} nodes ===")
    print(f"  horizons {hz}   theta {ws.thetas[0]:+.0%}..{ws.thetas[-1]:+.0%} "
          f"step {ws.theta_step:.0%}   {ws.n_bins} quantile bins   intraday high/low\n")

    evals, skipped = {}, {}
    for node in nodes:
        try:
            evals[node['id']] = build_surface(ws, node, get_data)
        except Exception as e:
            skipped[node['id']] = str(e)
            print(f"  {node['id']:<26} [skip] {e}")

    ws.write_json(ws.eval_path, {
        'workspace': ws.dir.name,
        'generated': datetime.now(timezone.utc).isoformat(),
        'criteria': {'min_dev': ws.min_dev, 'min_bin_n': ws.min_bin_n, 'min_run': ws.min_run},
        'passed':  sorted(n for n, e in evals.items() if e['passed']),
        'nodes':   {n: {'passed': e['passed'], 'best': e['best']} for n, e in evals.items()},
        'skipped': skipped,
    })
    n_pass = sum(1 for e in evals.values() if e['passed'])
    print(f"\n  economic filter: {n_pass} / {len(evals)} pass "
          f"(|dev| >= {ws.min_dev}pp across >= {ws.min_run} adjacent theta rows, bin n >= {ws.min_bin_n})")
    print(f"  wrote {ws.eval_path.relative_to(ROOT)}")


# ── Validation ────────────────────────────────────────────────────────────────

def cmd_validate(ws: Workspace, family: str | None = None, n_shifts: int = 20000) -> None:
    tree  = load_tree(ws.tree_path)
    nodes = [n for n in (all_in_family(tree, family) if family else all_nodes(tree))
             if ws.has_surface(n['family'], n['id'])]
    if not nodes:
        print('No surfaces to validate — run --surfaces first.')
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

    flat = [r for n in results.values() for r in n['horizons'].values()]
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
                  f"{b['dev']:+.1f}pp @ theta={b['theta']:+.0%} bin {b['bin']} n={b['bin_n']}")
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
        c[1] += bool(ws.has_surface(n['family'], n['id']))
        c[2] += bool(ev.get(n['id'], {}).get('passed'))
        c[3] += any(r.get('verdict') in ('structure', 'nominal')
                    for r in val.get(n['id'], {}).get('horizons', {}).values())
        c[4] += n['id'] in surv

    print(f"\n=== Status [{ws.dir.name}] ===")
    print(f"  theta {ws.thetas[0]:+.0%}..{ws.thetas[-1]:+.0%}   "
          f"horizons {', '.join(f'+{h}{ws.horizon_unit}' for h in ws.barrier_horizons)}   "
          f"{ws.n_bins} bins\n")
    print(f"  {'family':<16}{'nodes':>7}{'surface':>9}{'econ':>7}{'signif':>8}{'cleared':>10}")
    print('  ' + '-' * 57)
    for fam in sorted(by_fam):
        c = by_fam[fam]
        print(f"  {fam:<16}{c[0]:>7}{c[1]:>9}{c[2]:>7}{c[3]:>8}{c[4]:>10}")
    tot = [sum(by_fam[f][i] for f in by_fam) for i in range(5)]
    print('  ' + '-' * 57)
    print(f"  {'TOTAL':<16}{tot[0]:>7}{tot[1]:>9}{tot[2]:>7}{tot[3]:>8}{tot[4]:>10}")


def cmd_read(ws: Workspace, node_id: str) -> None:
    """Print the theta rows of a node's surface that carry a qualifying cell."""
    tree = load_tree(ws.tree_path)
    node = find_node(tree, node_id)
    data, feat = _node_feature(ws, node, _loader(ws))
    edges  = barrier.bin_edges(feat, ws.n_bins)
    labels = barrier.bin_labels(edges, feat)
    val    = ws.read_json(ws.validation_path).get('nodes', {}).get(node_id, {}).get('horizons', {})

    print(f"\n{node_id}  [{node['family']}]  {node['feature']}  params={node['params']}")
    for h in ws.barrier_horizons:
        label = f'+{h}{ws.horizon_unit}'
        s = barrier.touch_matrix(data, feat, h, ws.thetas, edges)
        v = val.get(label, {})
        tag = f"   [{v.get('verdict')}, p={v.get('p_value')}]" if v else ''
        print(f"\n  {label}{tag}")
        print(f"    {'theta':>7}{'base':>8}   " + ''.join(f'{l[:11]:>13}' for l in labels))
        shown = 0
        for i, th in enumerate(s['thetas']):
            row = s['dev'][i]
            finite = row[~np.isnan(row)]
            if finite.size == 0 or np.max(np.abs(finite)) < ws.min_dev:
                continue
            cells = ''.join('            —' if np.isnan(x) else f'{x:>+13.1f}' for x in row)
            print(f"    {th:>+7.0%}{s['base'][i]:>8.1%}   {cells}")
            shown += 1
        if not shown:
            print(f"    no cell reaches {ws.min_dev}pp")


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Barrier-touch pipeline: P(price reaches theta within h | condition), '
                    'then what that is worth as a trade.')
    parser.add_argument('--workspace', metavar='NAME', default='btc_daily_14days')
    parser.add_argument('--family', metavar='NAME', help='Restrict to one family')
    parser.add_argument('--rerun', action='store_true', help='Rebuild surfaces that already exist')
    parser.add_argument('--shifts', type=int, default=20000, metavar='N',
                        help='Shuffles per --validate test (default 20000)')
    parser.add_argument('--require', choices=['structure', 'nominal'], default='structure',
                        help='Verdict a node must reach at --gate (default structure)')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--surfaces', action='store_true', help='Build Deliverable A')
    g.add_argument('--validate', action='store_true', help='Null-test every surface')
    g.add_argument('--gate',     action='store_true', help='Intersect the economic and null filters')
    g.add_argument('--status',   action='store_true', help='Inventory by family')
    g.add_argument('--read',     metavar='ID', help='Print a node surface summary')

    args = parser.parse_args()
    ws = Workspace(args.workspace)

    if args.surfaces:
        cmd_surfaces(ws, args.family, args.rerun)
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
