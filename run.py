"""
Barrier-touch pipeline.

    --surface           1. surface     01_surface_array/    P(touch θ in h | bin)
                                       01_surface_xlsx/     readable full surface workbooks
    --summary           2. summary     02_summary_array/    coarse grid, + economic filter
                                       02_summary_xlsx/     readable summary workbooks
    --skew              3. skew        03_skew_array/       P(+θ) vs P(−θ): conditional & excess
                                       03_skew_xlsx/        readable skew sheets
    --validation        4. validation  04_validation_array/ exact shuffled null (touch + skew peak)
                                       04_validation_xlsx/  readable validation sheets
    --gate              5. correct     05_gate.json         Benjamini-Hochberg + economic
    --bayes             6. compose     06_bayez.npz + 06_bayes.json
                                                     walk-forward naive Bayes, out of sample
    --report            7. render      result/              the audience-facing figures

The full cube is the faithful record and is never judged. The summary is a strict
subset of it -- index selection, no interpolation -- chosen so adjacent cells are
genuinely different measurements. Everything that judges runs there: a peak over the
coarse grid has a far lower noise ceiling than one over the full grid, and counting
those cells as tests is an honest measure of how wide the search actually was.

Skew is the one thing the mirrored summary ladder is built for: P(touch +θ) read
against P(touch -θ) in the same column. --skew writes it as an artifact (conditional,
and excess of the baseline node's own drift); --validation null-tests its peak alongside
the touch peak; --gate corrects it as a separate BH family. It is reported, not gated:
what clears is still touch-peak discovery intersected with the economic filter.

Every node passes through every stage, the baseline included. Shuffling a constant
cannot change anything, so its validation is degenerate by construction -- one
meaningless artifact is cheaper than a branch in the pipeline.

Stages 1-5 measure over the whole history, which describes what happened and claims
nothing about what happens next. Stage 6 is the only one that predicts, and the only one
whose every quantity -- edges, rates, prior, scale -- is fit on a training window and
applied to a later one. Stage 7 renders; it never measures.

Usage:
    python run.py --surface
    python run.py --summary
    python run.py --skew
    python run.py --validation
    python run.py --gate
    python run.py --bayes
    python run.py --report

The default workspace is nasdaq_daily_30days. Pass --workspace <name> for another run.

See docs/METHOD.md for what each stage does and why.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data import features, fetcher
from engine import barrier, bayes, skew as skw, validate as val, writer
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


def _write_surface_array(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """Stage 1a. The faithful record: every barrier level, every horizon."""
    tree  = load_tree(ws.tree_path)
    nodes = all_in_family(tree, family) if family else all_nodes(tree)
    if not rerun:
        nodes = [n for n in nodes if not ws.has_cube(n['family'], n['id'])]
    if not nodes:
        print('No missing surface arrays (use --rerun to rebuild existing arrays).')
        return

    get_data = _loader(ws)
    hz = ws.horizons
    print(f"\n=== 1. 01_surface_array [{ws.dir.name}] - {len(nodes)} nodes ===")
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
    print(f"\n  wrote to {ws.dir.relative_to(ROOT)}/01_surface_array/"
          + (f"   ({len(skipped)} skipped)" if skipped else ""))


def _write_surface_xlsx(ws: Workspace, family: str | None = None) -> None:
    """Stage 1. The full cube, made readable; never re-measures."""
    _render(ws, family, summary=False)


def cmd_surface(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """Stage 1. Write the full surface array and workbook."""
    _write_surface_array(ws, family, rerun)
    _write_surface_xlsx(ws, family)


def _write_summary_array(ws: Workspace, family: str | None = None) -> None:
    """
    Stage 2. The coarse grid everything is judged on, selected from the full cube.

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
        print('No full surface arrays to summarize - run --surface first.')
        return

    # the baseline is the reference the filter reads, so its summary must exist first
    nodes = ([n for n in nodes if n['id'] == BASELINE_NODE]
             + [n for n in nodes if n['id'] != BASELINE_NODE])

    th, hz = ws.summary_thetas, ws.summary_horizons
    print(f"\n=== 2. 02_summary_array [{ws.dir.name}] - {len(nodes)} nodes ===")
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


def _write_summary_xlsx(ws: Workspace, family: str | None = None) -> None:
    """Stage 2. The summary cube, made readable. Same renderer, smaller grid."""
    _render(ws, family, summary=True)


def cmd_summary(ws: Workspace, family: str | None = None) -> None:
    """Stage 2. Write the summary array and workbook."""
    _write_summary_array(ws, family)
    _write_summary_xlsx(ws, family)


def _render(ws: Workspace, family: str | None, summary: bool) -> None:
    tree = load_tree(ws.tree_path)
    have = ws.has_summary_cube if summary else ws.has_cube
    src  = ws.summary_cube_path if summary else ws.cube_path
    dst  = ws.summary_surface_path if summary else ws.surface_path
    label = 'summary' if summary else 'full'
    stage = '2. 02_summary_xlsx' if summary else '1. 01_surface_xlsx'
    out_dir = '02_summary_xlsx' if summary else '01_surface_xlsx'

    nodes = [n for n in (all_in_family(tree, family) if family else all_nodes(tree))
             if have(n['family'], n['id'])]
    if not nodes:
        print(f'No {label} arrays to render - run --{"summary" if summary else "surface"} first.')
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
          f"{out_dir}/")


def cmd_skew(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """
    Stage 3, per node: the up/down asymmetry the mirrored summary grid is built for.

    For every +/- theta pair, horizon and condition bin, two numbers in percentage points:

      conditional skew   P(+theta | bin) - P(-theta | bin)          -- carries the drift
      excess skew        that, minus the baseline node's own skew    -- drift removed

    Pure arithmetic on the summary cube's `prob` array, so a skew cell and the two
    probability cells it came from can never disagree. The baseline node gets an artifact
    like everything else; its excess skew is ~0 everywhere by construction.
    """
    tree  = load_tree(ws.tree_path)
    nodes = [n for n in (all_in_family(tree, family) if family else all_nodes(tree))
             if ws.has_summary_cube(n['family'], n['id'])]
    if not rerun:
        nodes = [n for n in nodes if not (
            ws.has_skew(n['family'], n['id'])
            and ws.has_skew_sheet(n['family'], n['id'])
        )]
    if not nodes:
        print('Nothing to skew (use --rerun to redo, or --summary first).')
        return

    baseline = _baseline_surface(ws)
    if baseline is None:
        print('  no baseline summary array - run --summary first')
        return

    th, hz = ws.summary_thetas, ws.summary_horizons
    n_mag = int((len(th) - 1) // 2)
    print(f"\n=== 3. 03_skew_array [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(f"  {n_mag} θ magnitudes x {ws.n_bins} bins x {len(hz)} horizons per node")
    print(f"  conditional  P(+θ|bin) − P(−θ|bin)     excess  that − baseline skew\n")

    done_npz = 0
    done_xlsx = 0
    for node in nodes:
        try:
            cube = barrier.load_cube(ws.summary_cube_path(node['family'], node['id']))
            r = skw.skew_from_cube(cube, baseline)
            skw.save(r, ws.skew_path(node['family'], node['id']), {
                'node': node['id'], 'family': node['family'],
                'feature': node['feature'], 'params': node['params'],
                'workspace': ws.dir.name,
                'bin_labels': cube['meta']['bin_labels'],
                'generated': datetime.now(timezone.utc).isoformat(),
            })
            done_npz += 1
            writer.write_skew_xlsx(
                r, ws.skew_sheet_path(node['family'], node['id']),
                node['id'], node['feature'], node['params'], ws.horizon_unit)
            done_xlsx += 1
        except PermissionError:
            print(f"  {node['id']:<26} [locked] close it in Excel and re-run")
            continue
        except Exception as e:
            print(f"  {node['id']:<26} [skip] {e}")
            continue

        peaks = skw.peak_by_horizon(r)
        hit = {h: p for h, p in peaks.items() if p}
        if hit:
            h = max(hit, key=lambda k: abs(hit[k]['excess']))
            p = hit[h]
            print(f"  {node['id']:<26} peak excess {p['excess']:+5.1f}pp at +{h}"
                  f"{ws.horizon_unit}  |θ|={p['mag'] * 100:.3g}%  bin {p['bin'] + 1}  "
                  f"(cond {p['cond']:+.1f}pp)")
        else:
            print(f"  {node['id']:<26} no bin above the reporting threshold")

    print(f"\n  wrote {done_npz} .npz artifacts under {ws.dir.relative_to(ROOT)}/03_skew_array/")
    print(f"  wrote {done_xlsx} workbooks under {ws.dir.relative_to(ROOT)}/03_skew_xlsx/")


def cmd_validation(ws: Workspace, family: str | None = None, rerun: bool = False) -> None:
    """
    Stage 4, per node: null-test the *summary* cube and write validation artifacts.

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
    missing_skew = [n['id'] for n in nodes if not ws.has_skew(n['family'], n['id'])]
    nodes = [n for n in nodes if ws.has_skew(n['family'], n['id'])]
    if missing_skew:
        print(f"  {len(missing_skew)} node(s) have no skew artifact - run --skew first "
              f"(skipping {', '.join(missing_skew[:5])}{' ...' if len(missing_skew) > 5 else ''})")
    if not rerun:
        nodes = [n for n in nodes if (
            not ws.has_validation(n['family'], n['id'])
            or not ws.has_validation_sheet(n['family'], n['id'])
        )]
    if not nodes:
        print('Nothing to validate or render (use --rerun to redo, or --skew first).')
        return

    baseline_skew = skw.baseline_cond_skew(barrier.load_cube(ws.baseline_cube))

    get_data = _loader(ws)
    th, hz = ws.summary_thetas, ws.summary_horizons
    print(f"\n=== 4. 04_validation_array [{ws.dir.name}] - {len(nodes)} nodes ===")
    print(f"  on the summary grid: {len(th)} θ x {ws.n_bins} bins x {len(hz)} horizons "
          f"= {len(th) * ws.n_bins * len(hz)} cells per node")
    print(f"  a peak over that has a far lower noise ceiling than one over the full "
          f"{len(ws.thetas) * ws.n_bins * len(ws.horizons)}")
    print(f"  guard {val.EDGE_GUARD} bars either side, so the p-value floor is 1/(usable+1)")
    print(f"  nulls the touch peak and the excess-skew peak in the same pass\n")

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
                r = val.validate_node(data, feat, cube['horizons'], cube['thetas'],
                                      cube['edges'], baseline_skew=baseline_skew)
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

        sk_finite = np.isfinite(r['skew_peak_p']) if 'skew_peak_p' in r else np.array([False])
        if sk_finite.any():
            k = int(np.nanargmin(np.where(sk_finite, r['skew_peak_p'], np.nan)))
            print(f"  {'':<26}   skew p={r['skew_peak_p'][k]:.5f} at +{r['horizons'][k]}"
                  f"{ws.horizon_unit}  peak {r['skew_peak_real'][k]:5.1f} vs p95 "
                  f"{r['skew_peak_p95'][k]:5.1f}")

    print(f"\n  wrote {done_npz} .npz artifacts under {ws.dir.relative_to(ROOT)}/04_validation_array/")
    print(f"  wrote {done_xlsx} workbooks under {ws.dir.relative_to(ROOT)}/04_validation_xlsx/")
    print(f"  verdicts are assigned by --gate, which needs the whole sweep to correct across")


def cmd_gate(ws: Workspace, q: float = 0.05) -> None:
    """
    Stage 5's conclusion: correct across the sweep, then intersect with the
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
        print('No evaluation.json - run --summary first.')
        return

    tree = load_tree(ws.tree_path)
    rows = []
    skew_rows = []
    for node in all_nodes(tree):
        path = ws.validation_path(node['family'], node['id'])
        if not path.exists():
            continue
        r = val.load(path)
        has_skew = 'skew_peak_p' in r
        for j, h in enumerate(r['horizons']):
            rows.append({'node': node['id'], 'family': node['family'],
                         'horizon': int(h), 'peak_p': float(r['peak_p'][j]),
                         'peak_real': float(r['peak_real'][j]),
                         'peak_p95': float(r['peak_p95'][j]),
                         'n_shifts': int(r['n_shifts'][j])})
            if has_skew and np.isfinite(r['skew_peak_p'][j]):
                skew_rows.append({'node': node['id'], 'family': node['family'],
                                  'horizon': int(h),
                                  'peak_p': float(r['skew_peak_p'][j]),
                                  'peak_real': float(r['skew_peak_real'][j]),
                                  'peak_p95': float(r['skew_peak_p95'][j]),
                                  'n_shifts': int(r['n_shifts'][j])})
    if not rows:
        print('No validation artifacts - run --validation first.')
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

    # ── skew: a separate BH family, reported not gated ────────────────────────
    # Skew asks a different question from the touch peak -- which way a condition bends
    # the up/down asymmetry, not whether it moves a barrier at all -- so pooling the two
    # p-value families would mis-correct both. It is corrected and reported here; it does
    # not change what clears above.
    skew_summary = {'discovery': 0, 'nominal': 0, 'n_tests': 0, 'discoveries': []}
    if skew_rows:
        sk_p = np.array([r['peak_p'] for r in skew_rows])
        sk_rej, sk_qv = val.bh(sk_p, q)
        for r, rej, qv in zip(skew_rows, sk_rej, sk_qv):
            floor = 1.0 / (1.0 + r['n_shifts']) if r['n_shifts'] else np.nan
            r['q_value'] = None if not np.isfinite(qv) else round(float(qv), 6)
            r['at_floor'] = bool(np.isfinite(r['peak_p']) and np.isfinite(floor)
                                 and r['peak_p'] <= floor + 1e-12)
            r['verdict'] = val.verdict(bool(rej), r['peak_p'], r['at_floor'])
        sk_m = int(np.isfinite(sk_p).sum())
        sk_disc = sorted({r['node'] for r in skew_rows if r['verdict'] == 'discovery'})
        skew_summary = {
            'discovery': sum(1 for r in skew_rows if r['verdict'] == 'discovery'),
            'nominal':   sum(1 for r in skew_rows if r['verdict'] == 'nominal'),
            'n_tests':   sk_m,
            'discoveries': sk_disc,
        }
        print(f"\n  ── skew (separate BH family, reported not gated) ──")
        print(f"  BH skew discoveries : {skew_summary['discovery']:>4} / {sk_m}"
              f"   over {len(sk_disc)} node(s)")
        print(f"  nominal (p<=0.05)   : {skew_summary['nominal']:>4} / {sk_m}")
        if sk_disc:
            print(f"  {', '.join(sk_disc)}")
    else:
        print(f"\n  ── skew ── no skew p-values in the validation artifacts "
              f"(re-run --skew then --validation --rerun)")

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
        'skew_summary': skew_summary,
        'skew_tests': skew_rows,
    })
    print(f"\n  wrote {ws.cleared_path.relative_to(ROOT)}")


def _feature_panel(ws: Workspace) -> tuple[pd.DataFrame, dict, dict]:
    """
    Every non-baseline node's feature on one shared index, with its family.

    The baseline is excluded here alone in the pipeline: its feature is constant, so it
    contributes a log-odds delta of exactly zero to every bar and only widens the
    intersection of valid rows for nothing. Its *prior* still enters the model -- as the
    prior, estimated per fold on the training window.
    """
    tree = load_tree(ws.tree_path)
    get_data = _loader(ws)
    feats, fams, data = {}, {}, None
    for node in all_nodes(tree):
        if node['id'] == BASELINE_NODE:
            continue
        try:
            d, f = _node_feature(ws, node, get_data)
        except Exception:
            continue
        if data is None or len(d) > len(data):
            data = d
        feats[node['id']] = f
        fams[node['id']] = node['family']
    if data is None or not feats:
        raise ValueError('no computable features in this workspace')
    return data, {k: v.reindex(data.index) for k, v in feats.items()}, fams


def cmd_bayes(ws: Workspace) -> None:
    """
    Stage 6: do the conditions compose, and does that survive out of sample?

    Everything before this measures one condition at a time and over the whole history.
    This asks the only question that follows -- what happens when several hold at once --
    and answers it under the one discipline the rest of the pipeline does not need: the
    tables, the bin edges, the prior and the scale correction are all estimated on a
    training window and applied to a later one the fit never saw.

    Three models on the same walk forward, so the two failures are measured rather than
    asserted: every node (collinear, so evidence is counted many times over), one node
    per family (the collinearity mostly removed), and that rescaled (the ordering was
    real, the confidence was not).
    """
    baseline = _baseline_surface(ws)
    if baseline is None:
        print('No baseline summary array - run --summary first.')
        return

    theta, horizon = ws.bayes_target(baseline)
    data, feats, fams = _feature_panel(ws)

    print(f"\n=== 8. Composition [{ws.dir.name}] - {len(feats)} nodes, "
          f"{len(set(fams.values()))} families ===")
    print(f"  target: P(touch {theta:+.0%} within {horizon}{ws.horizon_unit})   "
          f"expanding walk forward, {ws.bayes_folds} folds over the last half")
    print(f"  {horizon}-bar embargo between train and test; scored on every "
          f"{horizon}th test bar so the windows do not overlap\n")

    head = bayes.walk_forward(data, feats, fams, theta, horizon,
                              n_bins=ws.n_bins, folds=ws.bayes_folds)
    m = head['metrics']
    print(f"  {'model':<24}{'Brier':>9}{'AUC':>8}{'mean P':>9}   vs prior")
    print('  ' + '-' * 60)
    ref = m['prior_only']['brier']
    for key, name in (('prior_only', 'prior only'),
                      ('all_nodes', 'naive Bayes, all nodes'),
                      ('one_per_family', 'one node per family'),
                      ('one_per_family_scaled', '   + scale corrected')):
        e = m[key]
        mark = '' if key == 'prior_only' else f"{(e['brier'] / ref - 1) * 100:+.1f}%"
        auc_s = '     -  ' if not np.isfinite(e['auc']) else f"{e['auc']:>8.3f}"
        print(f"  {name:<24}{e['brier']:>9.4f}{auc_s}{e['mean_predicted']:>9.1%}   {mark}")
    a_mean = float(np.mean([f['platt_a'] for f in head['folds']]))
    print(f"\n  realized {m['realized_rate']:.1%} over {m['n_scored']} non-overlapping "
          f"out-of-sample bars")
    print(f"  scale correction a = {a_mean:.3f}: the raw score is overconfident by "
          f"{1 / a_mean:.1f}x")

    # the same walk forward across the judged grid, so the composition result is
    # reported on the axes everything else in the pipeline uses
    th, hz = ws.summary_thetas, ws.summary_horizons
    targets = [(float(t), int(h)) for t in th if abs(t) > 1e-12 for h in hz]
    print(f"\n  sweeping the summary grid: {len(targets)} targets", end='', flush=True)
    grid = []
    for t, h in targets:
        try:
            r = bayes.walk_forward(data, feats, fams, t, h,
                                   n_bins=ws.n_bins, folds=ws.bayes_folds)
        except Exception:
            continue
        g = r['metrics']
        grid.append({'theta': t, 'horizon': h,
                     'auc': g['one_per_family_scaled']['auc'],
                     'brier_prior': g['prior_only']['brier'],
                     'brier_all': g['all_nodes']['brier'],
                     'brier_dedup': g['one_per_family']['brier'],
                     'brier_scaled': g['one_per_family_scaled']['brier'],
                     'platt_a': float(np.mean([f['platt_a'] for f in r['folds']])),
                     'realized': g['realized_rate'], 'n_scored': g['n_scored']})
    beat = sum(1 for g in grid if g['brier_scaled'] < g['brier_prior'])
    print(f" - {beat} of {len(grid)} beat the prior after correction")

    np.savez_compressed(
        ws.bayes_path,
        y=head['y'], p_all=head['p_all'], p_dedup=head['p_dedup'],
        p_scaled=head['p_scaled'], p_prior=head['p_prior'], fold=head['fold'],
        grid_theta=np.array([g['theta'] for g in grid], dtype=float),
        grid_horizon=np.array([g['horizon'] for g in grid], dtype=int),
        grid_auc=np.array([g['auc'] for g in grid], dtype=float),
        grid_brier_prior=np.array([g['brier_prior'] for g in grid], dtype=float),
        grid_brier_all=np.array([g['brier_all'] for g in grid], dtype=float),
        grid_brier_dedup=np.array([g['brier_dedup'] for g in grid], dtype=float),
        grid_brier_scaled=np.array([g['brier_scaled'] for g in grid], dtype=float),
        grid_platt_a=np.array([g['platt_a'] for g in grid], dtype=float),
        grid_realized=np.array([g['realized'] for g in grid], dtype=float),
        grid_n_scored=np.array([g['n_scored'] for g in grid], dtype=int),
        meta=np.array(json.dumps({
            'workspace': ws.dir.name, 'theta': theta, 'horizon': horizon,
            'unit': ws.horizon_unit,
            'generated': datetime.now(timezone.utc).isoformat(),
        })),
    )
    ws.write_json(ws.bayes_summary_path, {
        'workspace': ws.dir.name,
        'generated': datetime.now(timezone.utc).isoformat(),
        'target': {'theta': theta, 'horizon': horizon, 'unit': ws.horizon_unit},
        'design': {'folds': ws.bayes_folds, 'embargo_bars': horizon,
                   'scored_every': horizon, 'n_features': head['n_features'],
                   'n_families': len(set(fams.values())),
                   'shrinkage_k': bayes.SHRINK_K,
                   'note': 'bin edges, per-bin rates, the prior and the scale '
                           'correction are all fit on the training window only'},
        'metrics': m,
        'platt_a_mean': a_mean,
        'folds': head['folds'],
        'kept_per_fold': head['kept'],
        'grid': grid,
        'grid_beat_prior': beat,
    })
    print(f"\n  wrote {ws.bayes_path.relative_to(ROOT)} and "
          f"{ws.bayes_summary_path.relative_to(ROOT)}")


def cmd_report(ws: Workspace) -> None:
    """Stage 7: render the audience-facing figures from the artifacts on disk."""
    from engine import report
    report.build(ws)


# ── inventory ─────────────────────────────────────────────────────────────────

def cmd_status(ws: Workspace) -> None:
    tree = load_tree(ws.tree_path)
    ev   = ws.read_json(ws.eval_path).get('nodes', {})
    cl   = ws.read_json(ws.cleared_path)
    cleared = {c['node'] for c in cl.get('cleared', [])}
    disc = {t['node'] for t in cl.get('tests', []) if t.get('verdict') == 'discovery'}
    sk_disc = set(cl.get('skew_summary', {}).get('discoveries', []))

    cols = ['nodes', 'cube', 'sheet', 'sum', 'sheet', 'skew',
            'valid', 'v-sheet', 'econ', 'sk-disc', 'cleared']
    by_fam = defaultdict(lambda: [0] * len(cols))
    for n in all_nodes(tree):
        c = by_fam[n['family']]
        c[0] += 1
        c[1] += bool(ws.has_cube(n['family'], n['id']))
        c[2] += bool(ws.has_surface(n['family'], n['id']))
        c[3] += bool(ws.has_summary_cube(n['family'], n['id']))
        c[4] += bool(ws.has_summary_surface(n['family'], n['id']))
        c[5] += bool(ws.has_skew_sheet(n['family'], n['id']))
        c[6] += bool(ws.has_validation(n['family'], n['id']))
        c[7] += bool(ws.has_validation_sheet(n['family'], n['id']))
        c[8] += bool(ev.get(n['id'], {}).get('passed'))
        c[9] += n['id'] in sk_disc
        c[10] += n['id'] in cleared

    print(f"\n=== Status [{ws.dir.name}] ===")
    print(f"  θ {ws.thetas[0]:+.0%}..{ws.thetas[-1]:+.0%}   "
          f"horizons +{ws.horizons[0]}{ws.horizon_unit}..+{ws.horizons[-1]}{ws.horizon_unit}   "
          f"{ws.n_bins} bins")
    if cl:
        meth = cl.get('method', {})
        print(f"  gate: BH q={meth.get('q')} over {meth.get('n_tests')} tests, "
              f"{len(disc)} nodes with a discovery, {len(sk_disc)} with a skew discovery")
    print()
    hdr = f"  {'family':<15}" + ''.join(f'{name:>9}' for name in cols)
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for fam in sorted(by_fam):
        print(f"  {fam:<15}" + ''.join(f'{v:>9}' for v in by_fam[fam]))
    tot = [sum(by_fam[f][i] for f in by_fam) for i in range(len(cols))]
    print('  ' + '-' * (len(hdr) - 2))
    print(f"  {'TOTAL':<15}" + ''.join(f'{v:>9}' for v in tot))
    print(f"\n  full grid {len(ws.thetas)}θ x {len(ws.horizons)}h   |   "
          f"summary {len(ws.summary_thetas)}θ x {len(ws.summary_horizons)}h "
          f"(judged and validated here)")


def cmd_read(ws: Workspace, node_id: str) -> None:
    """
    Print one node's summary cube in the terminal: the θ rows carrying a qualifying
    cell, for whichever bin is strongest, with the null test beside them.

    Reads the stored artifacts rather than recomputing, so what prints is exactly what
    the workbook and the validation hold.
    """
    tree = load_tree(ws.tree_path)
    node = find_node(tree, node_id)
    path = ws.summary_cube_path(node['family'], node_id)
    if not path.exists():
        print(f"No summary array for {node_id} - run --summary first.")
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
        print('  no baseline summary array - run --summary first')
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
        if 'skew_peak_p' in vr and np.isfinite(vr['skew_peak_p']).any():
            k = int(np.nanargmin(np.where(np.isfinite(vr['skew_peak_p']),
                                          vr['skew_peak_p'], np.nan)))
            print(f"  skew null     : peak excess {vr['skew_peak_real'][k]:+.1f}pp vs p95 "
                  f"{vr['skew_peak_p95'][k]:.1f}, p={vr['skew_peak_p'][k]:.5f} "
                  f"at +{int(vr['horizons'][k])}{ws.horizon_unit}")
    else:
        print('  null          : not validated yet')

    spath = ws.skew_path(node['family'], node_id)
    if spath.exists():
        sr = skw.load(spath)
        peaks = skw.peak_by_horizon(sr)
        hit = {h: p for h, p in peaks.items() if p}
        if hit:
            h = max(hit, key=lambda k: abs(hit[k]['excess']))
            p = hit[h]
            print(f"  skew          : excess {p['excess']:+.1f}pp (cond {p['cond']:+.1f}pp) "
                  f"at |θ|={p['mag'] * 100:.3g}%, bin {p['bin'] + 1}, +{h}{ws.horizon_unit}")

    avail = {int(x) for x in hz}
    show  = [h for h in (1, 2, 3, 5, 7, 10, 14, 21, 30) if h in avail]
    cols  = [int(np.flatnonzero(hz == h)[0]) for h in show]
    dev   = (prob - baseline[:, None, :]) * 100.0
    pct_dec = 1 if ws.theta_step < 0.01 else 0
    theta_fmt = f'+.{pct_dec}%'

    print()
    header = ''.join(f"{'+' + str(h) + ws.horizon_unit:>8}" for h in show)
    print(f"  {'θ':>7}{header}")
    print('  ' + '-' * (7 + 8 * len(show)))
    shown = 0
    for i in sorted(range(len(thetas)), key=lambda k: -thetas[k]):
        d = dev[i, b, cols]
        finite = d[np.isfinite(d)]
        if finite.size == 0 or np.max(np.abs(finite)) < ws.min_dev:
            continue
        cells = ''.join('       -' if not np.isfinite(v) else f'{v:>8.1%}'
                        for v in prob[i, b, cols])
        print(f"  {format(thetas[i], theta_fmt):>7}{cells}")
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
    parser.add_argument('--workspace', metavar='NAME', default='nasdaq_daily_30days')
    parser.add_argument('--family', metavar='NAME', help='Restrict to one family')
    parser.add_argument('--rerun', action='store_true', help='Rebuild artifacts that already exist')
    parser.add_argument('--fdr', type=float, default=0.05, metavar='Q',
                        help='Benjamini-Hochberg false-discovery rate for --gate (default 0.05)')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--surface', action='store_true',
                   help='1. write 01_surface_array and 01_surface_xlsx')
    g.add_argument('--summary', action='store_true',
                   help='2. write 02_summary_array and 02_summary_xlsx; runs the economic filter')
    g.add_argument('--skew',     action='store_true',
                   help='3. write 03_skew_array and 03_skew_xlsx')
    g.add_argument('--validation', action='store_true',
                   help='4. write 04_validation_array and 04_validation_xlsx')
    g.add_argument('--gate',     action='store_true', help='5. correct across the sweep (BH) and intersect with the economic filter')
    g.add_argument('--bayes',    action='store_true',
                   help='6. compose the conditions and test the result out of sample, 06_bayez.npz + 06_bayes.json')
    g.add_argument('--report',   action='store_true', help='7. render the audience-facing figures into workspace result/')
    g.add_argument('--status',   action='store_true', help='Inventory by family')
    g.add_argument('--read',     metavar='ID', help='Print a node surface summary')

    args = parser.parse_args()
    ws = Workspace(args.workspace)

    if args.surface:
        cmd_surface(ws, args.family, args.rerun)
    elif args.summary:
        cmd_summary(ws, args.family)
    elif args.skew:
        cmd_skew(ws, args.family, args.rerun)
    elif args.validation:
        cmd_validation(ws, args.family, args.rerun)
    elif args.gate:
        cmd_gate(ws, args.fdr)
    elif args.bayes:
        cmd_bayes(ws)
    elif args.report:
        cmd_report(ws)
    elif args.status:
        cmd_status(ws)
    elif args.read:
        cmd_read(ws, args.read)
    else:
        parser.print_help()
