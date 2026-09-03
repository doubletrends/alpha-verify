# Pipeline — Procedure

Six stages, with machine and readable artifacts per node. The full grid is measured and rendered but
never judged; a coarse subset of it is what gets judged and validated.

```
--cubes             1. measure   cubes_full/         41 θ × 10 bins × 30 h
--surfaces          2. render    surfaces_full/
--cubes-summary     3. reduce    cubes_summary/      17 θ × 10 bins × 7 h  (+ economic filter)
--surfaces-summary  4. render    surfaces_summary/
--validate          5. falsify   validations/        exact shuffled null
                              validations_xlsx/   readable validation sheets
--gate              6. correct   cleared.json        Benjamini-Hochberg + economic
```

Pass `--workspace <name>` to every command.

---

## The question this answers

> Given that condition X holds, how likely is price to *reach* level θ within h bars?

That is the question a stop, a limit, a liquidation and a margin call all turn on, and
it is not the same as asking where price closes. The pipeline answers it for every
barrier level, every condition and every horizon at once, then filters the result twice
before letting you believe any of it.

---

## 0. Orient

```
python run.py --workspace <name> --status
```

The columns are the funnel: how many nodes exist, how many have a cube, how many clear
the economic filter, how many clear the null, how many clear both.

---

## 1. Measure — the full cube

```
python run.py --workspace <name> --cubes
python run.py --workspace <name> --cubes --family volatility   # or one family
```

Writes `cubes_full/<family>/<node>.npz` — a 3-D array per node:

| axis | contents |
|---|---|
| 0 | barrier level θ, −20% … +20%, ascending, **including 0** |
| 1 | condition bin — the feature's deciles |
| 2 | horizon h, +1 … +30 bars — **every** horizon, not a chosen few |

The value is the raw conditional probability:

```
P(price touches θ within h | X in bin)
```

Nothing is subtracted. The unconditional rate is **its own node**: `baseline`, whose
feature is constant, so every bar falls in a single bin and its one column is
P(touch θ in h) with nothing conditioned on. It is built first — it is the reference
every other node is judged against — and it flows through `--cubes` and `--surfaces`
with no special case in the engine.

`--validate` runs on it like everything else and produces a degenerate artifact —
shuffling a constant cannot change the surface, so its peak deviation is exactly 0 and
its p-value exactly 1. It is never a discovery, and keeping it in costs one file and
removes a branch from the pipeline.

Caveat worth knowing: the baseline covers every bar, while a long-warmup feature covers
fewer, and the bars it loses are the oldest. For a 200-bar moving average that is ~5% of
BTC's history and the most volatile part of it, so the comparison is very slightly
against a different past.

The file also carries `hits`, `bin_n`, `n_obs`, the three coordinate vectors and the bin
edges, so anything derivable from the measurement can be derived without recomputing it.
About 90 KB per node; loads in milliseconds.

Also writes `evaluation.json` — the economic filter's verdict per node, scanned across
the whole cube. `--rerun` rebuilds cubes that already exist.

**Why this is its own step.** The full horizon ladder is where the shape of a signal
lives: whether the edge decays or persists, whether it is a band of barrier levels or a
single spike. Four sheets cannot show that. Measuring once into a cube also makes the
workbook a *projection* rather than a second computation, so the two can never disagree.

---

## 2. Render the full cube

```
python run.py --workspace <name> --surfaces
```

Reads the cubes and writes `surfaces_full/<family>/<node>.xlsx` — **one tab per
condition bin**, each holding that bin's entire θ × h face:

| | |
|---|---|
| rows | barrier level θ, **+20% at the top → −20% at the bottom** |
| columns | horizon h, +1 … +30 |
| cells | `P(touch θ in h │ condition)` — the raw probability |
| n band | observations behind that bin at each horizon |

Ten tabs × 41 θ × 30 horizons is every value the cube holds, so the workbook is a
faithful view of the measurement rather than a summary of it.

Rows run the way a price ladder reads: up the sheet is up in price, with θ = 0 shaded in
the middle marking the boundary between the two questions — above it a cell asks whether
the *high* reached that level, below it whether the *low* did.

Read **across a row** to see the probability grow with horizon; read **down a column**
to see the band of levels price is likely to reach. Compare the +θ and −θ rows at the
same horizon and the condition's asymmetry is immediate, with no arithmetic.

The colour scale is a fixed 0–100% on every tab, so flipping between tabs shows the band
shifting with the condition instead of each tab being rescaled to look alike.

This step renders only; it never re-measures.

**Reading a sheet.** Negative θ asks whether the *low* reached it, positive θ whether
the *high* did — both on intraday extremes, not closes, because a stop fills on the low.
Two dead zones are expected and are not signal: θ near zero is touched almost always,
and large |θ| at short horizons is touched almost never.

---

## 3. Reduce — the summary grid

```
python run.py --workspace <name> --cubes-summary
python run.py --workspace <name> --surfaces-summary
```

Selects a coarse sub-grid out of each full cube and writes `cubes_summary/` and
`surfaces_summary/`. Pure index selection: every summary cell is bit-identical to the
full cell it came from, and a requested θ or h that is not on the full grid raises
rather than interpolating — interpolating would make the summary a second measurement,
and then the two could disagree.

**Why judge a subset.** Adjacent cells of the full grid are very nearly the same
measurement. A peak over 12,300 cells has a large noise ceiling by construction, and
counting them as 12,300 tests overstates the search in both directions at once. On the
summary the peak is over 1,190 cells and the test count is honest.

The **economic filter runs here**, not on the full cube, so that what is judged and what
is validated sit on the same grid. Writes `evaluation.json`.

Configure with `summary.theta_abs` (magnitudes, mirrored) and `summary.horizons`.

---

## 4. Falsify

```
python run.py --workspace <name> --validate
python run.py --workspace <name> --validate --family volatility   # or one family
```

Runs on the **summary** cube, using its stored θ, horizons and bin edges, so the test
and the thing it judges are the same grid down to the last cell.

Writes `validations/<family>/<node>.npz`, one per node — the baseline included, whose
artifact comes out degenerate because shuffling a constant cannot change anything. It
also writes `validations_xlsx/<family>/<node>.xlsx`, a readable workbook with a summary
tab plus per-bin signed-deviation and pointwise-p tabs. Per-node artifacts also mean two
runs never write the same path, which is what made a shared validation file
untrustworthy.

| | |
|---|---|
| `cell_real` | (θ, bin, h) signed deviation from the node's own sample rate |
| `cell_p` | (θ, bin, h) **pointwise** p-value — for reading a surface, never a discovery |
| `cell_p95` | (θ, bin, h) 95th percentile of each cell's own null |
| `peak_real`, `peak_p`, `peak_p95` | (h,) the max over the surface — the statistic that accounts for the search |
| `n_shifts` | (h,) usable shifts, so the p-value floor is `1/(n+1)` |

**The null is exact.** `counts[θ, bin]` as a function of shift is a circular
cross-correlation, so one FFT gives every shift at once — ~70× faster than resampling
and it returns the whole permutation distribution. On the summary grid the whole
workspace validates in well under a minute.

Counts are rounded to integers after the transform. They are counts of 0/1 indicators
and so exact by construction; the FFT returns them with ~1e-13 of roundoff. That matters
most in the degenerate case — a single-bin cube has a deviation of identically zero, and
without rounding the null compares one speck of numerical noise against another and
returns a meaningless p-value instead of 1.

**Its floor is real.** Only *n* distinct shifts exist, so no p-value below `1/(n+1)`
(~2.6e-4) is obtainable. Resampling 20,000 shifts from a group of 4,214 and reporting
`p = 1/20001` overstates significance about 5×.

**`cell_p` is pointwise.** With 12,300 cells, ~615 fall below 0.05 by chance. Use it to
read where a surface is unusual; never as a criterion for finding something.

---

## 5. The gate

```
python run.py --workspace <name> --gate
python run.py --workspace <name> --gate --require nominal   # looser
```

```
python run.py --workspace <name> --gate --fdr 0.05
```

Corrects across the whole sweep, then intersects with the economic filter, and writes
`cleared.json`.

- **Economic** (from `--cubes`): a cell deviates from the baseline node by at least
  `min_dev` pp, in a bin holding at least `min_bin_n` observations, across at least
  `min_run` *adjacent* θ rows with the same sign.
- **Statistical** (from `--validate`): the node's peak p-value survives
  Benjamini–Hochberg at `q`.

**Why BH and not Bonferroni.** Bonferroni needs a p-value below `q/m`. With m in the
hundreds that threshold falls *under* the `1/(n+1)` floor the exact null can physically
reach, so nothing could ever clear it however strong the signal. BH compares the k-th
smallest p-value against `k·q/m`, so tests at the floor clear collectively, and it
controls the share of false positives among discoveries instead of the probability of
any at all.

Verdicts live here, not in the artifacts: a correction is a property of the collection,
and a stored verdict would go stale the moment another node joined the sweep.

Both are required. A large steady edge that fails its null is a shape found by
searching. A significant edge concentrated in a single cell is real and useless — an
edge at exactly −7% and nowhere else is not a stop level, it is an artifact. The
adjacency requirement is what encodes "usable" rather than merely "large".

This is the end of the pipeline. What clears the gate is a measured, falsified statement
about where price reaches under a stated condition — not a strategy.

---

## What to distrust

**Overlapping windows.** Every count in a sheet is a count of *bars*, and consecutive
bars in the same bin share almost the same forward window. A decile condition on ten
years of daily bars is ~425 bars but only ~100–200 independent windows. Read `n` as an
upper bound on the evidence, not a measure of it, and more so at long horizons.

**The best cell of any search.** The null exists because the maximum of 400 noisy
estimates is large by construction. A number without its verdict beside it is not a
result.

**Short horizons.** At h = 1, "will price touch −2%" is barely distinguishable from "is
volatility high right now", so volatility features answer it near-tautologically. Most
of what clears the gate peaks at h = 1. Real, significant, and mostly mechanical — look
at where a node's edge sits before believing it means anything.

**Bar resolution.** Daily OHLC records a session's high and low but not their order, so
any question that depends on which came first is unanswerable at this bar size. That is
a data problem, not a code one.

---

## Adding a feature

Cross-asset features go in `data/features.py`:

```python
def _my_feature(close: pd.Series, period: int) -> pd.Series: ...

register('my_feature', lambda d, p: _my_feature(d['close'], p['period']))
```

Workspace-specific features and data sources go in `workspaces/<name>/plugin.py`, which
is imported automatically. Then append the node to `universe.json`:

```json
{
  "id": "rsi_spread_7_14",
  "family": "rsi",
  "category": "price_momentum",
  "feature": "rsi_spread",
  "params": {"fast": 7, "slow": 14},
  "data": ["ohlcv"],
  "derived_from": "rsi_7"
}
```

There is no `status` field — progress is read from disk.

---

## Adding a workspace

`universe.json` `meta` carries everything asset-specific:

| field | meaning |
|---|---|
| `asset` | `{provider, ticker, interval}` |
| `start_date` | history start |
| `horizons` | `{min, max}` — the cube's horizon ladder |

| `theta` | `{min, max, step}` — the full barrier ladder |
| `summary` | `{theta_abs, horizons}` — the coarse grid that gets judged |
| `n_bins` | condition bins per feature (quantile) |
| `evaluate` | `{min_dev, min_bin_n, min_run}` — the economic filter |

**Scale `theta` to the asset and horizon.** BTC daily runs ±20% in 1% steps; NASDAQ
hourly runs ±4% in 0.2% steps, because a 20% move inside a trading day does not happen
and every row of that ladder would be empty.

---

## Files

| Path | Written by | Contents |
|------|-----------|----------|
| `universe.json` | you | nodes + asset config; never written by the pipeline |
| `cubes_full/<family>/<node>.npz` | `--cubes` | the faithful record — (θ × bin × h) probabilities |
| `surfaces_full/<family>/<node>.xlsx` | `--surfaces` | 10 tabs, the full cube made readable |
| `cubes_summary/<family>/<node>.npz` | `--cubes-summary` | the judged grid, a strict subset |
| `surfaces_summary/<family>/<node>.xlsx` | `--surfaces-summary` | 10 tabs, the summary made readable |
| `evaluation.json` | `--cubes-summary` | economic filter verdict per node |
| `validations/<family>/<node>.npz` | `--validate` | per-cell and peak nulls, one per node |
| `validations_xlsx/<family>/<node>.xlsx` | `--validate` | readable validation summary, signed deviations and pointwise p-values |
| `cleared.json` | `--gate` | BH verdicts for every test, and what clears both filters |
