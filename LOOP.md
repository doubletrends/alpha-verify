# Pipeline — Procedure

One measurement, one deliverable. The measurement is a cube per node; the deliverable
is a workbook projected out of it. Nothing else is produced.

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

## 1. Measure — the cube (pre-A)

```
python run.py --workspace <name> --cubes
python run.py --workspace <name> --cubes --family volatility   # or one family
```

Writes `cubes/<family>/<node>.npz` — a 3-D array per node:

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

`--validate` skips it: shuffling a constant changes nothing, so the null is vacuous, and
excluding it keeps it out of the Bonferroni denominator where it would only make real
tests harder to pass.

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

## 2. Render — Deliverable A

```
python run.py --workspace <name> --surfaces
```

Reads the cubes and writes `surfaces/<family>/<node>.xlsx` — **one tab per condition
bin**, each holding that bin's entire θ × h face:

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

## 3. Validate

```
python run.py --workspace <name> --validate --shifts 20000
```

A surface's headline is a maximum over 41 barrier levels × 10 bins. That is 400 noisy
estimates, and the largest lands well into the tens of pp even on a feature carrying
nothing — a bigger bias than any one-dimensional scan, because the search is
two-dimensional. This builds the null of that same statistic by circularly shifting the
feature against price, preserving the feature's autocorrelation and destroying only its
alignment with the future.

Runs on the horizons in `barrier_horizons`, not all 30 — adjacent horizons are nearly
the same measurement, so testing every one would inflate the Bonferroni denominator
without adding independent evidence.

Writes `validation.json`. Verdicts: `structure` (clears Bonferroni), `nominal` (clears
α alone), `underpowered` (at the resolution floor — re-run with more shifts), `noise`.
Narrowing with `--family` **merges** into the existing file, and the correction is taken
over every test the merged file holds.

Choose `--shifts` so the floor `1/(shifts+1)` sits below `0.05 / n_tests`.

---

## 4. The gate

```
python run.py --workspace <name> --gate
python run.py --workspace <name> --gate --require nominal   # looser
```

Intersects the two filters and writes `cleared.json`.

- **Economic** (from `--surfaces`): a cell deviates from the base rate by at least
  `min_dev` pp, in a bin holding at least `min_bin_n` observations, across at least
  `min_run` *adjacent* θ rows with the same sign.
- **Statistical** (from `--validate`): the surface clears its shuffled null.

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
| `barrier_horizons` | horizons the null test runs on |
| `theta` | `{min, max, step}` — the barrier ladder |
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
| `cubes/_base/baseline.npz` | `--cubes` | the unconditional reference, built first |
| `cubes/<family>/<node>.npz` | `--cubes` | **the measurement** — (θ × bin × h) probabilities |
| `surfaces/<family>/<node>.xlsx` | `--surfaces` | **the deliverable** — 10 tabs, the cube made readable |
| `evaluation.json` | `--cubes` | economic filter verdict per node |
| `validation.json` | `--validate` | null-test p-values and verdicts |
| `cleared.json` | `--gate` | nodes clearing both filters |
