# Pipeline — Procedure

One deliverable: the barrier surface, per node. Nothing else is produced.

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

The columns are the funnel: how many nodes exist, how many have a surface, how many
clear the economic filter, how many clear the null, how many survive both.

---

## 1. Build the surfaces

```
python run.py --workspace <name> --surfaces
python run.py --workspace <name> --surfaces --family volatility   # or one family
```

Writes `surfaces/<family>/<node>.xlsx` — four sheets, one per horizon. Rows are barrier
levels θ, columns are the feature's quantile bins, cells are
P(price touches θ within h | feature in bin) with the event count beside them. Column B
is the unconditional base rate, so every cell reads against the row it sits on.

Also writes `evaluation.json`, the economic filter's verdict per node.
`--rerun` rebuilds surfaces that already exist.

**Reading a sheet.** Negative θ asks whether the *low* reached it, positive θ whether
the *high* did — both on intraday extremes, not closes, because a stop fills on the low.
Two dead zones are expected and are not signal: θ near zero is touched almost always,
and large |θ| at short horizons is touched almost never.

---

## 2. Validate

```
python run.py --workspace <name> --validate --shifts 20000
```

A surface's headline is a maximum over ~40 barrier levels × ~10 bins. That is 400 noisy
estimates, and the largest lands well into the tens of pp even on a feature carrying
nothing — a bigger bias than any one-dimensional scan, because the search is
two-dimensional. This builds the null of that same statistic by circularly shifting the
feature against price, preserving the feature's autocorrelation and destroying only its
alignment with the future.

Writes `validation.json`. Verdicts: `structure` (clears Bonferroni), `nominal` (clears
α alone), `underpowered` (at the resolution floor — re-run with more shifts), `noise`.

Choose `--shifts` so the floor `1/(shifts+1)` sits below `0.05 / n_tests`.

---

## 3. The gate

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
| `barrier_horizons` | the four sheets of Deliverable A |
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
| `surfaces/<family>/<node>.xlsx` | `--surfaces` | **the deliverable** |
| `evaluation.json` | `--surfaces` | economic filter verdict per node |
| `validation.json` | `--validate` | null-test p-values and verdicts |
| `cleared.json` | `--gate` | nodes clearing both filters |
