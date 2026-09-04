# Method

This repository measures conditional barrier-touch probabilities:

```text
P(price touches theta within h bars | condition bin)
```

That is different from asking where price closes. Stops, limits, liquidations and margin
calls care whether a level was reached at any point inside the forward window, so the
engine uses high/low extremes rather than close-to-close returns.

The product workflow has seven stages:

```text
01_surface_array/         --surface     compute: full theta x bins x horizons
01_surface_xlsx/          --surface     view: full surface workbooks

02_summary_array/         --summary     compute: judged theta x bins x horizons
02_summary_xlsx/          --summary     view: summary workbooks
evaluation.json           --summary     economic filter results

03_validation_array/      --validation  compute: exact shuffled nulls
03_validation_xlsx/       --validation  view: validation workbooks

04_skew_array/            --skew        compute: informative up/down asymmetry
04_skew_xlsx/             --skew        view: readable skew workbooks

05_gate.json              --gate        BH correction + economic intersect
06_bayes.npz              --bayes       walk-forward composition arrays
06_bayes.json             --bayes       walk-forward composition summary
workspaces/<name>/result/ --report      audience-facing figures
```

Stages 1-5 describe what happened over the whole history. Stage 6 is the out-of-sample
composition check. Stage 7 renders figures from stored artifacts and does not remeasure.

## 0. Orient

```bash
python run.py --workspace <name> --status
```

The status table is the funnel: nodes declared, surfaces measured, summary artifacts
written, nulls validated, skew artifacts rendered, economic filter passed, and nodes
that cleared both the economic and statistical gates.

## 1. Surface

```bash
python run.py --workspace <name> --surface
python run.py --workspace <name> --surface --family volatility
```

This stage computes one probability cube per node:

| axis | contents |
|---|---|
| 0 | barrier level `theta`, ascending and including zero |
| 1 | condition bin, usually feature deciles |
| 2 | forward horizon `h` |

The value is the raw conditional probability:

```text
P(price touches theta within h | feature is in this bin)
```

Nothing is subtracted. The unconditional rate is the `baseline` node: a constant feature
whose single bin contains every bar. Measuring the baseline with the same engine keeps
the pipeline uniform and makes it the reference every conditional node is judged against.

Surface artifacts are written to:

```text
01_surface_array/<family>/<node>.npz
01_surface_xlsx/<family>/<node>.xlsx
```

The `.npz` file is the measurement record. The workbook is a readable projection: one
tab per condition bin, with barrier rows and horizon columns.

## 2. Summary

```bash
python run.py --workspace <name> --summary
python run.py --workspace <name> --summary --family volatility
```

The full grid is useful for reading shape, but too dense to judge directly. Adjacent
barrier rows and horizons are highly similar, so searching the full cube inflates the
noise ceiling.

The summary stage selects a coarse subset declared in `universe.json`. Selection is
strict: requested theta and horizon values must already exist on the full grid. There is
no interpolation, so summary cells are bit-identical to full-grid cells.

This stage also runs the economic filter and writes `evaluation.json`. A node passes the
economic filter when it has a cell that:

- deviates from the baseline by at least `min_dev` percentage points
- has at least `min_bin_n` observations in the bin
- persists across at least `min_run` adjacent theta rows with the same sign

That adjacency requirement is important. A single extreme cell can be a search artifact;
a band of adjacent barriers is closer to something a reader could actually use.

## 3. Validation

```bash
python run.py --workspace <name> --validation
python run.py --workspace <name> --validation --family volatility
```

Validation tests the summary cube against an exact circular-shift null. The feature is
shifted against the price history, preserving its autocorrelation while destroying its
alignment with the future.

For each horizon, validation stores:

| field | shape | meaning |
|---|---:|---|
| `cell_real` | theta x bin x h | signed deviation from the node's own sample rate |
| `cell_p` | theta x bin x h | pointwise p-value for reading a surface |
| `cell_p95` | theta x bin x h | each cell's null 95th percentile |
| `peak_real` | h | max absolute deviation over the surface |
| `peak_p` | h | p-value of that max search statistic |
| `peak_p95` | h | null 95th percentile of that max statistic |
| `n_shifts` | h | usable shifts, giving the p-value floor `1/(n+1)` |

The exact null is fast because counts by circular shift are a cross-correlation. One FFT
returns every shift. The floor matters: with only `n` distinct shifts, no method can
honestly report a p-value below `1/(n+1)`.

`cell_p` is pointwise. It helps read where a surface is unusual, but it is not a
discovery criterion. The gate uses `peak_p`, which accounts for the search over the
surface.

## 4. Skew

```bash
python run.py --workspace <name> --skew
python run.py --workspace <name> --skew --family volatility
```

Skew is an informative product artifact derived from the summary cubes. It compares the
mirrored barrier ladder in each condition bin:

```text
conditional skew = P(touch +theta | bin) - P(touch -theta | bin)
excess skew      = conditional skew - baseline conditional skew
```

This stage does not remeasure price, does not run a null, and does not feed
cleared-both. It exists so direction/asymmetry can be inspected beside the probability
surfaces without turning it into a claim.

## 5. Gate

```bash
python run.py --workspace <name> --gate
python run.py --workspace <name> --gate --fdr 0.05
```

The gate applies Benjamini-Hochberg correction across the sweep, then intersects those
discoveries with the economic filter from `evaluation.json`.

A node clears the product claim only when both are true:

- **Economic:** a stable, useful-size deviation from the baseline exists.
- **Statistical:** the node's peak survives the exact shuffled-null test after FDR
  correction.

The result is written to `05_gate.json`:

```text
summary.discovery   nodes/horizons that survive the null
summary.nominal     pointwise p <= 0.05 before FDR
summary.cleared     nodes that cleared both filters
cleared[]           the headline rows
tests[]             every validation test with verdict and q-value
```

Bonferroni is intentionally not used. With hundreds of tests, its threshold falls below
the exact null's p-value floor, making it unreachable on these workspaces. BH controls
the expected false-discovery share among discoveries, which is the useful correction for
a screen of this size.

## 6. Compose

```bash
python run.py --workspace <name> --bayes
```

Stages 1-4 measure one condition at a time. Composition asks whether several conditions
can be combined out of sample.

Under conditional independence, log-odds add:

```text
logit P(touch theta in h | x1..xk)
  = logit P(touch theta in h)
    + sum_i [logit P(touch theta in h | xi) - logit P(touch theta in h)]
```

Every term already lives in the measured cubes. The composition stage tests this with an
expanding walk-forward design:

1. Fit bin edges, per-bin rates and prior on the training window only.
2. Embargo the last `horizon` training bars so labels do not reach into test data.
3. Keep one feature per family based only on training-window separation.
4. Predict the test block.
5. Score every `horizon`th bar so forward windows do not overlap.

It writes:

```text
06_bayes.npz
06_bayes.json
```

The report compares prior-only, all-node naive Bayes, one-node-per-family, and Platt
scale-corrected probabilities. The point is not to claim a trading strategy; it is to
test whether the measured conditional tables compose without leaking future data.

## 7. Report

```bash
python run.py --workspace <name> --report
```

The report renders `workspaces/<name>/result/*.png` plus a local index. Figures read
from `.npz` and `.json` artifacts; they do not recompute the pipeline.

The main product figures show:

- the measured forward envelope from the latest close
- the exact null distribution behind a headline node
- one conditional-shift surface for every node that cleared both
- the composition ranking grid
- the ATR regime ladder
- the null-gap ranking across top discoveries

## Adding a Feature

Register reusable feature functions in `data/features.py`:

```python
def _my_feature(close: pd.Series, period: int) -> pd.Series:
    ...

register("my_feature", lambda d, p: _my_feature(d["close"], p["period"]))
```

Workspace-specific sources and features live in `workspaces/<name>/plugin.py`, which is
loaded automatically. Then add a node to `workspaces/<name>/universe.json`.

There is no progress field. Progress is inferred from artifacts on disk.

## Adding a Workspace

`universe.json` `meta` carries the asset-specific configuration:

| field | meaning |
|---|---|
| `asset` | `{provider, ticker, interval}` |
| `start_date` | history start |
| `horizons` | full horizon ladder |
| `theta` | full barrier ladder |
| `summary` | coarse grid used for judging |
| `n_bins` | condition bins per feature |
| `evaluate` | economic filter thresholds |
| `bayes` | composition target and fold count |

Scale `theta` to the asset and horizon. BTC daily can support wider barriers than a
daily equity index; a good workspace spends barrier rows where touches occur often
enough to estimate.

## Files

| path | written by | contents |
|---|---|---|
| `universe.json` | you | nodes and asset config |
| `01_surface_array/<family>/<node>.npz` | `--surface` | full measured cube |
| `01_surface_xlsx/<family>/<node>.xlsx` | `--surface` | readable full workbook |
| `02_summary_array/<family>/<node>.npz` | `--summary` | judged grid |
| `02_summary_xlsx/<family>/<node>.xlsx` | `--summary` | readable summary workbook |
| `evaluation.json` | `--summary` | economic filter results |
| `03_validation_array/<family>/<node>.npz` | `--validation` | exact null artifacts |
| `03_validation_xlsx/<family>/<node>.xlsx` | `--validation` | readable validation workbook |
| `04_skew_array/<family>/<node>.npz` | `--skew` | informative up/down skew artifacts |
| `04_skew_xlsx/<family>/<node>.xlsx` | `--skew` | readable skew workbook |
| `05_gate.json` | `--gate` | FDR verdicts and cleared-both rows |
| `06_bayes.npz` | `--bayes` | pooled walk-forward predictions |
| `06_bayes.json` | `--bayes` | walk-forward metrics and fold metadata |
| `workspaces/<name>/result/*.png` | `--report` | product figures |

Rendered workbooks are git-ignored. The `.npz` files are the measurement record.
