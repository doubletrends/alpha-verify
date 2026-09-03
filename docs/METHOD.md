# Method

The per-node artifact folders are numbered by command: `surface` writes `01_*`,
`summary` writes `02_*`, `skew` writes `03_*`, and `validation` writes `04_*`. The
remaining outputs are run-level summaries and report figures. The full grid is measured
but never judged; a coarse subset of it is what gets judged and validated.

```text
01_surface_array/      --surface     compute: full theta x bins x horizons
01_surface_xlsx/       --surface     view: full surface workbooks

02_summary_array/      --summary     compute: judged theta x bins x horizons
02_summary_xlsx/       --summary     view: summary workbooks

03_skew_array/         --skew        compute: P(+theta) vs P(-theta)
03_skew_xlsx/          --skew        view: readable skew sheets

04_validation_array/   --validation  compute: exact shuffled nulls
04_validation_xlsx/    --validation  view: readable validation sheets

05_gate.json           --gate        run-level BH correction + economic intersect
06_bayez.npz           --bayes       run-level walk-forward composition arrays
06_bayes.json          --bayes       run-level walk-forward composition summary
workspaces/<name>/result/ --report   audience-facing figures
```

Pass `--workspace <name>` to every command. Stages 1-5 describe what happened over the
whole history; stage 6 is the only one that makes a prediction, and it is the only one
fit on a training window. See [../README.md](../README.md) for the findings.

---

## The question this answers

> Given that condition X holds, how likely is price to *reach* level θ within h bars?

That is the question a stop, a limit, a liquidation and a margin call all turn on, and
it is not the same as asking where price closes. The pipeline answers it for every
barrier level, every condition and every horizon at once, then filters the result twice
before letting you believe any of it.

---

## 0 — Orient

```
python run.py --workspace <name> --status
```

The columns are the funnel: how many nodes exist, how many have a cube, how many clear
the economic filter, how many clear the null, how many clear both.

---

## 1 — Surface

```
python run.py --workspace <name> --surface
python run.py --workspace <name> --surface --family volatility   # or one family
```

Writes `01_surface_array/<family>/<node>.npz` — a 3-D array per node — and
`01_surface_xlsx/<family>/<node>.xlsx`, one workbook view per node.

| axis | contents |
|---|---|
| 0 | barrier level θ, ascending, **including 0** |
| 1 | condition bin — the feature's deciles |
| 2 | horizon h, every configured horizon, not a chosen few |

The value is the raw conditional probability:

```
P(price touches θ within h | X in bin)
```

Nothing is subtracted. The unconditional rate is **its own node**: `baseline`, whose
feature is constant, so every bar falls in a single bin and its one column is
P(touch θ in h) with nothing conditioned on. It is built first — it is the reference
every other node is judged against — and it flows through `--surface`
with no special case in the engine.

`--validation` runs on it like everything else and produces a degenerate artifact —
shuffling a constant cannot change the surface, so its peak deviation is exactly 0 and
its p-value exactly 1. It is never a discovery, and keeping it in costs one file and
removes a branch from the pipeline.

Caveat worth knowing: the baseline covers every bar, while a long-warmup feature covers
fewer, and the bars it loses are the oldest. For a 200-bar moving average that is a
material slice of the sample, so the comparison is very slightly against a different
past.

The file also carries `hits`, `bin_n`, `n_obs`, the three coordinate vectors and the bin
edges, so anything derivable from the measurement can be derived without recomputing it.
About 90 KB per node; loads in milliseconds.

`--rerun` rebuilds surface arrays that already exist.

**Why this is its own step.** The full horizon ladder is where the shape of a signal
lives: whether the edge decays or persists, whether it is a band of barrier levels or a
single spike. Four sheets cannot show that. Measuring once into a cube also makes the
workbook a *projection* rather than a second computation, so the two can never disagree.

The workbook has **one tab per condition bin**, each holding that bin's entire θ × h face:

| | |
|---|---|
| rows | barrier level θ, highest barrier at the top and lowest at the bottom |
| columns | horizon h, across the configured full horizon ladder |
| cells | `P(touch θ in h │ condition)` — the raw probability |
| n band | observations behind that bin at each horizon |

Ten tabs times the workspace's full theta ladder times the full horizon ladder is every
value the cube holds, so the workbook is a faithful view of the measurement rather than
a summary of it.

Rows run the way a price ladder reads: up the sheet is up in price, with θ = 0 shaded in
the middle marking the boundary between the two questions — above it a cell asks whether
the *high* reached that level, below it whether the *low* did.

Read **across a row** to see the probability grow with horizon; read **down a column**
to see the band of levels price is likely to reach. Compare the +θ and −θ rows at the
same horizon and the condition's asymmetry is immediate, with no arithmetic.

The colour scale is a fixed 0–100% on every tab, so flipping between tabs shows the band
shifting with the condition instead of each tab being rescaled to look alike.

**Reading a sheet.** Negative θ asks whether the *low* reached it, positive θ whether
the *high* did — both on intraday extremes, not closes, because a stop fills on the low.
Two dead zones are expected and are not signal: θ near zero is touched almost always,
and large |θ| at short horizons is touched almost never.

---

## 2 — Summary

```
python run.py --workspace <name> --summary
python run.py --workspace <name> --summary --family volatility   # or one family
```

Selects a coarse sub-grid out of each full cube and writes `02_summary_array/`.
`02_summary_xlsx/` is the workbook view of that same data. Pure index selection:
every summary cell is bit-identical to the
full cell it came from, and a requested θ or h that is not on the full grid raises
rather than interpolating — interpolating would make the summary a second measurement,
and then the two could disagree.

**Why judge a subset.** Adjacent cells of the full grid are very nearly the same
measurement. A peak over every full-grid cell has a large noise ceiling by construction,
and counting all of them as tests overstates the search in both directions at once. On
the summary grid the cells are farther apart and the test count is honest.

The `--read <node>` command prints a node's strongest bin, its qualifying θ rows and its
null test side by side in the terminal, straight off the stored artifacts.

The **economic filter runs here**, not on the full cube, so that what is judged and what
is validated sit on the same grid. Writes `evaluation.json`.

Configure with `summary.theta_abs` (magnitudes, mirrored) and `summary.horizons`.

---

## 3 — Skew

```
python run.py --workspace <name> --skew
python run.py --workspace <name> --skew --family volatility   # or one family
```

The mirrored θ ladder exists so `P(touch +θ)` and `P(touch −θ)` can be read against each
other in the same column. This stage computes that, per +/- pair, horizon and condition
bin, and writes `03_skew_array/<family>/<node>.npz` + `03_skew_xlsx/<family>/<node>.xlsx`.

| | |
|---|---|
| `cond_skew` | `P(+θ \| bin) − P(−θ \| bin)`, pp — the raw asymmetry, still carrying the asset's drift |
| `excess_skew` | `cond_skew` minus the **baseline node's** own conditional skew — drift removed, so what is left is the asymmetry *this condition* introduces |
| `prob_up`, `prob_dn` | the paired rates, carried so a sheet can show the pair |

It is pure arithmetic on the summary cube's `prob` array — no second measurement, so a
skew cell and the two probability cells behind it can never disagree. The baseline node
gets an artifact like everything else; its `excess_skew` is identically 0 by
construction.

`excess_skew` is the quantity **`--validation` null-tests** (its peak per horizon, same
circular-shift null, no extra FFT), and **`--gate` corrects** as a separate BH family.
Skew is *reported, not gated*: what clears the pipeline is still the touch peak
intersected with the economic filter. `--validation` requires this stage to have run.

---

## 4 — Validation

```
python run.py --workspace <name> --validation
python run.py --workspace <name> --validation --family volatility   # or one family
```

Runs on the **summary** cube, using its stored θ, horizons and bin edges, so the test
and the thing it judges are the same grid down to the last cell. Requires `--skew` to
have run: it null-tests the touch peak and the excess-skew peak in the same pass.

Writes `04_validation_array/<family>/<node>.npz`, one per node — the baseline included, whose
artifact comes out degenerate because shuffling a constant cannot change anything. It
also writes `04_validation_xlsx/<family>/<node>.xlsx`, a readable workbook with a summary
tab plus per-bin signed-deviation and pointwise-p tabs. Per-node artifacts also mean two
runs never write the same path, which is what made a shared validation file
untrustworthy.

| | |
|---|---|
| `cell_real` | (θ, bin, h) signed deviation from the node's own sample rate |
| `cell_p` | (θ, bin, h) **pointwise** p-value — for reading a surface, never a discovery |
| `cell_p95` | (θ, bin, h) 95th percentile of each cell's own null |
| `peak_real`, `peak_p`, `peak_p95` | (h,) the max \|deviation\| over the surface — the statistic that accounts for the search |
| `skew_peak_real`, `skew_peak_p`, `skew_peak_p95` | (h,) the same, for max \|excess skew\| over the mirrored (magnitude, bin) face |
| `n_shifts` | (h,) usable shifts, so the p-value floor is `1/(n+1)` |

The skew peak reuses the same shift-resolved counts the touch null already computes —
`P(+θ)` and `P(−θ)` per shift are recoverable from them — so it costs no extra FFT.

**The null is exact.** `counts[θ, bin]` as a function of shift is a circular
cross-correlation, so one FFT gives every shift at once — ~70× faster than resampling
and it returns the whole permutation distribution. On the summary grid the whole
workspace validates in well under a minute.

Counts are rounded to integers after the transform. They are counts of 0/1 indicators
and so exact by construction; the FFT returns them with ~1e-13 of roundoff. That matters
most in the degenerate case — a single-bin cube has a deviation of identically zero, and
without rounding the null compares one speck of numerical noise against another and
returns a meaningless p-value instead of 1.

**Its floor is real.** Only *n* distinct shifts exist, so no p-value below `1/(n+1)` is
obtainable. Resampling more shifts than the group contains and reporting `1/(draws+1)`
claims a resolution the data cannot produce.

**`cell_p` is pointwise.** On a large grid, many cells fall below 0.05 by chance. Use it
to read where a surface is unusual; never as a criterion for finding something.

---

## 5 — The gate

```
python run.py --workspace <name> --gate
python run.py --workspace <name> --gate --fdr 0.05
```

Corrects across the whole sweep, then intersects with the economic filter, and writes
`05_gate.json`.

- **Economic** (from `--summary`): a cell deviates from the baseline node by at least
  `min_dev` pp, in a bin holding at least `min_bin_n` observations, across at least
  `min_run` *adjacent* θ rows with the same sign.
- **Statistical** (from `--validation`): the node's peak p-value survives
  Benjamini–Hochberg at `q`.

The **skew peak** is corrected here too, as a *separate* BH family (it asks a different
question, so pooling the p-values would mis-correct both), and written to `05_gate.json`
as `skew_summary` / `skew_tests`. It is reported alongside — it does **not** change what
clears.

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

What clears the gate is a measured, falsified statement about where price reaches under
a stated condition. It is not a strategy, and it is not a prediction — every number in
stages 1-5 is measured over the whole history, which is the right way to describe what
happened and the wrong way to claim anything about what happens next. That claim is
stage 6's job, and it is held to a different standard.

---

## 6 — Compose

```
python run.py --workspace <name> --bayes
```

Every stage so far measures conditions one at a time. The obvious next question — what
happens when several hold at once — already has its answer in the artifacts, because
under conditional independence the log-odds add and every term is a cube cell:

```
logit P(touch θ in h | x1..xk)
    = logit P(touch θ in h) + Σ_i [ logit P(touch θ in h | x_i) − logit P(touch θ in h) ]
```

So the ensemble is a sum over existing measurements: no new estimator, no fitted weights.
What is new is the discipline. Writes `06_bayez.npz` and `06_bayes.json`.

**The walk forward.** Expanding window, `folds` test blocks over the last half of the
sample. Per fold, in order:

1. cut the training window at the fold's start, then drop its last `horizon` bars — the
   **embargo**, because those labels reach into the test block
2. estimate bin edges, per-bin rates and the prior on what is left
3. rank features by training-window separation and keep **one per family**
4. predict the test block with the edges and tables from step 2
5. score every `horizon`-th test bar, so the scored windows do not overlap

Nothing crosses from step 4 back to step 2. The deduplication in step 3 is inside the
fold on purpose: choosing representatives over the whole history would be a selection
made with hindsight, and it is exactly the kind of leak this stage exists to avoid.

**Three models, so the failures are measured rather than asserted.**

| model | what it shows |
|---|---|
| every node | the independence assumption failing: 65 collinear features count one piece of evidence many times, and the forecast runs to 94% where the realized rate never leaves 25–33% |
| one per family | most of the collinearity removed — still worse than a constant prior |
| + scale corrected | a two-parameter Platt sigmoid fit on the training window; now it beats the prior |

Platt scaling uses smoothed targets, `(N₊+1)/(N₊+2)` and `1/(N₋+2)`, rather than hard
0/1 labels: maximum likelihood against hard labels is unbounded whenever the training
scores separate, and the fit runs away to `a ≈ 1e8`. The Newton step is damped and
backtracked for the same reason — the raw score saturates the logistic, the IRLS weights
underflow, and an undamped step leaves the basin.

`a < 1` is the measured overconfidence. AUC is identical before and after, because
rescaling moves calibration and never order — which is the point: the ranking was real
all along, the confidence was not.

The same walk forward is then run at every target on the judged grid, so the composition
result is reported on the axes the rest of the pipeline uses.

---

## 7 — Report

```
python run.py --workspace <name> --report
```

Renders `workspaces/<name>/result/*.png` plus an index, from the artifacts on disk. Nothing here
measures anything: every number in every figure is read back out of an `.npz` or a
`.json` an earlier stage wrote, for the same reason the workbook renderer works that way
— a picture that recomputes its own inputs can disagree with the file it illustrates.

The one exception is the null histogram, which needs the whole shift distribution rather
than the three numbers the validation artifact stores. It re-derives it from the node's
own stored coordinates and edges, so it is the same test on the same grid, unreduced.

Two presentation choices are worth naming, because both bend a rule the pipeline holds
elsewhere:

- **The band figure interpolates.** `barrier.touch_band` inverts the surface — *what
  level does price touch half the time* — and finds the crossing between two measured θ
  rows one percentage point apart. The refusal to interpolate applies to `summarize`,
  where an interpolated cell would be judged and counted as a test. Nothing here is
  judged. Where the crossing falls outside the ladder, or finer than one step of it, the
  result is NaN rather than a clipped value that would draw a flat ceiling and read as a
  finding.
- **The figures render the full grid, not the summary.** The full cube is the faithful
  record, and 41 rows a percentage point apart draw a smooth cone where 17 draw a
  staircase. The summary grid exists to be judged on.

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
| `bayes` | `{theta, horizon, folds}` — stage 8's headline target; derived from the grid when absent |

**Scale `theta` to the asset and horizon.** BTC daily runs ±20% in 1% steps; NASDAQ
daily runs ±15% in 0.5% steps. A workspace should spend its barrier rows where touches
actually occur often enough to estimate.

---

## Files

| Path | Written by | Contents |
|------|-----------|----------|
| `universe.json` | you | nodes + asset config; never written by the pipeline |
| `01_surface_array/<family>/<node>.npz` | `--surface` | the faithful record — (θ × bin × h) probabilities |
| `01_surface_xlsx/<family>/<node>.xlsx` | `--surface` | 10 tabs, the full cube made readable |
| `02_summary_array/<family>/<node>.npz` | `--summary` | the judged grid, a strict subset |
| `02_summary_xlsx/<family>/<node>.xlsx` | `--summary` | 10 tabs, the summary made readable |
| `evaluation.json` | `--summary` | economic filter verdict per node |
| `03_skew_array/<family>/<node>.npz` | `--skew` | conditional and excess up/down skew, `(magnitude × bin × h)` |
| `03_skew_xlsx/<family>/<node>.xlsx` | `--skew` | readable skew sheets: summary tab + per-bin conditional and excess tabs |
| `04_validation_array/<family>/<node>.npz` | `--validation` | per-cell and peak nulls (touch + skew), one per node |
| `04_validation_xlsx/<family>/<node>.xlsx` | `--validation` | readable validation summary, signed deviations and pointwise p-values |
| `05_gate.json` | `--gate` | BH verdicts for every test, what clears both filters, and the separate skew BH family |
| `06_bayez.npz` | `--bayes` | pooled out-of-sample predictions of all three models at the headline target, plus the metrics across the judged grid |
| `06_bayes.json` | `--bayes` | the walk-forward design, per-fold boundaries and which node each fold kept per family, and the scores |
| `workspaces/<name>/result/*.png` | `--report` | the figures, plus a `README.md` index |

Rendered workbooks are git-ignored: they are a deliverable, ~25 MB across 400 files, and
regenerate in a couple of minutes. The `.npz` arrays are tracked, because they are the
measurement.
