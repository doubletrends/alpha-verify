# Method

This repository measures conditional barrier-touch probabilities:

```text
P(price touches Δ within t bars | condition bin)
```

That is different from asking where price closes. Stops, limits, liquidations and margin
calls care whether a level was reached at any point inside the forward window, so the
engine uses high/low extremes rather than close-to-close returns.

The product workflow has seven stages:

```text
01_surface/               --surface     full arrays and surface workbooks

02_shift/                 --shift       baseline-subtracted arrays and workbooks

03_selection/             --selection   ranked arrays, manifest, and workbooks

04_validation/            --validation  exact null arrays and workbooks
04_validation/04_validation.json
                          --validation  raw null + BH + economic intersection

05_redundancy/            --redundancy  numerical matrix and readable workbook
05_redundancy/05_redundancy.json
                          --redundancy  compact manifest and cluster summary
06_bayes/06_bayes.npz
                          --bayes       walk-forward composition arrays
06_bayes/06_bayes.json
                          --bayes       walk-forward composition summary
06_bayes/bayes.xlsx       --bayes       current weighted probability surface
06_bayes/bayes_shift.xlsx
                          --bayes       current weighted shift from baseline
workspaces/<name>/result/ --report      audience-facing figures
```

Stages 1-4 build the measured, selected and validated surfaces and assign final
validation verdicts. Stage 5 maps redundancy, Stage 6 is the out-of-sample composition
check, and Stage 7 renders figures from
stored artifacts and does not remeasure.

The dependency chain is one-way: shift reads surface artifacts, selection copies each
top-ranked node's representative-bin artifact from shift, validation reads those
selection artifacts, validation finalizes its own global verdicts, redundancy maps the cleared nodes,
Bayes ranks its full candidate panel within each training fold, and report reads the stored artifacts produced
by those stages. Validation is the one exception that may
recompute the circular-shift null from source data, because the null needs ordered
feature alignment rather than only aggregate cube rates.

## Implementation Boundaries

`infrastructure/workspaces` owns workspace configuration, node traversal, and plugin
loading. `infrastructure/artifacts` owns artifact paths, JSON persistence, and artifact
history reconstruction. `pipeline` owns numbered stage orchestration, `domain` owns
numerical models and feature transforms, and `presentation` owns workbooks and figures.
Workspace plugins expose `register(sources, features)` and receive per-run registries;
they no longer mutate process-global registries during import.

## 0. Orient

```bash
volatility-matrix --workspace <name> --status
```

The status table is the funnel: nodes declared, surfaces measured, shift artifacts
written, selected representative bins copied, nulls validated, representative-bin
economics passed, and nodes that cleared all three validation gates.

## 1. Surface

```bash
volatility-matrix --workspace <name> --surface
volatility-matrix --workspace <name> --surface --family volatility
```

This stage computes one probability cube per node:

| axis | contents |
|---|---|
| 0 | barrier level `Δ`, ascending and including zero |
| 1 | condition bin, usually feature deciles |
| 2 | forward horizon `t` |

The value is the raw conditional probability:

```text
P(price touches Δ within t | feature is in this bin)
```

Nothing is subtracted. The unconditional rate is the `baseline` node: a constant feature
whose single bin contains every bar. Measuring the baseline with the same engine keeps
the pipeline uniform and makes it the reference every conditional node is judged against.

Surface artifacts are written to:

```text
01_surface/<family>/<node>.npz
01_surface/<family>/<node>.xlsx
```

The `.npz` file is the measurement record. The workbook is a readable projection: one
tab per condition bin, with barrier rows and horizon columns.

## 2. Shift

```bash
volatility-matrix --workspace <name> --shift
volatility-matrix --workspace <name> --shift --family volatility
```

The shift stage keeps the full Stage 1 grid and subtracts the baseline node from every
condition bin:

```text
shift = 100 * (P(price touches Δ within t | bin) - P(price touches Δ within t))
```

The shift stage does not judge nodes. It writes the full baseline-subtracted cube that
selection, validation and reporting read.

The workbook uses a diverging scale centered at zero: red cells mean the barrier is
reached more often than baseline, blue cells mean less often.

## 3. Selection

```bash
volatility-matrix --workspace <name> --selection
volatility-matrix --workspace <name> --selection --family volatility
```

Selection ranks one *node* per predictor, rather than allowing several decile sheets
from one predictor to consume the shortlist. It uses the workspace's Bayes target
`P(touch Δ within t)`, and scores the full conditional table:

```text
score = Σ_bin P(bin) × KL(Bernoulli(P(touch | bin)) || Bernoulli(P(touch)))
```

This is expected log-loss improvement per observation, also called mutual information
in nats. A strong effect confined to one decile receives credit only for the bars in
that decile; a weak but useful gradient receives credit from every bin. Conditional
rates use the same 25-pseudo-observation shrinkage as Bayes.

The stage ranks all nodes globally and writes the top `selection.top_k` rows, defaulting
to 20, to `03_selection/selection.json`. Each selected row records its highest-
contributing **representative bin** and its effective number of contributing bins; that
bin is copied to `03_selection/<family>/` as both an array and workbook. Stage 3
contains no economic verdict: a broad, modest
table can rank well even when no individual bin meets a product-effect threshold.

## 4. Validation and Decision

```bash
volatility-matrix --workspace <name> --validation
volatility-matrix --workspace <name> --validation --family volatility
volatility-matrix --workspace <name> --validation --fdr 0.05
```

Validation tests every selected node against an exact circular-shift null. The workbook
still displays its representative bin, but the decision statistic takes the peak over every
bin in that node, accounting for the bin search used to select it. The feature is shifted
against the price history, preserving its autocorrelation while destroying its alignment
with the future.

Stage 4 also applies the economic filter to the representative bin selected by Stage 3.
It passes when a cell:

- deviates from the baseline by at least `min_dev` percentage points
- has at least `min_bin_n` observations in the bin
- persists across at least `min_run` adjacent Δ rows with the same sign

The adjacency requirement prevents one extreme cell from qualifying on its own; a band
of adjacent barriers is closer to something a reader could actually use.

For each horizon, validation stores:

| field | shape | meaning |
|---|---:|---|
| `cell_real` | Δ x bin x t | signed deviation from the node's own sample rate |
| `cell_p` | Δ x bin x t | pointwise p-value for reading a surface |
| `cell_p95` | Δ x bin x t | each cell's null 95th percentile |
| `sheet_peak_real` | bin x t | max absolute deviation in the displayed representative bin |
| `sheet_peak_p` | bin x t | p-value for reading that displayed bin |
| `sheet_peak_p95` | bin x t | null 95th percentile for that displayed bin |
| `node_peak_real` | t | max absolute deviation over all bins in the selected node |
| `node_peak_p` | t | p-value of that node-wide search statistic |
| `node_peak_p95` | t | null 95th percentile of that node-wide statistic |
| `n_shifts` | t | usable shifts, giving the p-value floor `1/(n+1)` |

The exact null is fast because counts by circular shift are a cross-correlation. One FFT
returns every shift. The floor matters: with only `n` distinct shifts, no method can
honestly report a p-value below `1/(n+1)`.

`cell_p` is pointwise. It helps read where a surface is unusual, but it is not a
discovery criterion. Validation uses `node_peak_p` for the nodes selected by
`03_selection/selection.json`, which accounts for the search over their bins and
2D sheets.

Once every selected node has a current null artifact, validation applies a raw
`node_peak_p <= validation.null_alpha` gate, then Benjamini-Hochberg correction across
the selected node/horizon sweep, then intersects those results with the economic filter
it calculated from the Stage 2 shift cube.

A selected node clears the product claim only when all three are true at one horizon:

- **Raw null:** the node-wide shuffled-null p-value is at most `0.01` by default.
- **FDR:** the test survives Benjamini-Hochberg at `q = 0.05` by default.
- **Economic:** a stable, useful-size deviation from the baseline exists.

The result is written to `04_validation/04_validation.json`:

```text
summary.null_pass    selected node/horizons passing the raw null threshold
summary.fdr_pass     selected node/horizons surviving BH correction
summary.discovery   selected node/horizons passing both statistical gates
summary.nominal     pointwise p <= 0.05 before FDR
summary.cleared     selected nodes that cleared all three gates
economics[]          Stage 4 economic result for every selected node
cleared[]           the headline rows
tests[]             every test with null_pass, fdr_pass, economic_pass, and cleared
```

Bonferroni is intentionally not used. With hundreds of tests, its threshold falls below
the exact null's p-value floor, making it unreachable on these workspaces. BH controls
the expected false-discovery share among discoveries, which is the useful correction for
a screen of this size.

Partial `--family` runs may refresh individual null artifacts, but cannot publish a
smaller correction universe. Until every selected node is current,
`04_validation/04_validation.json` records `complete: false` and the missing nodes. The deprecated
`--gate` compatibility alias can only finalize already-complete null artifacts; it is
not a separate pipeline stage.

## 5. Redundancy

`--redundancy` maps the validation-cleared nodes using normalized
conditional mutual information between their ten-bin states:

```text
I(bin_i ; bin_j | touch outcome)
```

Pairs above the threshold form connected clusters. This is a research-facing map, not
a hard deletion rule: it exposes where Naive Bayes' conditional-independence assumption
is weakest while leaving partial information available to the weighted model.

Stage 5 writes three synchronized artifacts. `05_redundancy/redundancy.npz` is
the numerical source of truth: ordered node ids, information scores, the square
conditional-NMI matrix, cluster ids, representative flags, target, threshold, sample
count and outcome rate. `05_redundancy/redundancy.xlsx` renders Overview, Matrix,
Clusters, Nodes, Pairs and Definitions sheets. `05_redundancy/05_redundancy.json`
is written last as a
compact manifest; `complete: true` certifies that both larger artifacts exist and match
the current Stage 4 validation fingerprint.

```bash
volatility-matrix --workspace <name> --redundancy
```

## 6. Compose

```bash
volatility-matrix --workspace <name> --bayes
```

Stages 1-5 evaluate one selected condition sheet at a time. Composition asks whether
the best whole-node conditional tables can be combined out of sample.

Under conditional independence, log-odds add:

```text
logit P(touch Δ in t | x1..xk)
  = logit P(touch Δ in t)
    + sum_i [logit P(touch Δ in t | xi) - logit P(touch Δ in t)]
```

Every term already lives in the measured cubes. The composition stage tests this with an
expanding walk-forward design:

1. Fit bin edges, per-bin rates and prior on the training window only.
2. Embargo the last `horizon` training bars so labels do not reach into test data.
3. Score every complete bin table by training-window information and retain the top nodes.
4. Cross-fit their log-odds contributions inside the training window.
5. Fit non-negative ridge-logistic reliability weights, tuning ridge strength on a later
   chronological calibration slice.
6. Predict the test block.
7. Score every `horizon`th bar so forward windows do not overlap.

After the configured headline target, Stage 6 evaluates every non-zero barrier/horizon
pair in the workspace grid. A sweep row is always one explicit `(Delta, horizon)` pair;
the zero barrier is excluded because its touch event is near-degenerate. The resulting
grid is a diagnostic surface, not a collection of automatically publishable claims:
rare-event rows must be interpreted with their realized rate and scored-bar count.

Stage 6 separately fits a deployment-side forecast for the latest jointly available
feature state. For every barrier/horizon cell, it trains on completed outcomes only,
uses the Stage 4 validation-cleared nodes, cross-fits their contributions, and learns
target-specific non-negative ridge weights. This is a current forecast, not an average
of the walk-forward predictions. Because separately estimated cells can contain sampling
reversals, a final isotonic projection enforces the defining nesting rules: farther
barriers cannot be more likely than nearer barriers, and longer horizons cannot be less
likely than shorter horizons. The raw fitted face remains in the NPZ for auditability.

It writes:

```text
06_bayes/06_bayes.npz
06_bayes/06_bayes.json
06_bayes/bayes.xlsx
06_bayes/bayes_shift.xlsx
```

`06_bayes/06_bayes.json` records prior-only, raw top-node Naive Bayes,
one-node-per-family,
Platt-scaled, and redundancy-aware weighted metrics. Every outer fold records its node
weights, ridge strength, and fold-local redundancy clusters. The point is not to claim a
trading strategy; it is to test whether the measured conditional tables compose without
leaking future data. The NPZ additionally stores the current `Delta x horizon`
probability face and its observation counts. When `report.demonstration_date` is set in
`universe.json`, it also stores a leakage-safe historical face fit from the node states
and completed outcomes available on that date. Historical cells without enough
cross-fitted labels to learn reliability weights fall back to that target's historical
prior, rather than silently presenting unit-weight Naive Bayes. The `bayes.xlsx`
single-sheet workbook renders the current face with exactly the same layout, percentage
format, fixed color scale, and frozen panes as a Stage 1 probability sheet.
`bayes_shift.xlsx` subtracts the Stage 2 unconditional baseline from that face and
renders the percentage-point difference with Stage 2's signed formatting and fixed
blue/white/red color scale.

## 7. Report

```bash
volatility-matrix --workspace <name> --report
```

The report renders `workspaces/<name>/result/*.png` plus a local index. Figures read
from `.npz` and `.json` artifacts; they do not recompute the pipeline. Plot A uses the
historical full-Bayes face produced by Stage 6 for `report.demonstration_date` and
overlays the subsequently realized price path.

The main product figures show:

- the full weighted-Bayes envelope from the configured demonstration date
- the exact null distribution behind a headline node
- one conditional-shift surface for every node that cleared all three
- the ATR regime ladder
- the null-gap ranking across top discoveries

## Adding a Feature

Register reusable feature functions in `src/domain/features.py`:

```python
def _my_feature(close: pd.Series, period: int) -> pd.Series:
    ...

def register_builtin_features(registry: FeatureRegistry) -> None:
    registry.register("my_feature", lambda d, p: _my_feature(d["close"], p["period"]))
```

Workspace-specific sources and features live in `workspaces/<name>/plugin.py`, which
defines `register(sources, features)`. Then add a node to
`workspaces/<name>/universe.json`.

There is no progress field. Progress is inferred from artifacts on disk.

## Adding a Workspace

`universe.json` `meta` carries the asset-specific configuration:

| field | meaning |
|---|---|
| `asset` | `{provider, ticker, interval}` |
| `start_date` | history start |
| `horizons` | full horizon ladder |
| `Δ` | full barrier ladder |
| `n_bins` | condition bins per feature |
| `evaluate` | economic filter thresholds |
| `validation` | raw node-null threshold, default `null_alpha: 0.01` |
| `bayes` | composition target and fold count |

Scale `Δ` to the asset and horizon. BTC daily can support wider barriers than a
daily equity index; a good workspace spends barrier rows where touches occur often
enough to estimate.

## Files

| path | written by | contents |
|---|---|---|
| `universe.json` | you | nodes and asset config |
| `01_surface/<family>/<node>.npz` | `--surface` | full measured cube |
| `01_surface/<family>/<node>.xlsx` | `--surface` | readable full workbook |
| `02_shift/<family>/<node>.npz` | `--shift` | full baseline-subtracted shift cube |
| `02_shift/<family>/<node>.xlsx` | `--shift` | red/blue shift workbook |
| `03_selection/selection.json` | `--selection` | ranking index for selected nodes |
| `03_selection/<family>/rank_*.npz` | `--selection` | copied representative-bin arrays |
| `03_selection/<family>/rank_*.xlsx` | `--selection` | representative-bin workbooks |
| `04_validation/<family>/<node>.npz` | `--validation` | exact null artifacts |
| `04_validation/<family>/<node>.xlsx` | `--validation` | readable validation workbook |
| `04_validation/04_validation.json` | `--validation` | three-gate verdicts and cleared rows |
| `05_redundancy/redundancy.npz` | `--redundancy` | numerical conditional-NMI matrix and clusters |
| `05_redundancy/redundancy.xlsx` | `--redundancy` | six-sheet readable redundancy map |
| `05_redundancy/05_redundancy.json` | `--redundancy` | compact manifest and cluster summary |
| `06_bayes/06_bayes.npz` | `--bayes` | pooled walk-forward predictions |
| `06_bayes/06_bayes.json` | `--bayes` | walk-forward metrics and fold metadata |
| `06_bayes/bayes.xlsx` | `--bayes` | one-sheet current weighted probability surface |
| `06_bayes/bayes_shift.xlsx` | `--bayes` | one-sheet current weighted shift from baseline |
| `workspaces/<name>/result/*.png` | `--report` | product figures |

Rendered workbooks are git-ignored. The `.npz` files are the measurement record.
