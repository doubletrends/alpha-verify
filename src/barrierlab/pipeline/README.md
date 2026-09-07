# Pipeline protocol

`pipeline/` owns the seven-stage artifact workflow: what each command reads, writes,
and proves. It does not own reusable feature implementations, data-provider adapters,
workspace declarations, or figure styling; those boundaries are documented in the
[package README](../README.md) and [workspace README](../../../workspaces/README.md).

Run stages through the `barrierlab` CLI from the repository root. The command registry
in [`../cli.py`](../cli.py) is the public execution entrypoint; the artifact names in
`infrastructure/artifacts.py` are the source of truth for paths.

This repository measures conditional barrier-touch probabilities:

```text
P(price touches Δ within t bars | condition bin)
```

That is different from asking where price closes. Stops, limits, liquidations and margin
calls care whether a level was reached at any point inside the forward window, so the
engine uses high/low extremes rather than close-to-close returns.

The product workflow has seven stages:

```text
01_surface/               surface       full arrays and surface workbooks

02_shift/                 shift         baseline-subtracted arrays and workbooks

03_selection/             selection     ranked arrays, manifest, and workbooks

04_validation/            validation    exact null arrays and workbooks
04_validation/validation.json
                          validation    raw null + BH + economic intersection

05_redundancy/            redundancy    numerical matrix and readable workbook
05_redundancy/redundancy.json
                          redundancy    compact manifest and cluster summary
06_composition/composition.npz
                          composition   walk-forward composition arrays
06_composition/composition.json
                          composition   walk-forward composition summary
06_composition/probability.xlsx
                          composition   current weighted probability surface
06_composition/shift.xlsx
                          composition   current weighted shift from baseline
07_report/                report        audience-facing figures
```

Stages 1-4 build the measured, selected and validated surfaces and assign final
validation verdicts. Stage 5 maps redundancy, Stage 6 is the out-of-sample composition
check, and Stage 7 renders figures from stored artifacts. Its null figure reconstructs
the saved null distribution from Stage 3 history but does not fetch or remeasure market data.

The artifact flow is a one-way DAG rather than a strict chain. Shift reads surface
artifacts; selection promotes each top-ranked node's complete shift cube and records its
representative bin; validation reads only those selection artifacts. Redundancy is an
audit branch from the validation decisions. Composition separately reads the Stage 2
candidate panel and Stage 4 cleared-node decisions, then report reads the stored artifacts needed
by each figure. Validation recomputes the circular-shift null from the ordered history
carried by each selected-node artifact because aggregate cube rates alone are insufficient.

## Implementation Boundaries

`src/barrierlab/infrastructure/workspace.py` owns workspace configuration and node
traversal; `src/barrierlab/infrastructure/workspace_plugins.py` owns plugin loading.
`src/barrierlab/infrastructure/artifacts.py` owns artifact paths and history
reconstruction; `src/barrierlab/infrastructure/artifact_io.py` owns JSON and NPZ
persistence. `src/barrierlab/pipeline` owns numbered stage orchestration,
`src/barrierlab/domain` owns numerical models and feature transforms, and
`src/barrierlab/presentation` owns workbooks and figures.
Workspace plugins expose `register(sources, features)` and receive per-run registries;
they no longer mutate process-global registries during import.

## 0. Orient

```bash
barrierlab status --workspace <name>
barrierlab status <node> --workspace <name>
```

The status table is the funnel: nodes declared, surfaces measured, shift artifacts
written, selected representative bins copied, nulls validated, representative-bin
economics passed, and nodes that cleared all three validation gates. Passing a node ID
instead prints that node's strongest baseline-relative shift and available null result.

## 1. Surface

```bash
barrierlab surface --workspace <name>
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
01_surface/array/<node>.safetensors
01_surface/spreadsheet/<node>.xlsx
```

The `.safetensors` file is the measurement record. The workbook is a readable projection: one
tab per condition bin, with barrier rows and horizon columns.

## 2. Shift

```bash
barrierlab shift --workspace <name>
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
barrierlab selection --workspace <name>
```

Selection ranks one *node* per predictor, rather than allowing several decile sheets
from one predictor to consume the shortlist. It uses the workspace's composition target
`P(touch Δ within t)`, calculates the two-sided skew of each bin, and scores the
node by its strongest bin. Here Δ is the positive barrier magnitude:

```text
positive shift = Stage 2 shift at +Δ
negative shift = Stage 2 shift at -Δ
score          = max_bin |positive shift - negative shift|
```

The score is the largest absolute asymmetry between the two stored Stage 2 shifts.
Thus a node with one large two-sided skew outranks a node whose effect is spread weakly
across several bins. Statistical significance and minimum-observation requirements
remain the responsibility of Stage 4.

The stage ranks all nodes globally and writes the top `selection.top_k` rows, defaulting
to 20, to `03_selection/selection.json`. Each selected row records its highest-scoring
**representative bin**. The
complete selected-node cube is promoted to `03_selection/<family>/` as the computational
artifact; its workbook retains only the representative bin as a concise view. Stage 3
contains no economic verdict: a broad, modest
table can rank well even when no individual bin meets a product-effect threshold.

## 4. Validation and Decision

```bash
barrierlab validation --workspace <name>
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
it calculated from the Stage 3 selected-node cube.

A selected node clears the product claim only when all three are true at one horizon:

- **Raw null:** the node-wide shuffled-null p-value is at most `0.01` by default.
- **FDR:** the test survives Benjamini-Hochberg at `q = 0.05` by default.
- **Economic:** a stable, useful-size deviation from the baseline exists.

The result is written to `04_validation/validation.json`:

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

Until every selected node has a current null artifact,
`04_validation/validation.json` records `complete: false` and the missing nodes.

## 5. Redundancy

`redundancy` maps the validation-cleared nodes using normalized
conditional mutual information between their ten-bin states:

```text
I(bin_i ; bin_j | touch outcome)
```

Pairs above the threshold form connected clusters. Stage 5 retains the highest-information
node from each cluster as the equal-weight Naive-Bayes representative.

Stage 5 writes three synchronized artifacts. `05_redundancy/redundancy.npz` is
the numerical source of truth: ordered node ids, information scores, the square
conditional-NMI matrix, cluster ids, representative flags, target, threshold, sample
count and outcome rate. `05_redundancy/redundancy.xlsx` renders Overview, Matrix,
Clusters, Nodes, Pairs and Definitions sheets. `05_redundancy/redundancy.json`
is written last as a
compact manifest; `complete: true` certifies that both larger artifacts exist and match
the current Stage 4 validation fingerprint.

```bash
barrierlab redundancy --workspace <name>
```

## 6. Compose

```bash
barrierlab composition --workspace <name>
```

Stage 5 forwards its cluster representatives. Composition asks whether those independent
representatives can be combined out of sample with equal Naive-Bayes contributions.

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
3. Use the fold-local Stage 5 representative plan, which chooses one node from each
   conditional-dependence cluster using only that fold's training history.
4. Predict the test block with equal node contributions.
5. Score every `horizon`th bar so forward windows do not overlap.

Stage 6 separately fits a deployment-side forecast for the latest jointly available
feature state. For every barrier/horizon cell, it trains on completed outcomes only and
uses the Stage 5 cluster representatives with equal contributions. This is a current forecast, not an average
of the walk-forward predictions. Because separately estimated cells can contain sampling
reversals, a final isotonic projection enforces the defining nesting rules: farther
barriers cannot be more likely than nearer barriers, and longer horizons cannot be less
likely than shorter horizons. The raw fitted face remains in the NPZ for auditability.

It writes:

```text
06_composition/composition.npz
06_composition/composition.json
06_composition/probability.xlsx
06_composition/shift.xlsx
```

`06_composition/composition.json` records prior-only and equal-weight representative
metrics. Every outer fold records its Stage 5 representative set. The point is not to claim a
trading strategy; it is to test whether the measured conditional tables compose without
leaking future data. The NPZ additionally stores the current `Delta x horizon`
probability face and its observation counts. It also stores a leakage-safe report face
fit from the node states and completed outcomes available as of the day Stage 6 runs.
On a non-trading day, that resolves to the latest available market bar. The `probability.xlsx`
single-sheet workbook renders the current face with exactly the same layout, percentage
format, fixed color scale, and frozen panes as a Stage 1 probability sheet.
`shift.xlsx` subtracts the Stage 2 unconditional baseline from that face and
renders the percentage-point difference with Stage 2's signed formatting and fixed
blue/white/red color scale.

## 7. Report

```bash
barrierlab report --workspace <name>
```

The report renders `workspaces/<name>/07_report/*.png`. Figures read
from `.npz` and `.json` artifacts; they do not recompute the pipeline. Plot A uses the
full-Bayes face produced by Stage 6 as of today (or the latest available market bar) and
shows the price path through that date.

The main product figures show:

- the full weighted-Bayes envelope as of today
- the exact null distribution behind a headline node
- one conditional-shift surface for every node that cleared all three
- the ATR regime ladder
- the null-gap ranking across top discoveries

## Extension routing

Add reusable features at the package’s domain boundary; add asset-specific sources,
features, nodes, grids, and thresholds at the workspace boundary. The [package README](../README.md#safe-modification-guide)
and [workspace README](../../../workspaces/README.md) name the correct entrypoints and
contracts. There is no mutable progress field: `barrierlab status` infers progress from
the artifacts on disk.

## Files

| path | written by | contents |
|---|---|---|
| `universe.json` | you | nodes and asset config |
| `01_surface/array/<node>.safetensors` | `surface` | full measured cube |
| `01_surface/spreadsheet/<node>.xlsx` | `surface` | readable full workbook |
| `02_shift/array/<node>.safetensors` | `shift` | full baseline-subtracted shift cube |
| `02_shift/spreadsheet/<node>.xlsx` | `shift` | red/blue shift workbook |
| `03_selection/selection.json` | `selection` | ranking index for selected nodes |
| `03_selection/array/rank_*.safetensors` | `selection` | complete selected-node shift cubes |
| `03_selection/spreadsheet/rank_*.xlsx` | `selection` | representative-bin workbooks |
| `04_validation/array/rank_*.safetensors` | `validation` | exact null artifacts |
| `04_validation/spreadsheet/rank_*.xlsx` | `validation` | readable validation workbook |
| `04_validation/validation.json` | `validation` | three-gate verdicts and cleared rows |
| `05_redundancy/redundancy.npz` | `redundancy` | numerical conditional-NMI matrix and clusters |
| `05_redundancy/redundancy.xlsx` | `redundancy` | six-sheet readable redundancy map |
| `05_redundancy/redundancy.json` | `redundancy` | compact manifest and cluster summary |
| `06_composition/composition.npz` | `composition` | pooled walk-forward predictions |
| `06_composition/composition.json` | `composition` | walk-forward metrics and fold metadata |
| `06_composition/probability.xlsx` | `composition` | one-sheet current weighted probability surface |
| `06_composition/shift.xlsx` | `composition` | one-sheet current weighted shift from baseline |
| `07_report/*.png` | `report` | product figures |

All generated workspace artifacts and report images are git-ignored except the
reviewed NASDAQ evidence image used by the root README. `universe.json` and an optional
`plugin.py` are source-controlled per workspace.
