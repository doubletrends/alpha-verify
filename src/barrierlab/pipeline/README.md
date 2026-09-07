# Pipeline protocol

`pipeline/` owns the four-stage artifact workflow: what each command reads, writes,
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

The product workflow has four stages:

```text
01_surface/               surface       full arrays and surface workbooks

02_shift/                 shift         baseline-subtracted arrays and workbooks

03_selection/             selection     ranked arrays, manifest, and workbooks

04_validation/            validation    exact null arrays and workbooks
04_validation/validation.json
                          validation    raw null + BH + economic intersection

```

Stages 1-4 build the measured and selected surfaces, test them against the null,
and assign final validation verdicts.

The artifact flow is a one-way chain. Shift reads surface artifacts; selection promotes
each top-ranked node's complete shift cube and records its representative bin; validation
reads only those selection artifacts. Validation recomputes the circular-shift null from
the ordered history carried by each selected-node artifact because aggregate cube rates
alone are insufficient.

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
selection and validation read.

The workbook uses a diverging scale centered at zero: red cells mean the barrier is
reached more often than baseline, blue cells mean less often.

## 3. Selection

```bash
barrierlab selection --workspace <name>
```

Selection ranks one *node* per predictor, rather than allowing several decile sheets
from one predictor to consume the shortlist. It uses the workspace's configured target
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

All generated workspace artifacts are git-ignored. `universe.json` and an optional
`plugin.py` are source-controlled per workspace.
