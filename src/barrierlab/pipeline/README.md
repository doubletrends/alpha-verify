# Pipeline protocol

`pipeline/` owns BarrierLab's four-stage artifact workflow. Run stages through the
`barrierlab` CLI; [`../cli.py`](../cli.py) is the public entrypoint and
`infrastructure/artifacts.py` owns path construction.

```text
measure   → 01_surface/     full conditional probability cubes and workbooks
compare   → 02_shift/       baseline-subtracted cubes and workbooks
select    → 03_selection/   ranked selected-node cubes, views, and manifest
validate  → 04_validation/  simulated-null summary and histograms
```

The measured quantity is:

```text
P(price touches Δ within t bars | condition bin)
```

## 0. Orient

```bash
barrierlab status --workspace <name>
barrierlab status <node> --workspace <name>
```

`status` derives progress from artifacts on disk. It reports cubes and readable views
for the first three stages, then the current Stage 4 summary and cleared selected bins.

## 1. Surface

```bash
barrierlab measure --workspace <name>
```

This produces one raw probability cube per node. The baseline node is a constant feature
with one bin and supplies the unconditional reference rate. Artifacts:

```text
01_surface/array/<node>.safetensors
01_surface/spreadsheet/<node>.xlsx
```

## 2. Shift

```bash
barrierlab compare --workspace <name>
```

For every condition bin, Stage 2 writes percentage-point deviation from the baseline:

```text
100 * (P(touch Δ within t | bin) - P(touch Δ within t))
```

Artifacts:

```text
02_shift/array/<node>.safetensors
02_shift/spreadsheet/<node>.xlsx
```

## 3. Selection

```bash
barrierlab select --workspace <name>
```

Selection ranks candidate condition bins at the workspace target by two-sided skew:

```text
score = abs(shift(+Δ, bin) - shift(-Δ, bin))
```

It retains the configured top rows, records them in `selection.json`, and copies each
selected node's complete Stage 2 cube. The selected workbook is a representative-bin
view; the Safetensor retains every bin.

```text
03_selection/selection.json
03_selection/array/rank_*.safetensors
03_selection/spreadsheet/rank_*.xlsx
03_selection/plot/*.png
```

## 4. Validation

```bash
barrierlab validate --workspace <name>
```

Stage 4 tests each selected condition-bin score against 1,000 shared synthetic OHLC
histories. The synthetic ensemble is fitted once to the selected history; market-derived
features are recomputed on each path, while external condition histories remain fixed.
For each selected bin the summary records its observed two-sided skew, the synthetic
null p-value and 95th percentile, then applies Benjamini-Hochberg correction across the
selected sweep. A bin is cleared only when it survives BH and its raw null p-value is at
most the workspace's `validation.null_alpha`.

Stage 4 has one machine-readable artifact and readable histogram plots:

```text
04_validation/validation.json
04_validation/plot/*.png
```

The JSON includes `tests[]` for every selected bin and `cleared[]` for the bins that pass
both statistical gates. It contains no per-node validation array or workbook artifacts.

## Boundaries

`infrastructure/workspace.py` owns workspace configuration and node traversal;
`workspace_plugins.py` loads per-run plugin registrations; `artifacts.py` owns paths and
history reconstruction; `artifact_io.py` owns JSON and SafeTensor persistence. Domain
modules own numerical work and `presentation/` owns workbook/plot rendering.
