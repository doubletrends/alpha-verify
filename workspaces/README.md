# Workspaces

Each experiment directory is a versioned declaration and a local artifact namespace. A workspace owns its providers, cleaning, alignment, asset, history start, barrier grid, horizons, condition catalog, and optional features. Shared numerical definitions, input validation, artifact schemas, renderers, and CLI behavior belong to [`src/`](../src/README.md). `_shared/` contains optional download and snapshot helpers, not an experiment or automatic source registry.

## Boundary and source of truth

```text
workspaces/<name>/
  universe.json              versioned experiment declaration
  data.py                    required source loading, cleaning, and alignment
  plugin.py                  optional feature registration
  00_data/                   local source-frame snapshots and provider caches
  01_surface/                generated measurement artifacts
  02_shift/                  generated baseline-relative artifacts
  03_validation/             generated null results and plots
```

`universe.json` is the source of truth for cross-file experiment facts. `alphaverify.infrastructure.workspace.Workspace` loads it into an immutable runtime configuration and node catalog. Do not duplicate asset symbols, grids, horizons, or feature parameters in package code.

`universe.json`, `data.py`, optional `plugin.py`, and shared Python helpers are versioned. Download snapshots, caches, and stage outputs are local and ignored by Git. Later stages depend on the exact history and metadata embedded upstream.

## `universe.json` contract

The document contains `meta` and `families` objects.

### `meta`

| Field | Meaning |
|---|---|
| `workspace` | Declaration identity; keep it equal to the directory name |
| `asset` | Provider label, ticker, and bar interval |
| `start_date` | Earliest requested observation |
| `min_obs` | Minimum valid feature observations required for a node |
| `barriers` | Inclusive signed barrier grid: `min`, `max`, and `step` |
| `horizons` | Inclusive forward-bar range: `min` and `max` |
| `n_bins` | Quantile-bin count for conditional features |
| `evaluate` | `min_dev`, `min_bin_n`, and `min_run` used by detailed node status inspection |
| `target` | Parsed target barrier/horizon available to Python callers; the current CLI validation stage scores the complete grid rather than this single target |

The `evaluate` block does not gate Stage 3. It controls the practical-effect summary printed by `alphaverify status <node>`.

### `families`

Each family maps to a list of node declarations. Every node must provide:

- `id`: unique artifact-safe identifier;
- `family`: family label, matching its enclosing family and artifact grouping;
- `category`: descriptive grouping for consumers;
- `feature`: a registered feature name;
- `params`: feature arguments;
- `data`: ordered source names understood by this workspace's `data.py`;
- `derived_from`: provenance hint or `null`.

Every workspace needs the `_base` family's `baseline` node. Its constant feature produces the unconditional probability surface that Stage 2 subtracts from all conditional nodes.

`NodeCatalog` currently validates the top-level shape, while missing node keys fail when a stage consumes them. Treat the complete node shape above as the authoring contract even where validation is deferred.

## Required `data.py`

A workspace provides this factory, returning a callable that prepares a complete panel for an ordered list of sources:

```python
def create_loader(*, start, asset, cache_dir):
    # Return your workspace's callable: load(sources) -> pandas.DataFrame.
    return DailyInputs(start=start, asset=asset, cache_dir=cache_dir).load
```

`DailyInputs` above represents your own implementation; the existing workspaces use closures with the same interface. `start` and `asset` come from `universe.json`; `cache_dir` is the workspace's `00_data/`. The factory runs once per measurement context and owns raw-feed caching across panels. The core caches completed panels and does not provide fallback sources. Missing `data.py` fails when input data is requested, so reading or validating existing artifacts remains offline.

Prepared frames must have unique increasing nonmissing `DatetimeIndex` labels, a documented timezone-naive time basis, and unique real numeric columns. Required `open`, `high`, `low`, `close`, and `volume` must be finite; prices must be positive, volume nonnegative, and `low <= open/close <= high`. Auxiliary columns can retain NaN. The core rejects violations without repairing data. Gaps between bars are allowed: horizons count subsequent observations, not elapsed wall-clock intervals.

The workspace decides how to handle sessions, time zones, duplicate dates, missing values, adjustments, and auxiliary releases. The shipped daily loaders retain exchange/source daily labels and same-date auxiliary alignment, forward-filling across missing dates without backfilling. This preserves the former end-of-day research assumption; daily labels and revised downloads are not proof of point-in-time availability. Experiments needing release-time guarantees must implement availability timestamps and lags here.

All three loaders snapshot the selected source frames before cleaning as content-addressed CSVs under `00_data/`. These are decoded source-frame snapshots, not exact HTTP payload archives. Stage 1 stores `data_provenance` containing cleaning version, time basis, alignment, source identifiers, snapshot hashes, and raw/prepared row counts. Source snapshots are audit inputs; loaders currently fetch again on a new run rather than offering automatic snapshot replay.

The daily loaders drop incomplete OHLCV rows, keep the last duplicate, and reject invalid OHLC ordering through core validation. BTC hourly converts timestamps to UTC, keeps the last duplicate, drops incomplete bars, and never resamples or fills missing bars. These explicit policies can change a remeasurement of previously malformed data. Earlier stored artifacts with missing prices are now rejected instead of silently losing rows. Rebuild measurement and downstream stages to apply new cleaning; changing `data.py` alone does not rewrite or invalidate existing stored results.

## Optional `plugin.py`

Plugins now expose `register(features)` and own custom feature registrations only. Move former `register(sources, features)` source logic into `data.py`. Each run gets a fresh feature registry; a plugin should not write stage artifacts or invoke pipeline commands.

Current examples:

- `btc_daily/data.py` owns Yahoo and CoinMetrics inputs; `plugin.py` registers on-chain and halving-cycle features.
- `btc_hourly/data.py` reads raw hourly bars from the public `mouadja02/bitcoin-technical-indicators-dataset` CSV; no feature plugin is needed.
- `nasdaq_daily/data.py` explicitly declares Yahoo price and cross-asset feeds and their cleaning policy.
- `_shared/yahoo.py` provides transport only; workspace modules explicitly select it and handle cleaning themselves.

## Artifact lifecycle

| Stage | Command | Machine-readable contract | Human-readable views |
|---|---|---|---|
| `00_cache` | internal | Per-history excursions, touch matrix, and baseline | None |
| `01_surface` | `measure` | Per-node SafeTensors probability cube and embedded ordered history | Per-node XLSX workbook |
| `02_shift` | `compare` | Per-node shift tensor plus Stage 1 references/fingerprints | Per-node XLSX workbook |
| `03_validation` | `validate` | `validation.json` with fingerprint, observed scores, null scores, p95, and raw p-values | Per-bin null-histogram PNG |
| `04_selection` | `select` | `selection.json` containing every validation-cleared bin | Per-selected-bin shift heatmap and copied null-distribution PNG |

Stage 3 uses Stage 2 as its completion gate and reads histories and observed condition data from the referenced Stage 1 artifacts, with no provider calls. Observed excursions, touches, baselines, and bin assignments come from the versioned source cache. Synthetic batches generate one touch matrix per horizon and share it across every node. `validation.json` fingerprints both Stage 1 and Stage 2 source bytes, node declarations, bin count, measurement version, and simulation settings.

Array artifacts use schema version 2 and descriptive ASCII field names such as
`barriers`, `conditional_probability`, `baseline_probability`,
`probability_shift`, `bin_observation_counts`, and `bin_edges`. Readers
normalize version-1 names, including `Δs`, so existing generated workspaces can
still be consumed and regenerated.

Do not copy generated artifacts between workspaces. Paths may look compatible while grids, histories, features, or fingerprints disagree.

## Current declarations

### `nasdaq_daily`

The flagship Alpha Verifier experiment. It requests Nasdaq Composite (`^IXIC`) daily bars from 2015, a −20% to +20% barrier grid in 1% steps, horizons from 1 to 30 days, and ten condition bins. Its catalog combines price/volume indicators with VIX, Treasury-yield, DXY, and calendar conditions.

The historical result in the [root README](../README.md) comes from this workspace's selected top 20 condition bins and 10,000-history validation run.

### `btc_daily`

A daily BTC (`BTC-USD`) comparison workspace from 2015 with a −20% to +20% grid and 1- to 30-day horizons. Its plugin extends the built-in price features with CoinMetrics on-chain histories and halving-cycle conditions. Those plugin-defined conditions are held fixed during the current synthetic-OHLC null because they cannot be reconstructed from OHLC alone.

### `btc_hourly`

An exploratory hourly BTC workspace from 2018 with a −10% to +10% grid and 1- to 48-hour horizons. It uses the publisher's raw OHLCV columns and recomputes indicators locally; it does not trust precomputed indicator columns from the source dataset.

The GitHub media URL in its `data.py` deliberately dereferences a Git LFS object. The workspace explicitly chooses this historical dataset rather than Yahoo hourly history.

## Run and inspect a workspace

Run from the repository root and keep the stages in order:

```powershell
alphaverify measure  --workspace nasdaq_daily
alphaverify compare  --workspace nasdaq_daily
alphaverify validate --workspace nasdaq_daily
alphaverify select   --workspace nasdaq_daily
alphaverify status   --workspace nasdaq_daily
```

Every command accepts `--cuda` when CUDA is available through PyTorch. A command reuses data and price excursions in memory only for that command; the next stage reads persisted artifacts.

Inspect one node after `compare`:

```powershell
alphaverify status vix_level --workspace nasdaq_daily
```

If a workbook is open in Excel, a stage may report it as locked while continuing with other artifacts. Close the workbook and rerun that stage. If validation is stale, rerun `validate` after rebuilding Stage 2 when its inputs have changed.

## Add or change a workspace safely

1. Copy the closest existing declaration into a new, clearly named child directory.
2. Set the asset, date range, barrier grid, horizons, bin count in `universe.json`.
3. Keep the baseline node and give every node a unique ID, registered feature, valid parameters, and declared data sources.
4. Implement `data.py`, including source selection, cleaning, alignment, and provenance. Add `plugin.py` only for custom features. Keep reusable numerical behavior in `src/alphaverify/`.
5. Run `measure`, `compare`, `validate`, and `select` in order, then inspect workspace and representative node status.
6. Review shift workbooks, null histograms, selected heatmaps, and JSON manifests; a successful command alone does not validate their scientific interpretation.
7. Add or update [tests](../tests/README.md) when the declaration introduces a repository-level source, schema, plugin, or path contract.

Changing history, grid, feature definitions, or node parameters invalidates downstream interpretation even if old artifacts remain readable. Prefer a clean new workspace identity for materially different experiments; otherwise rerun the full pipeline and use the input fingerprint to detect stale validation.

### Probability-difference shift units

Stage 2 writes `probability_shift` with `shift_unit: probability_difference` and
`shift_version: probability-difference-v1`. This is a shift-specific version;
Stage 1 arrays and observed caches remain reusable. Recognized legacy `shift`
and `probability_shift_pp` arrays are converted from percentage points on load.
Unknown or conflicting unit declarations are rejected.

Workspace `evaluate.min_dev` uses fractions when `evaluate.shift_unit` is
`probability_difference`: 0.10 means a 10-percentage-point effect. Older declarations
without `shift_unit` retain their percentage-point interpretation (10 means 0.10).
New declarations should always specify the unit. These thresholds govern status
inspection, not the Stage 4 statistical selection rule.

Run `compare`, `validate`, and `select` to regenerate derived artifacts and views
in the new units. Old validation and selection summaries are stale under the new
scoring version. No Stage 1 remeasurement is required solely for this unit change.
