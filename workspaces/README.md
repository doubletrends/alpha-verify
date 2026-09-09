# Workspaces

Each child directory is a versioned experiment declaration and a local artifact namespace. A workspace owns its asset, history start, barrier grid, horizons, condition catalog, and optional extensions. It does not own shared numerical definitions, artifact schemas, renderers, or CLI behavior; those belong to [`src/`](../src/README.md).

## Boundary and source of truth

```text
workspaces/<name>/
  universe.json              versioned experiment declaration
  plugin.py                  optional versioned source/feature registration
  01_surface/                generated measurement artifacts
  02_shift/                  generated baseline-relative artifacts
  03_validation/             generated null results and plots
```

`universe.json` is the source of truth for cross-file experiment facts. `barrierlab.infrastructure.workspace.Workspace` loads it into an immutable runtime configuration and node catalog. Do not duplicate asset symbols, grids, horizons, or feature parameters in package code.

Only `universe.json` and an optional `plugin.py` are versioned. The stage directories contain derived local artifacts and are ignored by Git. They can be regenerated, but later stages depend on the exact history and metadata embedded upstream.

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

The `evaluate` block does not gate Stage 3. It controls the practical-effect summary printed by `barrierlab status <node>`.

### `families`

Each family maps to a list of node declarations. Every node must provide:

- `id`: unique artifact-safe identifier;
- `family`: family label, matching its enclosing family and artifact grouping;
- `category`: descriptive grouping for consumers;
- `feature`: a registered feature name;
- `params`: feature arguments;
- `data`: ordered registered source names;
- `derived_from`: provenance hint or `null`.

Every workspace needs the `_base` family's `baseline` node. Its constant feature produces the unconditional probability surface that Stage 2 subtracts from all conditional nodes.

`NodeCatalog` currently validates the top-level shape, while missing node keys fail when a stage consumes them. Treat the complete node shape above as the authoring contract even where validation is deferred.

## Optional `plugin.py`

A workspace plugin is appropriate when a source or feature is experiment-specific. It must expose:

```python
def register(sources, features) -> None:
    ...
```

`RunContext` creates new registries and loads the plugin once per command. Registrations therefore do not leak between workspaces or runs. A plugin can add or replace a named source and add feature implementations; it should not write stage artifacts or invoke pipeline commands.

Current examples:

- `btc_daily/plugin.py` adds the CoinMetrics community source and BTC on-chain and halving-cycle features.
- `btc_hourly/plugin.py` replaces `ohlcv` with raw hourly bars from the public `mouadja02/bitcoin-technical-indicators-dataset` CSV.
- `nasdaq_daily` needs no plugin; it uses built-in Yahoo and cross-asset sources.

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
`probability_shift_pp`, `bin_observation_counts`, and `bin_edges`. Readers
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

The GitHub media URL in its plugin deliberately dereferences a Git LFS object. Yahoo hourly history is not used because Yahoo exposes only a trailing intraday window, preventing the intended historical rerun.

## Run and inspect a workspace

Run from the repository root and keep the stages in order:

```powershell
barrierlab measure  --workspace nasdaq_daily
barrierlab compare  --workspace nasdaq_daily
barrierlab validate --workspace nasdaq_daily
barrierlab select   --workspace nasdaq_daily
barrierlab status   --workspace nasdaq_daily
```

Every command accepts `--cuda` when CUDA is available through PyTorch. A command reuses data and price excursions in memory only for that command; the next stage reads persisted artifacts.

Inspect one node after `compare`:

```powershell
barrierlab status vix_level --workspace nasdaq_daily
```

If a workbook is open in Excel, a stage may report it as locked while continuing with other artifacts. Close the workbook and rerun that stage. If validation is stale, rerun `validate` after rebuilding Stage 2 when its inputs have changed.

## Add or change a workspace safely

1. Copy the closest existing declaration into a new, clearly named child directory.
2. Set the asset, date range, barrier grid, horizons, bin count in `universe.json`.
3. Keep the baseline node and give every node a unique ID, registered feature, valid parameters, and registered data sources.
4. Add `plugin.py` only for workspace-specific registrations. Keep reusable numerical behavior in `src/barrierlab/`.
5. Run `measure`, `compare`, `validate`, and `select` in order, then inspect workspace and representative node status.
6. Review shift workbooks, null histograms, selected heatmaps, and JSON manifests; a successful command alone does not validate their scientific interpretation.
7. Add or update [tests](../tests/README.md) when the declaration introduces a repository-level source, schema, plugin, or path contract.

Changing history, grid, feature definitions, or node parameters invalidates downstream interpretation even if old artifacts remain readable. Prefer a clean new workspace identity for materially different experiments; otherwise rerun the full pipeline and use the input fingerprint to detect stale validation.
