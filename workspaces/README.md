# Workspaces

Each directory here is one experiment. It holds a versioned declaration (`universe.json`, `data.py`, an optional `plugin.py`) and, once the pipeline has run, a local tree of generated artifacts. A workspace owns everything that is a choice about one market: provider, cleaning, time alignment, history start, barrier grid, horizons, the condition catalog, and experiment-only features. Everything that must mean the same thing across experiments belongs to [`src/`](../src/alphaverify/README.md). `_shared/` is a helper library that workspaces import explicitly. It is not an experiment and not a registry.

# How-to guides

### Run a workspace

See the [pipeline how-to](../src/alphaverify/pipeline/README.md#run-a-workspace). The default workspace is `nasdaq_daily`.

### Add a condition to an existing workspace

Add a node to a family in `universe.json` (see [node fields](#families)). Use a [built-in feature](../src/alphaverify/domain/README.md#lookups) or one registered by this workspace's `plugin.py`. List every source the feature reads in `data`. Then rerun from `measure`.

### Add or change a data source

1. In `data.py`, fetch the feed inside the loader's `feed(name)`, snapshot the raw frame with `_shared.snapshots.snapshot`, and clean and align it.
2. Accept the new source name in `load(sources)`.
3. Record anything a reader would need to reproduce the panel in `panel.attrs["provenance"]`.
4. Bump the loader's `CLEANING_VERSION` if prepared rows change.
5. Rerun from `measure`. Existing artifacts are not invalidated automatically (see [freshness checks](../src/alphaverify/pipeline/README.md#freshness-checks)).

### Add an experiment-only feature

In `plugin.py`, implement `register(features)` and call `features.register_torch(name, fn)`. Use only the [host API](#pluginpy). Plugin features stay fixed in the synthetic null. If a feature can be computed from OHLCV alone and should be tested against resampled prices, make it a built-in instead ([tutorial](../src/alphaverify/README.md#tutorials)).

### Clean up a workspace

Everything except `universe.json`, `data.py`, and `plugin.py` is generated and ignored by Git. Deleting a stage directory forces that stage and everything after it to run again. Deleting `00_data/` forces a fresh download. Deleting `00_cache/` rebuilds observed outcomes on the next `measure` or `validate`. Directories from an earlier stage numbering (`05_summary/`, `06_forecast/`) are never read and can be deleted.

### Test a workspace change

Mock the provider and assert cleaning behaviour in `tests/test_workspace_data.py` when you introduce a new cleaning rule or a new repository-wide promise, such as a required baseline or a plugin boundary. Do not add a test for every node.

# Tutorials

### Create a new daily workspace

This walkthrough creates `spx_daily`, an S&P 500 copy of `nasdaq_daily`.

**1. Copy the declaration.**

```powershell
New-Item -ItemType Directory workspaces/spx_daily
Copy-Item workspaces/nasdaq_daily/universe.json, workspaces/nasdaq_daily/data.py workspaces/spx_daily/
```

**2. Edit `universe.json` `meta`.** Set `"workspace": "spx_daily"` and `"asset": {"provider": "yfinance", "ticker": "^GSPC", "interval": "1d"}`, and update `description`. Keep the grid, the horizons, and `n_bins` unless you have a reason, decided **before** you look at any result, to change them.

**3. Check `data.py`.** The NASDAQ loader checks only `provider` and `interval`, so it works unchanged for another Yahoo daily ticker. Rename `CLEANING_VERSION` (for example to `spx-daily-v1`) so that provenance tells the two workspaces apart.

**4. Keep the baseline.** `families._base` must still contain the `baseline` node with the `constant` feature.

**5. Run.**

```powershell
alphaverify measure  --workspace spx_daily
alphaverify compare  --workspace spx_daily
alphaverify validate --workspace spx_daily
alphaverify select   --workspace spx_daily
alphaverify forecast --workspace spx_daily
```

**6. Inspect.** Read the `measure` warnings: every skipped node is listed. Open a few `02_shift/spreadsheet/*.xlsx` and `03_validation/plot/*.png` files, then check `validation.json` `summary` and `selection.json` `summary.expected_by_chance`. A command that succeeds says nothing about whether its result is scientifically meaningful.

# Explanation

### Why data preparation lives here

When a daily bar counts as known, whether a VIX close may be joined to a NASDAQ session, and what to do with a malformed hourly bar are experimental assumptions, not engineering details. If the core made those choices, they would apply silently to every market. Instead the core only *validates* a prepared panel (see [contract](#prepared-data-contract)) and never repairs one. Each loader states its policy in code and in `provenance`.

### Point-in-time caveat

The shipped daily loaders use the source's date labels and join auxiliary closes on the same date, forward-filling and never backfilling. This is an end-of-day research convention. It is not evidence that every feed was available at that bar's close, and downloads can be revised later. An experiment that needs release-time guarantees must implement availability timestamps and lags in its `data.py`.

### Why `universe.json` is the source of truth

The asset, grid, horizons, and node parameters are read from one file into an immutable `WorkspaceConfig`. Package code carries no defaults, so a missing field fails when the workspace is opened. It cannot fall back to a value from some other experiment.

### Notes on the shipped workspaces

Each workspace's `universe.json` `meta` and the docstring of its `data.py` are the authoritative description of that workspace. A few facts they do not record:

- The figures in the [root README](../README.md) come from an earlier `nasdaq_daily` pipeline (selection first, 10,000 null histories) and have not been regenerated.
- The `btc_hourly` grid (±3% in 0.25% steps) was matched to hourly volatility and fixed before any hourly validation result existed.
- `nasdaq_daily` does not exclude a still-forming session bar; both BTC loaders exclude theirs.

### Why artifacts are not portable

A stage artifact is meaningful only together with the exact history, grid, and declaration that produced it. Copying one between workspaces can produce files that load but contradict each other. For a materially different experiment, create a new workspace rather than editing one that has results you want to keep.

# Reference

### `universe.json`

Loaded by `infrastructure/workspace.py`. Every `meta` field below is required, and nothing has a default.

| `meta` field | Meaning |
|---|---|
| `asset` | `provider`, `ticker`, `interval` (ending in `h` → hourly horizons) |
| `start_date`, `min_obs` | First requested bar; minimum finite feature values per node |
| `barriers` | Signed fractions `min`, `max`, `step`; keep symmetric |
| `horizons` | Forward bars `min`, `max` |
| `n_bins` | Quantile bins per feature (≤ 256) |
| `evaluate` | `min_dev`, `min_bin_n`, `min_run` for the forecast's terminal line only |

#### `families`

Family name → list of nodes, each with `id`, `family`, `category`, `feature`, `params`, `data` (source names, `ohlcv` first), and `derived_from`. `_base` must hold node `baseline` with feature `constant`.

### `data.py`

```python
def create_loader(*, start, asset, cache_dir):   # cache_dir = 00_data/
    return load                                   # load(sources: list[str]) -> DataFrame
```

### Prepared-data contract

Enforced, never repaired, by `infrastructure/market_data.py` `validate_market_data`.

### `plugin.py`

`register(features)` may use only this host API:

| Entry point | Use |
|---|---|
| `features.register_torch(name, fn)` | `fn(data, params)` returns a 1-D tensor aligned with `data.index` |
| `alphaverify.domain.tensor_runtime.tensor(values)` | Float64 tensor on the run's device |
| `alphaverify.domain.torch_features.rolling_mean(values, window)` | Rolling mean over the last axis |
