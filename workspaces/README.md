# Workspaces

Each directory here is a versioned experiment declaration plus a local artifact namespace. A workspace owns its asset-specific configuration and node catalog; it does **not** own the shared numerical engine, artifact schema, or presentation implementation.

`universe.json` is the source of truth. `barrierlab.infrastructure.workspace.Workspace` validates and reads it once for a run. Keep cross-file facts there rather than copying asset, grid, horizon, or threshold values into Python code or documentation.

## Contents and lifecycle

```text
workspaces/<name>/
  universe.json              source-controlled declaration
  plugin.py                  optional source/feature registrations
  01_surface/ … 04_validation/ generated, local pipeline artifacts
```

The standard stages produce a one-way artifact chain. Generated files are ignored by Git.

## Contract for `universe.json`

`meta` must declare:

- `asset`: provider, ticker, and interval;
- `start_date`, `min_obs`, `horizons`, and `Δ` (or `Delta`);
- `n_bins` for conditional features;
- `evaluate.min_dev`, `evaluate.min_bin_n`, and `evaluate.min_run` for practical significance;
- `validation.null_alpha`; and
- `target.Δ` and `target.horizon`, shared by selection and validation.

`families` maps a family name to nodes. Every node needs `id`, `family`, `feature`, `params`, and the `data` source names it requires. The family name must match the artifact subdirectory used by the pipeline. Include the `_base/baseline` constant node: it is the unconditional probability surface every conditional result is compared with.

## Add a workspace safely

1. Copy an existing declaration, choose a new workspace name, and set an asset-appropriate daily barrier grid and horizon range.
2. Declare only source names registered by the built-in source registry or by this workspace’s `plugin.py`.
3. Add a plugin only for workspace-specific sources or features. It must define `register(sources, features)`; registration is per run, not process-global.
4. Run `barrierlab surface --workspace <name>` and then the remaining stages in order. Use `barrierlab status --workspace <name>` to inspect the artifact funnel before relying on later-stage output.
5. Add or update tests if the new workspace establishes a contract beyond its own declaration.

Do not copy generated array, workbook, or plot artifacts between workspaces: artifact history, validation fingerprints, and asset configuration must agree. See [the package boundary](../src/barrierlab/README.md) for stage ownership and [the pipeline protocol](../src/barrierlab/pipeline/README.md) for the full artifact contract.

## Current declarations

- `nasdaq_daily` is the reproducible daily NASDAQ Composite flagship (`^IXIC`, from 2015-01-01).
- `btc_daily` is the daily BTC comparison workspace (`BTC-USD`, from 2015-01-01). Its plugin supplies CoinMetrics on-chain data and BTC cycle features.
- `btc_hourly` is an exploratory BTC workspace with +1h through +48h horizons. It loads raw OHLCV from the public `mouadja02/bitcoin-technical-indicators-dataset` CSV and recomputes every feature locally.

Hourly Yahoo workspaces remain out of scope: Yahoo exposes only a trailing hourly window, so historical reruns are not reproducible. `btc_hourly` avoids that limit with its workspace-local source.
