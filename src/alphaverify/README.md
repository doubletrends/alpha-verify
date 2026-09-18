# Source architecture

`src/alphaverify/` is the installable `alphaverify` package: the numerical definitions, the data and artifact adapters, stage orchestration, the human-readable views, and the CLI. Two sub-boundaries have their own READMEs:

- [`domain/`](domain/README.md): measurement, scoring, the synthetic null, and forecast arithmetic.
- [`pipeline/`](pipeline/README.md): the five stages, their artifacts, and how staleness is detected.

`infrastructure/` (workspace loading, persistence) and `presentation/` (workbooks, figures) are small and are covered here.

# How-to guides

### Install and run

```powershell
python -m pip install -e .
alphaverify --help
```

`pyproject.toml` requires `torch>=2.0`. If pip has to install it, PyPI supplies a CPU-only build on Windows. For `--cuda`, first install a CUDA build from the index pytorch.org gives for your CUDA version, then install this package; pip keeps a torch that already satisfies the requirement. `--cuda` also needs a visible device.

Run commands from the repository root. The CLI resolves workspaces as `Path.cwd() / "workspaces" / <name>`. Stage-by-stage usage is in the [pipeline README](pipeline/README.md#how-to-guides).

### Find where a change belongs

| Change | Start here | Also update |
|---|---|---|
| Add a reusable OHLCV indicator | `domain/torch_features.py`, then `domain/features.py` | See the [tutorial](#tutorials) |
| Add an experiment-only feature | Workspace `plugin.py` | [Workspace plugin contract](../../workspaces/README.md#pluginpy) |
| Add or change a data source or cleaning rule | Workspace `data.py` | Rerun from `measure` |
| Change measurement, bin edges, or touch rules | `domain/barrier.py` | Bump `MEASUREMENT_VERSION`; [domain how-to](domain/README.md#how-to-guides) |
| Change the bin score | `domain/scoring.py` | Bump `SCORING_VERSION`; [domain how-to](domain/README.md#how-to-guides) |
| Change the null model | `domain/validation.py` | Bump `NULL_VERSION` in `pipeline/step_03_validation.py` |
| Change a stage artifact, path, or schema | `infrastructure/workspace.py` (paths), `infrastructure/artifact_io.py` (arrays) | [Pipeline how-to](pipeline/README.md#change-a-stage-artifact) |
| Add, rename, or reorder a command | `infrastructure/workspace.py` (`STAGES`), `cli.py` (`COMMANDS`) | [Pipeline how-to](pipeline/README.md#add-a-stage) |
| Change a workbook or figure | `presentation/workbooks.py`, `presentation/bin_figures.py` | Leave JSON and SafeTensors contracts unchanged |
| Change terminal output | `pipeline/reporting.py` | Nothing else. Output text is not a contract |

### Keep a change inside its layer

The layer rules are enforced by `tests/test_architecture_boundaries.py`. When a change needs an upward import, move the shared fact downward instead. For example, `MIN_BIN_N` lives in `domain/notation.py` so that measurement, scoring, and the forecast all read one constant.

### Verify a change

Run `python -m pytest -q` from the repository root after any source change. The suite is offline and runs on CPU, and `pyproject.toml` puts `src/` on its import path. It checks the engineering contracts: that the layer rules hold, that observed and synthetic histories score identically, and that stale artifacts are refused. It does not show that providers are reachable, that a full run fits in memory, or that a result is real. So a change to market data, a numerical kernel, a feature definition, scoring, the null, or a rendered view also needs a real staged run of an affected workspace, followed by a look at its outputs.

# Tutorials

### Add a built-in OHLCV indicator

This walkthrough adds `close_zscore`: how many rolling standard deviations the close sits from its rolling mean. It touches every layer a feature crosses.

**1. Implement the kernel.** Add a branch to `_compute` in `domain/torch_features.py`. Use the cached `mean`/`std` helpers so that synthetic batches share rolling windows with the other features:

```python
if feature == "close_zscore":
    period = params["period"]
    return (close - mean(close, period)) / std(close, period)
```

The input is a `(path, bar)` view, so the real history and every synthetic path are computed in one call.

**2. Register it as an OHLCV feature.** Add `'close_zscore'` to `_TORCH_OHLCV_FEATURES` in `domain/features.py`. This decides how Stage 3 treats it: OHLCV features are **recomputed** on every synthetic path, and all other features are held fixed at their observed values. If you register a price-derived feature anywhere else, it is silently held fixed, and the null then understates chance.

**3. Declare a node.** In `workspaces/nasdaq_daily/universe.json`, add it to a family:

```json
{"id": "close_zscore_20", "family": "ma", "category": "trend",
 "feature": "close_zscore", "params": {"period": 20},
 "data": ["ohlcv"], "derived_from": null}
```

**4. Test.** Run `python -m pytest -q`. Existing tests do not exercise the new branch, so if the kernel has an edge case (here, a flat window divides by zero), add a small case to `tests/test_scoring_parity.py`. That case must check that the observed and null paths agree.

**5. Run the pipeline.**

```powershell
alphaverify measure  --workspace nasdaq_daily
alphaverify compare  --workspace nasdaq_daily
alphaverify validate --workspace nasdaq_daily
alphaverify select   --workspace nasdaq_daily
alphaverify forecast --workspace nasdaq_daily
```

Stage 1 reports `surface skipped close_zscore_20: ...` if the node has fewer than `min_obs` finite values. Validation now covers the new node's bins, and its figures appear in `03_validation/plot/`.

# Explanation

### What this boundary owns

The package owns everything that must mean the same thing in every experiment: how a touch is measured, how a bin is scored, what counts as a synthetic history, how artifacts are stored and fingerprinted, and what order the stages run in. It does not own anything asset-specific. Providers, cleaning, time alignment, the barrier grid, and the condition catalog belong to [`workspaces/`](../../workspaces/README.md). The evidence narrative belongs to the [root README](../../README.md), and the mathematics to [`mathematics.tex`](../../mathematics.tex).

### Why these layers

Dependencies point downward only: `cli` → `pipeline` → `presentation` → `domain`, and `pipeline` → `infrastructure`. Neither `infrastructure` nor `domain` imports another layer.

- **`domain/`** is pure: tensors in, tensors out, with no files, clocks, or CLI. The project's central claim is that an observed history and a synthetic history are scored identically. That only holds if the scoring code cannot see which role it is serving, and keeping I/O out of the domain makes such branches hard to add by accident.
- **`infrastructure/`** imports no other layer. Persistence and workspace loading can then be tested and replaced without numerical code, and it stays the only place that knows file formats.
- **`pipeline/`** is the only layer that joins the others. Sequencing, provenance, and freshness checks sit here, so no calculation or schema needs to know which stage it is serving.
- **`presentation/`** turns stored artifacts into XLSX and PNG files. It is derived: deleting every workbook and figure loses no evidence.

### Why stages persist to disk

Each stage writes its result and fingerprints its inputs. Stage 3 is by far the most expensive stage, and later stages would otherwise have to repeat it. The fingerprints also stop a result from quietly outliving its inputs. How the chain works is described in the [pipeline Explanation](pipeline/README.md#explanation).

### Descriptive names in code, symbols in the math

`domain/notation.py` fixes the OHLCV axis order, the typed measurement and score results, and `MIN_BIN_N`. Arrays that cross a Python API use descriptive ASCII names such as `conditional_probability`, `probability_shift`, and `bin_observation_counts`. Short symbols such as $Z_k$ and $Q_k$ appear only in `mathematics.tex`. When mapping one to the other, use the table in the [domain Reference](domain/README.md#mathematicstex-map).

# Reference

### Entrypoints

| Entrypoint | Notes |
|---|---|
| `alphaverify <command> [--workspace NAME] [--cuda]` | Commands from `cli.py` `COMMANDS`; `--workspace` defaults to `nasdaq_daily`; no command prints the stage overview |
| `infrastructure.workspace.Workspace(name, workspaces_dir=None)` | One workspace's declaration and paths |
| `pipeline.step_0N_*.cmd_*(workspace)` | One stage |

### Sources of truth

| Fact | Owner |
|---|---|
| Stage order and directory names | `infrastructure/workspace.py` `STAGES` |
| Artifact paths | `infrastructure/workspace.py` |
| Array schemas, fingerprints | `infrastructure/artifact_io.py` |
| Prepared-data contract | `infrastructure/market_data.py` `validate_market_data` |
| Layer rules | `tests/test_architecture_boundaries.py` `ALLOWED_LAYER_DEPENDENCIES` |
| Built-in features | `domain/features.py`, `domain/torch_features.py` |
| Minimum bin size | `domain/notation.py` `MIN_BIN_N` |
| Stage rules and settings | `METHOD` in each `pipeline/step_0N_*.py` |
| Display names, colour limits | `presentation/display.py` |
| Asset, grid, horizons, nodes | `workspaces/<name>/universe.json` |

### Version constants

Bump the matching constant whenever you change what a quantity means.

| Constant | Where | Invalidates |
|---|---|---|
| `MEASUREMENT_VERSION` | `domain/barrier.py` | Observed caches (rebuilt automatically), validation |
| `SCORING_VERSION` | `domain/scoring.py` | Validation onward |
| `NULL_VERSION` | `pipeline/step_03_validation.py` | Validation onward |
| `SHIFT_VERSION` | `infrastructure/artifact_io.py` | Stage 2 (rejected on load) |
| `ARTIFACT_SCHEMA_VERSION` | `infrastructure/artifact_io.py` | Observed caches only; Stage 1 and 2 loaders do not check it |

### Limits

- Python ≥ 3.12, PyTorch ≥ 2.0; a CUDA build is installed separately (see [Install and run](#install-and-run)).
- CPU and CUDA draw different nulls for the same seed, and `validation.json` does not record the device.
- `n_bins` ≤ 256 (`bin_assignments` is `uint8`).
- Null batches plan against half of free CUDA memory, or 4 GiB on CPU.
