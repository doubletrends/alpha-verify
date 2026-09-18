# Pipeline

`pipeline/` runs the five stages. It decides what each stage reads and writes, records provenance, detects stale inputs, and keeps one failing node from stopping the others. It defines no formulas (see [`domain/`](../domain/README.md)) and no file formats or paths (see `infrastructure/artifact_io.py` and `infrastructure/workspace.py`).

# How-to guides

### Run a workspace

From the repository root, in order:

```powershell
alphaverify measure  --workspace nasdaq_daily   # 01_surface, 00_cache
alphaverify compare  --workspace nasdaq_daily   # 02_shift
alphaverify validate --workspace nasdaq_daily   # 03_validation
alphaverify select   --workspace nasdaq_daily   # 04_selection
alphaverify forecast --workspace nasdaq_daily   # 05_forecast
```

Add `--cuda` to any of them. Only `measure` loads provider data. Later stages read stored artifacts, and `forecast` is only as recent as the last `measure`.

### Decide what to rerun

Start from the earliest stage whose inputs changed and run every stage after it. Rerunning `measure` rewrites Stage 1 bytes, which makes everything downstream stale.

| What changed | Rerun from |
|---|---|
| New market data wanted; `data.py`; anything in `universe.json` `meta` except `evaluate`; a node's `feature`/`params`/`data`; a feature kernel; `barrier.py` | `measure` |
| `domain/shift.py` `from_cube` | `compare` |
| `scoring.py`, `validation.py`, Stage 3 constants | `validate` |
| Stage 4 rule or ranking | `select` |
| Stage 5 rule, `combination.py`, `universe.json` `evaluate` | `forecast` |
| A Stage 1 or 2 workbook view | `measure` or `compare`. Workbooks cannot be regenerated on their own |
| A bin figure | `validate` (Stage 3 figures) or `select` (Stage 4 figures) |

Not every change is detected automatically. See [what freshness checks cover](#freshness-checks).

### Fix a stage that refuses to run or skips nodes

| Message | Meaning | Fix |
|---|---|---|
| `--cuda requested but no CUDA device is available` | Torch cannot see a GPU | Install a CUDA build of PyTorch or drop `--cuda` |
| `surface skipped <node>: only N valid observations` | Fewer than `min_obs` finite feature values | Lower `min_obs`, shorten the feature's window, or start earlier |
| `surface skipped <node>: '<param>'` | The node lacks a parameter its feature reads | Fix `params` in `universe.json` |
| `surface skipped <node>: workspace ...` | `data.py` returned data that fails the [prepared-data contract](../../../workspaces/README.md#prepared-data-contract) | Fix cleaning in `data.py` |
| `no baseline surface array available; run measure first` | The `baseline` node has no Stage 1 array | Run `measure`, then check its warnings for the baseline |
| `Stage 1 source changed for <node>; rerun compare` | Stage 1 was rewritten after Stage 2 | `compare`, then everything after it |
| `unknown shift units/version; rerun compare` | The Stage 2 artifact predates the current `SHIFT_VERSION` | `compare` |
| `incomplete: N nodes lack shift arrays` | Declared nodes have no Stage 2 artifact. The result is written with `complete: false`, which selection rejects | Fix the Stage 1 warnings, then `measure` and `compare` |
| `validation is missing, incomplete, or stale; run validate first` | Fingerprint or `METHOD` mismatch, or `complete: false` | `validate` |
| `selection is missing or stale; run select first` | The validation bytes changed, or the selection is stale in turn | `select` |
| `... changed during validation/selection/forecast` | An input was rewritten while the stage was running | Rerun that stage alone |
| `workbook locked for <node>; close it in Excel and re-run` | Windows file lock | Close the workbook and rerun that stage |

### Change a stage artifact

1. Paths go in `infrastructure/workspace.py`, and array fields in `infrastructure/artifact_io.py`. JSON summaries are built inline in each `step_0N_*.py`.
2. Update every reader downstream. Stage 2 readers go through `context.materialized_shift`, which merges the Stage 1 arrays with the Stage 2 shift.
3. If a field's meaning changes, make old artifacts fail loudly, either by bumping a [version constant](../README.md#version-constants) or by adding a key to the freshness check. Do not accept them silently.
4. Update `tests/conftest.py` `selected_workspace` if the Stage 3 summary shape changes.

### Add a stage

1. Append its name to `STAGES` in `infrastructure/workspace.py`. The directory number comes from its position.
2. Add `step_0N_<name>.py` with a `cmd_<name>(ws)` entry. It may import only earlier stages (`test_stages_import_only_earlier_stages`).
3. Add a `Command` to `cli.py` `COMMANDS` in the same position (`test_stage_commands_mirror_the_stage_order`).
4. Add a heading to `StageReport._HEADINGS` in `reporting.py`.
5. Fingerprint the upstream bytes the stage consumes, and reject a stale upstream the way `selection_summary_is_current` does.

# Explanation

### A chain of fingerprints

Each stage pins the exact bytes it consumed:

```text
01_surface ──sha256──► 02_shift ──sha256 (both)──► validation.json ──sha256──► selection.json ──sha256──► forecast.json
```

A later stage refuses stale input instead of rebuilding it, because rebuilding could mean re-downloading data or rerunning the null without anyone noticing. Each summary also records a `METHOD` dict. A result produced under a different rule is stale even when its inputs have not changed. Before writing, Stages 3–5 re-hash their input and abort if it changed during the run.

### Freshness checks

What the chain detects:

- A Stage 1 array rewritten after `compare` (Stage 2 records `source_sha256` and `baseline_sha256`).
- Any Stage 1 or 2 byte change, any change to a node declaration, `n_bins`, or anything in Stage 3's `METHOD` (validation).
- Any byte change in `validation.json` (selection) or `selection.json` (forecast).

What it does not detect:

- A `data.py` change, new provider data, or a change to the grid or horizons in `universe.json`, until `measure` runs again. The stored cubes carry their own axes, and later stages use those axes.
- A code change that alters results but leaves every version constant alone. That is why [version constants](../README.md#version-constants) exist.
- The Torch device used by `validate` (see [limits](../README.md#limits)).

### Failure isolation

Stages 1 and 2 catch an exception per node, print a warning to stderr, and move on, so one bad feature does not cost a full run. Stage 3 is all-or-nothing about completeness: a missing node yields `complete: false`, which Stage 4 rejects, because a selection drawn from part of the catalog would understate how many tests were run.

### Why validation groups nodes by market history

Nodes whose stored histories are identical share one synthetic ensemble, and the ensemble is drawn once per group (`market_history_key`). Groups are processed one at a time, so only one ensemble sits in memory. A history's null is never applied to a different history. The shipped loaders give every node the same OHLCV rows, so each of them forms one group, but the pipeline does not assume this: a loader may return different rows for different source lists.

### Why selection only ranks

Stage 4 keeps exactly the bins Stage 3 cleared. Ranking by raw p, plus q-values and the count expected by chance, is information for the reader, not a second filter. Adding a filter would change the experiment. The ranking uses evidence alone: equal p-values share a rank, and the observed effect size never breaks a tie.

### Why the forecast uses one bin per family

Nodes in the same family (for example, RSI at two periods) measure nearly the same thing. Counting both in a naive Bayes sum would count the same evidence twice. The best-ranked active bin in each family is used; the others are listed with the reason they were skipped. Bins whose stored history differs from the baseline's are also skipped, because their counts would not line up bar for bar.

# Reference

### Stages

| # | Command | Module | Writes | Refuses when |
|---|---|---|---|---|
| 1 | `measure` | `step_01_surface.py` | `00_cache/`, `01_surface/` | never; skips failing nodes |
| 2 | `compare` | `step_02_shift.py` | `02_shift/` | baseline array missing |
| 3 | `validate` | `step_03_validation.py` | `03_validation/` | never; writes `complete: false` if nodes are missing |
| 4 | `select` | `step_04_selection.py` | `04_selection/` | `validation_summary_is_current` is false |
| 5 | `forecast` | `step_05_forecast.py` | `05_forecast/` | `selection_summary_is_current` is false |

### Where artifact contracts live

| Contract | Owner |
|---|---|
| Paths | `infrastructure/workspace.py` |
| Stage 1, cache, and Stage 2 array fields | `infrastructure/artifact_io.py` (`save_surface`, `save_observed_cache`, `save_shift`) |
| JSON summary keys | The `summary`/`forecast` dict at the end of each `step_0N_*.py` |
| Rule and setting records | `METHOD` in `step_03`–`step_05` |
