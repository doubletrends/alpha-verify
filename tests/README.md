# Test contracts

`tests/` protects the repository's fast, deterministic engineering contracts: package boundaries, CLI shape, workspace and artifact schemas, bin-score semantics, validation provenance, and progress output. It does not certify the economic conclusion of a real market-data run or execute the full-size null simulation end to end.

## Run the suite

From the repository root:

```powershell
python -m pytest -q
```

`pyproject.toml` adds `src/` to pytest's import path, so an editable install is not required for the suite. PyTorch and the other project dependencies must still be available. Tests use `unittest` cases plus pytest-compatible test functions.

Run one boundary while developing:

```powershell
python -m pytest -q tests/test_scoring_parity.py
python -m pytest -q tests/test_validation_pipeline.py
```

## What each module owns

| Module | Contract protected |
|---|---|
| `test_architecture_boundaries.py` | Absolute package imports; inward-only domain dependencies; infrastructure ownership of array persistence; flat infrastructure/presentation packages; presentation independence; Stage 3 consumption of Stage 2 artifacts |
| `test_artifact_io.py` | JSON fallback and round-trip behavior; SafeTensors keys and metadata; stable stored/runtime dtypes for surface and shift cubes |
| `test_cli_contract.py` | Supported command set and order; workspace/CUDA option parsing; optional node status; rejection of removed or unsupported flags |
| `test_validation_pipeline.py` | Observed/null role parity through actual pipeline calls despite corrupted presentation arrays; all-bin validation without selection; shared/distinct history ensembles; input fingerprints; missing artifacts; zero-score bins; rank-free plots |
| `test_selection_pipeline.py` | Cleared-only Stage 4 selection, validation fingerprint freshness, and selected-bin heatmap rendering |
| `test_scoring_parity.py` | Full-grid versus streamed score/validity parity; cached versus streamed excursions; per-path baselines; drift regression; quantiles, ties, missing values, thin bins, and fixed external edges |
| `test_notation_contract.py` | Canonical tensor-axis names, OHLCV component order, and version-1 artifact-name normalization |
| `test_stage_reporting.py` | Stable stage headings, summaries, timing shape, and bounded progress milestones |
| `test_workspace_contracts.py` | Nasdaq catalog and stage paths; artifact-history alignment; per-run source caching; BTC hourly workspace-local OHLCV override |

## Test boundaries

The suite is intentionally offline and small:

- Temporary directories contain artifact round trips; tests do not write into real workspace artifact trees.
- Provider calls are replaced with small in-memory frames where data-source behavior matters.
- Numerical fixtures use small deterministic cubes that make score expectations inspectable.
- Architecture tests parse imports and source text to keep dependency rules executable.
- Validation tests run small synthetic ensembles through artifact loading, scoring, provenance, and PNG rendering; they do not run 1,000 full-length paths.

Consequently, a green suite does not prove that Yahoo, CoinMetrics, or the BTC hourly dataset is currently reachable; that a full CPU/CUDA run fits in memory; that generated XLSX/PNG output looks correct; or that a selected market effect is statistically or economically durable.

## Adding tests safely

### Domain calculation

Use the smallest array or DataFrame that exposes the invariant. Assert units, axes, invalid-bin behavior, and exact boundary cases. Scoring changes need matching assertions for observed and null histories, including validity, because both must use the same full-grid statistic.

### Infrastructure or artifacts

Use `TemporaryDirectory` and public persistence helpers. Verify both the stored contract and the loaded runtime representation. If a required key, dtype, filename, or metadata field changes, add migration or explicit rejection behavior rather than silently accepting incompatible artifacts.

### CLI or pipeline

Test parsing and routing without downloading data. Preserve the public order `measure`, `compare`, `validate`, `select`, `status`. Output assertions should protect useful structure and contracts, not incidental whitespace unless the formatting itself is the interface under test.

### Workspace

Keep global behavior in package tests and experiment-specific facts in `universe.json`. Mock remote reads. Add a workspace assertion when a declaration establishes a repository-wide promise—such as a required baseline, stage path, or plugin registration boundary—not for every indicator row.

### Presentation

Prefer assertions about filenames, labels, dimensions, and data handed to the renderer. Visual review of representative generated files is still required; pixel-level snapshot tests would be brittle for the current Matplotlib and workbook outputs.

## Verification beyond pytest

Changes to the scientific path require a staged workspace run:

```powershell
barrierlab measure  --workspace <name>
barrierlab compare  --workspace <name>
barrierlab validate --workspace <name>
barrierlab select   --workspace <name>
barrierlab status   --workspace <name>
```

Then inspect `validation.json`, `selection.json`, representative spreadsheets, null histograms, and selected-bin heatmaps. Generated artifacts are local and ignored by Git; see the [workspace contract](../workspaces/README.md).
