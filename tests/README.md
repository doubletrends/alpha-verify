# Test contracts

`tests/` protects the repository's fast, deterministic engineering contracts: package boundaries, CLI shape, workspace and artifact schemas, bin-score semantics, and validation provenance. It does not certify the economic conclusion of a real market-data run or execute the full-size null simulation end to end.

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
| `test_architecture_boundaries.py` | Provider-free core; declared layer dependencies (cli → pipeline → presentation → domain, with infrastructure and domain importing no other layer); an acyclic module graph; stages importing only earlier stages; no persistence dependencies in the domain; commands matching the stage order |
| `test_workspace_data.py` | Prepared-data rejection without mutation; missing auxiliary values kept; workspace cleaning/alignment without backfill; panel isolation; excluding forming bars; per-run feed caching (regression for repeated downloads); market-history cache keys; measurement provenance; an offline five-stage run |
| `test_scoring_parity.py` | Full-grid versus streamed score/validity parity; cached versus uncached measurement; per-path baselines and edges; batched versus independent scoring; drift regression; valid, reproducible null ensembles; median bin edges and ties; the bin score against hand-worked weights and a pair-by-pair reference |
| `test_validation_pipeline.py` | Observed/null role parity through actual pipeline calls despite corrupted stored probabilities, counts, and shifts; all-bin validation without selection; the Monte Carlo p-value; separate nulls for distinct histories; staleness when method, settings, catalog, or artifact bytes change; missing artifacts; zero-score bins; rejection of shift artifacts in other units |
| `test_selection_pipeline.py` | Cleared-only Stage 4 selection, evidence-only ranking with shared ranks for equal p, q-values across all tested bins, validation freshness; Benjamini–Hochberg q-values against a hand-worked example and the step-up definition, including ties, the Monte Carlo p floor, and empty input |
| `test_forecast_pipeline.py` | Stage 5 groups cleared bins by node and orders nodes by their strongest bin; uses one active bin per family; matches naive Bayes to stored Stage 1 counts and the joint rate to an independent Stage 1 measurement; falls back to the baseline with no cleared bins; requires a current selection; naive Bayes log-odds for two conditions and thin cells; stored-edge bin membership with the Stage 1 tie convention |

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

Test parsing and routing without downloading data. Preserve the public order `measure`, `compare`, `validate`, `select`, `forecast`. Output assertions should protect useful structure and contracts, not incidental whitespace unless the formatting itself is the interface under test.

### Workspace

Keep global behavior in package tests and experiment-specific facts in `universe.json`. Mock remote reads. Add a workspace assertion when a declaration establishes a repository-wide promise—such as a required baseline, stage path, or plugin registration boundary—not for every indicator row.

### Presentation

Prefer assertions about filenames, labels, dimensions, and data handed to the renderer. Visual review of representative generated files is still required; pixel-level snapshot tests would be brittle for the current Matplotlib and workbook outputs.

## Verification beyond pytest

Changes to the scientific path require a staged workspace run:

```powershell
alphaverify measure   --workspace <name>
alphaverify compare   --workspace <name>
alphaverify validate  --workspace <name>
alphaverify select    --workspace <name>
alphaverify forecast  --workspace <name>
```

Then inspect `validation.json`, `selection.json`, representative spreadsheets, and bin figures. Generated artifacts are local and ignored by Git; see the [workspace contract](../workspaces/README.md).
