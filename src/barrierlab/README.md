# BarrierLab package

This package turns one workspace declaration into staged, inspectable barrier-touch artifacts. It owns implementation; it does **not** own the experiment definition. A workspace’s `universe.json` is the source of truth for the asset, barrier grid, horizons, nodes, and thresholds.

## Public entrypoints

- `barrierlab.cli:main` is the console entrypoint installed as `barrierlab`.
- `barrierlab <surface|shift|selection|validation>` runs one named stage.
- `barrierlab status [NODE] --workspace NAME` is read-only and reports artifact/validation state.
- `infrastructure.workspace.Workspace` is the runtime view of a workspace declaration and its artifact namespace.

The CLI command registry in [`cli.py`](cli.py) is the source of truth for command names and stage ordering. Keep it consistent with `infrastructure.artifacts.STAGE_DIRECTORIES`; `tests/test_cli_contract.py` protects that contract.

## Data flow and stage contract

```text
workspace declaration + source data
        │
        ▼
01 surface → 02 shift → 03 selection → 04 validation
```

This is a one-way artifact chain, not a request to rerun every earlier stage indiscriminately. Validation consumes the selected Stage 3 artifacts and writes the final verdict. See [the pipeline protocol](pipeline/README.md) for the full contract.

## Boundaries

### Domain

`domain/` owns deterministic numerical work: barrier calculations, features, baseline shifts, selection scores, and validation statistics. It must not depend on `infrastructure`, `pipeline`, or `presentation`; it also does not read/write files or provide CLI entrypoints. Add reusable, source-independent feature transforms in `domain/features.py`.

### Infrastructure

`infrastructure/` owns system edges: data fetching, workspace parsing, optional plugin loading, artifact naming, and JSON/NPZ persistence. `artifact_io.py` is the only owner of direct NumPy persistence. `artifacts.py` is the source of truth for stage paths and artifact filenames; do not reconstruct those paths elsewhere.

### Pipeline

`pipeline/` owns numbered stage orchestration and artifact handoffs. Each `step_0N_*.py` writes that stage’s artifacts and validates the upstream contract it requires. `context.py` constructs per-run source and feature registries, so plugins cannot leak registrations across commands.

### Presentation

`presentation/` owns human-readable Excel workbooks and diagnostic figures. It may consume domain outputs and persisted artifacts, but must not depend on CLI or pipeline modules. Presentation never changes a measurement, decision, or artifact schema.

## Safe modification guide

| Change | Entry point | Preserve |
|---|---|---|
| Built-in feature | `domain/features.py` | domain’s no-I/O/no-outward-dependency rule |
| Data provider | `infrastructure/market_data.py` or workspace plugin | source names named in `universe.json` |
| Artifact schema/path | `infrastructure/artifact_io.py`, `artifacts.py` | consumers and contract tests |
| Statistical decision | `domain/validation.py`, `pipeline/step_04_validation.py` | simulated-OHLC selected-bin null and BH correction |
| Workbook or figure | `presentation/` | artifacts remain the numerical source of truth |
| New stage or command | `pipeline/`, then `cli.py` | command/order and stage-directory contracts |

Run `python -m pytest` after a change. The architecture tests explicitly enforce dependency direction, persistence ownership, and the Stage 3 → Stage 4 validation handoff.

## Operational limits

Run commands from the repository root: `Workspace` resolves `workspaces/` from the current working directory. A stage may overwrite its own generated artifacts; declarations (`universe.json`) and plugins are source-controlled separately. Network-backed stages depend on the provider’s current data availability.
