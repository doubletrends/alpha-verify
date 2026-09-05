"""Deprecated ``--gate`` compatibility alias for combined Stage 4 validation."""

from __future__ import annotations

from infrastructure.workspaces.workspace import Workspace
from pipeline.step_04_validation import finalize_validation


def cmd_gate(ws: Workspace, q: float = 0.05) -> None:
    """Finalize existing Stage 4 artifacts without recomputing their nulls."""
    print("--gate is deprecated; --validation now computes nulls and finalizes verdicts.")
    finalize_validation(ws, q)
