"""Stage 7 command: render report figures from stored artifacts."""

from __future__ import annotations

from infrastructure.workspace import Workspace
from pipeline.step_04_validation import validation_summary_is_current


def cmd_report(ws: Workspace) -> None:
    """Stage 7: render the audience-facing figures from the artifacts on disk."""
    validation = ws.read_json(ws.validation_summary_path)
    if not validation_summary_is_current(ws, validation):
        print("No current complete 04_validation/validation.json - run validation first.")
        return

    from presentation.report import build

    build(ws, validation)
