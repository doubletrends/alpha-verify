"""Stage 7 command: render report figures from stored artifacts."""

from __future__ import annotations

from barrierlab.infrastructure.workspace import Workspace


def cmd_report(ws: Workspace) -> None:
    """Stage 7: render the audience-facing figures from the artifacts on disk."""
    summary = ws.read_json(ws.composition_summary_path)
    validation = summary.get("report_validation")
    if not summary or not isinstance(validation, dict) or not validation.get("complete"):
        print("No complete Stage 6 report bundle - run composition first.")
        return

    from barrierlab.presentation.report import build

    build(ws, validation)
