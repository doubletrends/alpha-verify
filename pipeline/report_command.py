"""Stage 7 command: render report figures from stored artifacts."""

from __future__ import annotations

from workspace import Workspace


def cmd_report(ws: Workspace) -> None:
    """Stage 7: render the audience-facing figures from the artifacts on disk."""
    from engine import report

    report.build(ws)
