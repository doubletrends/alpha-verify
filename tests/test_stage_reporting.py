from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from barrierlab.pipeline.reporting import MilestoneProgress, StageReport


def test_stage_report_has_a_compact_consistent_shape() -> None:
    output = StringIO()
    with patch("barrierlab.pipeline.reporting.perf_counter", side_effect=(10.0, 12.34)):
        with redirect_stdout(output):
            report = StageReport(2, "shift", "nasdaq_daily")
            report.line("shifting 58 nodes against baseline")
            report.completed()

    assert output.getvalue().splitlines() == [
        "",
        "============================================================",
        "Stage 2: Conditional effect relative to the market baseline",
        "P(touch Δ within t bars | condition) − P(touch Δ within t bars)",
        "============================================================",
        "",
        "  shifting 58 nodes against baseline",
        "  completed in 2.3s",
    ]


def test_milestone_progress_emits_only_five_fixed_checkpoints() -> None:
    output = StringIO()
    with redirect_stdout(output):
        report = StageReport(1, "surface", "nasdaq_daily")
        progress = MilestoneProgress(report, "calculating arrays", 58)
        for _ in range(58):
            progress.advance()

    lines = output.getvalue().splitlines()
    assert lines[5:] == [
        "",
        "",
        "  Calculating arrays",
        "    12/58 · 20%",
        "    24/58 · 40%",
        "    35/58 · 60%",
        "    47/58 · 80%",
        "    58/58 · 100%",
    ]
