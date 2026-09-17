"""Small, consistent terminal reports for pipeline stages."""

from __future__ import annotations

from time import perf_counter
import sys

RULE = "=" * 60


def color_enabled() -> bool:
    """Use color for interactive runs; captured logs remain plain text."""
    return sys.stdout.isatty()


def ansi_styles(enabled: bool) -> tuple[str, str, str, str]:
    """Terminal ``(accent, bold, dim, reset)`` codes, or empty strings when disabled."""
    if not enabled:
        return "", "", "", ""
    return "\033[38;2;194;65;12m", "\033[1m", "\033[2m", "\033[0m"


class StageReport:
    """Report one stage without turning routine progress into log noise."""

    _HEADINGS = {
        1: (
            "Conditional barrier-touch probabilities",
            "conditional_probability[barrier, bin, horizon]",
        ),
        2: (
            "Conditional effect relative to the market baseline",
            "conditional_probability − baseline_probability",
        ),
        3: (
            "Statistical validation against simulated price paths",
            "Is each condition-bin score larger than expected by chance?",
        ),
        4: (
            "Selection of statistically cleared condition bins",
            "Retain every and only condition bin with raw p < 0.05",
        ),
        5: (
            "Summary of cleared nodes and bins",
            "Group the selected condition bins by node",
        ),
        6: (
            "Forecast from cleared conditions active on the last bar",
            "Naive Bayes over one active bin per family, checked against the historical joint rate",
        ),
    }

    def __init__(self, number: int) -> None:
        self._started = perf_counter()
        title, calculation = self._HEADINGS[number]
        accent, bold, dim, reset = ansi_styles(color_enabled())
        print(
            f"\n{accent}{RULE}{reset}\n"
            f"{accent}{bold}Stage {number}:{reset} {bold}{title}{reset}\n"
            f"{dim}{calculation}{reset}\n"
            f"{accent}{RULE}{reset}\n"
        )

    def line(self, message: str) -> None:
        color = color_enabled()
        _, _, dim, reset = ansi_styles(color)
        if message.startswith(("wrote ", "cleared ")) and color:
            print(f"  \033[32m{message}{reset}")
            return
        if message.startswith("completed in") and color:
            print(f"  {dim}{message}{reset}")
            return
        print(f"  {message}")

    def progress_group(self, label: str) -> None:
        """Start one compact, visually separate group of milestone updates."""
        accent, bold, _, reset = ansi_styles(color_enabled())
        print(f"\n  {accent}{bold}{label[:1].upper()}{label[1:]}{reset}")

    def progress(self, completed: int, total: int, milestone: int) -> None:
        print(f"    {completed}/{total} · {milestone}%")

    def summary(self, message: str) -> None:
        """Separate the completed work from its concise outcome."""
        print()
        self.line(message)

    def completed(self) -> None:
        self.line(f"completed in {perf_counter() - self._started:.1f}s")

    @staticmethod
    def warnings(items: list[str]) -> None:
        """Keep exceptional per-item detail visible without polluting stdout."""
        for item in items:
            print(f"warning: {item}", file=sys.stderr)


class MilestoneProgress:
    """Emit five capture-safe progress updates, regardless of workload size."""

    _MILESTONES = (20, 40, 60, 80, 100)

    def __init__(self, report: StageReport, phase: str, total: int) -> None:
        self._report = report
        self._phase = phase
        self._total = total
        self._completed = 0
        self._next = 0
        self._report.progress_group(phase)

    def advance(self) -> None:
        if self._total <= 0:
            return
        self._completed += 1
        while self._next < len(self._MILESTONES):
            milestone = self._MILESTONES[self._next]
            threshold = (self._total * milestone + 99) // 100
            if self._completed < threshold:
                return
            self._report.progress(self._completed, self._total, milestone)
            self._next += 1
