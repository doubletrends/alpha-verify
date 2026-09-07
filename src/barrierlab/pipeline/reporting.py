"""Small, consistent terminal reports for pipeline stages."""

from __future__ import annotations

from time import perf_counter
import sys


class StageReport:
    """Report one stage without turning routine progress into log noise."""

    _HEADINGS = {
        1: (
            "Conditional barrier-touch probabilities",
            "P(touch Δ within t bars | condition)",
        ),
        2: (
            "Conditional effect relative to the market baseline",
            "P(touch Δ within t bars | condition) − P(touch Δ within t bars)",
        ),
        3: (
            "Strongest directional condition effects",
            "Rank bins by summed |shift(+Δ) − shift(−Δ)| across the full grid",
        ),
        4: (
            "Statistical validation against simulated price paths",
            "Is the selected directional effect larger than expected by chance?",
        ),
    }
    _RULE = "=" * 60

    def __init__(self, number: int, command: str, workspace: str) -> None:
        self._started = perf_counter()
        title, calculation = self._HEADINGS[number]
        color = self._color_enabled()
        cyan = "\033[36m" if color else ""
        bold = "\033[1m" if color else ""
        dim = "\033[2m" if color else ""
        reset = "\033[0m" if color else ""
        print(
            f"\n{cyan}{self._RULE}{reset}\n"
            f"{cyan}{bold}Stage {number}:{reset} {bold}{title}{reset}\n"
            f"{dim}{calculation}{reset}\n"
            f"{cyan}{self._RULE}{reset}\n"
        )

    def line(self, message: str) -> None:
        if message.startswith(("wrote ", "cleared ")) and self._color_enabled():
            print(f"  \033[32m{message}\033[0m")
            return
        if message.startswith("completed in") and self._color_enabled():
            print(f"  \033[2m{message}\033[0m")
            return
        print(f"  {message}")

    def progress_group(self, label: str) -> None:
        """Start one compact, visually separate group of milestone updates."""
        color = self._color_enabled()
        cyan = "\033[36m" if color else ""
        bold = "\033[1m" if color else ""
        reset = "\033[0m" if color else ""
        print(f"\n  {cyan}{bold}{label[:1].upper()}{label[1:]}{reset}")

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

    @staticmethod
    def _color_enabled() -> bool:
        """Use color for interactive runs; captured logs remain plain text."""
        return sys.stdout.isatty()


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
