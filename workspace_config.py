"""Immutable, execution-relevant workspace settings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorkspaceConfig:
    asset: dict
    start_date: str
    min_obs: int
    t_min: int
    t_max: int
    delta_min: float
    delta_max: float
    delta_step: float
    n_bins: int
    bayes_delta: float | None
    bayes_horizon: int | None
    bayes_folds: int
    selection_top_k: int
    min_dev: float
    min_bin_n: int
    min_run: int

    @classmethod
    def from_meta(cls, meta: dict) -> "WorkspaceConfig":
        asset = meta["asset"]
        horizons = meta.get("horizons", {})
        barriers = meta.get("Delta", meta.get("\u0394", {}))
        bayes = meta.get("bayes", {})
        selection = meta.get("selection", {})
        evaluate = meta.get("evaluate", {})
        bayes_delta = bayes.get("Delta", bayes.get("\u0394"))
        return cls(
            asset=dict(asset),
            start_date=meta["start_date"],
            min_obs=int(meta.get("min_obs", 100)),
            t_min=int(horizons.get("min", 1)),
            t_max=int(horizons.get("max", 30)),
            delta_min=float(barriers.get("min", -0.20)),
            delta_max=float(barriers.get("max", 0.20)),
            delta_step=float(barriers.get("step", 0.01)),
            n_bins=int(meta.get("n_bins", 10)),
            bayes_delta=None if bayes_delta is None else float(bayes_delta),
            bayes_horizon=None if bayes.get("horizon") is None else int(bayes["horizon"]),
            bayes_folds=int(bayes.get("folds", 5)),
            selection_top_k=int(selection.get("top_k", 20)),
            min_dev=float(evaluate.get("min_dev", 10.0)),
            min_bin_n=int(evaluate.get("min_bin_n", 50)),
            min_run=int(evaluate.get("min_run", 2)),
        )

    @property
    def horizon_unit(self) -> str:
        return "h" if self.asset.get("interval", "1d").endswith("h") else "d"

    @property
    def horizons(self) -> np.ndarray:
        return np.arange(self.t_min, self.t_max + 1)

    @property
    def deltas(self) -> np.ndarray:
        count = int(round((self.delta_max - self.delta_min) / self.delta_step)) + 1
        return np.round(np.linspace(self.delta_min, self.delta_max, count), 10)
