"""Workspace configuration, node catalog, and artifact namespace."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from barrierlab.infrastructure.artifact_io import read_json, write_json
from barrierlab.infrastructure.artifacts import ArtifactPaths

BASELINE_NODE = "baseline"


@dataclass(frozen=True)
class WorkspaceConfig:
    """Immutable, execution-relevant workspace settings."""

    asset: dict
    start_date: str
    min_obs: int
    t_min: int
    t_max: int
    barrier_min: float
    barrier_max: float
    barrier_step: float
    n_bins: int
    target_barrier: float | None
    target_horizon: int | None
    min_dev: float
    min_bin_n: int
    min_run: int

    @classmethod
    def from_meta(cls, meta: dict) -> "WorkspaceConfig":
        asset = meta["asset"]
        horizons = meta.get("horizons", {})
        barriers = meta.get("barriers", meta.get("Delta", meta.get("\u0394", {})))
        target = meta.get("target", {})
        evaluate = meta.get("evaluate", {})
        # Unmarked declarations predate fractional shifts and express min_dev in pp.
        shift_unit = evaluate.get("shift_unit", "percentage_points")
        if shift_unit not in ("percentage_points", "probability_difference"):
            raise ValueError(f"unknown evaluate.shift_unit: {shift_unit}")
        min_dev = float(evaluate.get(
            "min_dev", 10.0 if shift_unit == "percentage_points" else 0.10,
        ))
        if shift_unit == "percentage_points":
            min_dev /= 100.0
        if not np.isfinite(min_dev) or not 0 <= min_dev <= 1:
            raise ValueError("evaluate.min_dev must represent a probability difference in [0, 1]")
        target_barrier = target.get("barrier", target.get("Delta", target.get("\u0394")))
        return cls(
            asset=dict(asset),
            start_date=meta["start_date"],
            min_obs=int(meta.get("min_obs", 100)),
            t_min=int(horizons.get("min", 1)),
            t_max=int(horizons.get("max", 30)),
            barrier_min=float(barriers.get("min", -0.20)),
            barrier_max=float(barriers.get("max", 0.20)),
            barrier_step=float(barriers.get("step", 0.01)),
            n_bins=int(meta.get("n_bins", 10)),
            target_barrier=(
                None if target_barrier is None else float(target_barrier)
            ),
            target_horizon=(
                None
                if target.get("horizon") is None
                else int(target["horizon"])
            ),
            min_dev=min_dev,
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
    def barriers(self) -> np.ndarray:
        count = int(round(
            (self.barrier_max - self.barrier_min) / self.barrier_step
        )) + 1
        return np.round(np.linspace(self.barrier_min, self.barrier_max, count), 10)


@dataclass(frozen=True)
class NodeCatalog:
    """The immutable node and family declaration loaded from ``universe.json``."""

    raw: dict

    @classmethod
    def load(cls, path: Path) -> "NodeCatalog":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw.get("meta"), dict) or not isinstance(raw.get("families"), dict):
            raise ValueError(f"invalid workspace declaration: {path}")
        return cls(raw)

    @property
    def meta(self) -> dict:
        return self.raw["meta"]

    def all_nodes(self) -> list[dict]:
        return [node for nodes in self.raw["families"].values() for node in nodes]

    def find(self, node_id: str) -> dict:
        for node in self.all_nodes():
            if node["id"] == node_id:
                return node
        raise ValueError(f"Node '{node_id}' not found in universe.json")


class Workspace:
    """One workspace's immutable declaration and artifact namespace."""

    def __init__(self, name: str, workspaces_dir: Path | None = None):
        root = workspaces_dir or Path.cwd() / "workspaces"
        self.dir = root / name
        self.artifacts = ArtifactPaths(self.dir)
        self.catalog = NodeCatalog.load(self.artifacts.universe_path)
        self.config = WorkspaceConfig.from_meta(self.catalog.meta)

    @property
    def root_dir(self) -> Path:
        return self.dir.parent.parent

    @property
    def asset(self) -> dict:
        return self.config.asset

    @property
    def start_date(self) -> str:
        return self.config.start_date

    @property
    def min_obs(self) -> int:
        return self.config.min_obs

    @property
    def horizon_unit(self) -> str:
        return self.config.horizon_unit

    @property
    def horizons(self) -> np.ndarray:
        return self.config.horizons

    @property
    def barriers(self) -> np.ndarray:
        return self.config.barriers

    @property
    def deltas(self) -> np.ndarray:
        """Compatibility alias for the pre-v2 Python API."""
        return self.barriers

    @property
    def barrier_step(self) -> float:
        return self.config.barrier_step

    @property
    def delta_step(self) -> float:
        """Compatibility alias for the pre-v2 Python API."""
        return self.barrier_step

    @property
    def n_bins(self) -> int:
        return self.config.n_bins

    @property
    def target_barrier(self) -> float | None:
        return self.config.target_barrier

    @property
    def target_delta(self) -> float | None:
        """Compatibility alias for the pre-v2 Python API."""
        return self.target_barrier

    @property
    def target_horizon(self) -> int | None:
        return self.config.target_horizon

    @property
    def min_dev(self) -> float:
        return self.config.min_dev

    @property
    def min_bin_n(self) -> int:
        return self.config.min_bin_n

    @property
    def min_run(self) -> int:
        return self.config.min_run

    @property
    def baseline_cube(self) -> Path:
        return self.cube_path(BASELINE_NODE)

    @property
    def validation_summary_path(self) -> Path:
        return self.artifacts.validation_summary_path

    @property
    def selection_summary_path(self) -> Path:
        return self.artifacts.selection_summary_path

    def cube_path(self, node_id: str) -> Path:
        return self.artifacts.cube_path(node_id)

    def observed_cache_path(self, history_key: str) -> Path:
        return self.artifacts.observed_cache_path(history_key)

    def has_cube(self, node_id: str) -> bool:
        return self.cube_path(node_id).exists()

    def surface_path(self, node_id: str) -> Path:
        return self.artifacts.surface_path(node_id)

    def has_surface(self, node_id: str) -> bool:
        return self.surface_path(node_id).exists()

    def shift_cube_path(self, node_id: str) -> Path:
        return self.artifacts.shift_cube_path(node_id)

    def has_shift_cube(self, node_id: str) -> bool:
        return self.shift_cube_path(node_id).exists()

    def shift_surface_path(self, node_id: str) -> Path:
        return self.artifacts.shift_surface_path(node_id)

    def has_shift_surface(self, node_id: str) -> bool:
        return self.shift_surface_path(node_id).exists()

    def target(self, baseline: np.ndarray) -> tuple[float, int]:
        """Return an optional inspection target; validation scores the full grid."""
        horizons = self.horizons
        horizon = self.target_horizon or int(horizons[len(horizons) // 2])
        if self.target_barrier is not None:
            return self.target_barrier, horizon
        horizon_index = int(np.flatnonzero(horizons == horizon)[0])
        barriers = self.barriers
        downside = np.flatnonzero(barriers < 0)
        rates = baseline[downside, horizon_index]
        delta_index = int(downside[int(np.nanargmin(np.abs(rates - 0.30)))])
        return float(barriers[delta_index]), horizon

    @staticmethod
    def read_json(path: Path) -> dict:
        return read_json(path)

    @staticmethod
    def write_json(path: Path, payload: dict) -> None:
        write_json(path, payload)
