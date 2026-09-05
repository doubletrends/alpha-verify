"""Workspace configuration, node catalog, and artifact namespace."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from infrastructure.artifacts import ArtifactPaths, read_json, write_json

BASELINE_NODE = "baseline"


@dataclass(frozen=True)
class WorkspaceConfig:
    """Immutable, execution-relevant workspace settings."""

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
    demonstration_date: str | None
    selection_top_k: int
    null_alpha: float
    min_dev: float
    min_bin_n: int
    min_run: int

    @classmethod
    def from_meta(cls, meta: dict) -> "WorkspaceConfig":
        asset = meta["asset"]
        horizons = meta.get("horizons", {})
        barriers = meta.get("Delta", meta.get("\u0394", {}))
        bayes = meta.get("bayes", {})
        report = meta.get("report", {})
        selection = meta.get("selection", {})
        validation = meta.get("validation", {})
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
            demonstration_date=(
                None if report.get("demonstration_date") is None
                else str(report["demonstration_date"])
            ),
            selection_top_k=int(selection.get("top_k", 20)),
            null_alpha=float(validation.get("null_alpha", 0.01)),
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
    def deltas(self) -> np.ndarray:
        return self.config.deltas

    @property
    def delta_step(self) -> float:
        return self.config.delta_step

    @property
    def n_bins(self) -> int:
        return self.config.n_bins

    @property
    def bayes_delta(self) -> float | None:
        return self.config.bayes_delta

    @property
    def bayes_horizon(self) -> int | None:
        return self.config.bayes_horizon

    @property
    def bayes_folds(self) -> int:
        return self.config.bayes_folds

    @property
    def demonstration_date(self) -> str | None:
        return self.config.demonstration_date

    @property
    def selection_top_k(self) -> int:
        return self.config.selection_top_k

    @property
    def null_alpha(self) -> float:
        return self.config.null_alpha

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
        return self.cube_path("_base", BASELINE_NODE)

    @property
    def validation_summary_path(self) -> Path:
        return self.artifacts.validation_summary_path

    @property
    def redundancy_path(self) -> Path:
        return self.artifacts.redundancy_path

    @property
    def redundancy_array_path(self) -> Path:
        return self.artifacts.redundancy_array_path

    @property
    def redundancy_workbook_path(self) -> Path:
        return self.artifacts.redundancy_workbook_path

    @property
    def bayes_path(self) -> Path:
        return self.artifacts.bayes_path

    @property
    def bayes_summary_path(self) -> Path:
        return self.artifacts.bayes_summary_path

    @property
    def bayes_workbook_path(self) -> Path:
        return self.artifacts.bayes_workbook_path

    @property
    def bayes_shift_workbook_path(self) -> Path:
        return self.artifacts.bayes_shift_workbook_path

    @property
    def result_dir(self) -> Path:
        return self.artifacts.result_dir

    def cube_path(self, family: str, node_id: str) -> Path:
        return self.artifacts.cube_path(family, node_id)

    def has_cube(self, family: str, node_id: str) -> bool:
        return self.cube_path(family, node_id).exists()

    def surface_path(self, family: str, node_id: str) -> Path:
        return self.artifacts.surface_path(family, node_id)

    def has_surface(self, family: str, node_id: str) -> bool:
        return self.surface_path(family, node_id).exists()

    def shift_cube_path(self, family: str, node_id: str) -> Path:
        return self.artifacts.shift_cube_path(family, node_id)

    def has_shift_cube(self, family: str, node_id: str) -> bool:
        return self.shift_cube_path(family, node_id).exists()

    def shift_surface_path(self, family: str, node_id: str) -> Path:
        return self.artifacts.shift_surface_path(family, node_id)

    def has_shift_surface(self, family: str, node_id: str) -> bool:
        return self.shift_surface_path(family, node_id).exists()

    @property
    def selection_path(self) -> Path:
        return self.artifacts.selection_path

    def selection_array_path(self, row: dict) -> Path:
        return self.artifacts.selection_array_path(row)

    def has_selection_array(self, row: dict) -> bool:
        return self.selection_array_path(row).exists()

    def selection_surface_path(self, row: dict) -> Path:
        return self.artifacts.selection_surface_path(row)

    def has_selection_surface(self, row: dict) -> bool:
        return self.selection_surface_path(row).exists()

    def validation_array_path(self, row: dict) -> Path:
        return self.artifacts.validation_array_path(row)

    def has_validation_array(self, row: dict) -> bool:
        return self.validation_array_path(row).exists()

    def validation_surface_path(self, row: dict) -> Path:
        return self.artifacts.validation_surface_path(row)

    def has_validation_surface(self, row: dict) -> bool:
        return self.validation_surface_path(row).exists()

    def bayes_target(self, baseline: np.ndarray) -> tuple[float, int]:
        horizons = self.horizons
        horizon = self.bayes_horizon or int(horizons[len(horizons) // 2])
        if self.bayes_delta is not None:
            return self.bayes_delta, horizon
        horizon_index = int(np.flatnonzero(horizons == horizon)[0])
        deltas = self.deltas
        downside = np.flatnonzero(deltas < 0)
        rates = baseline[downside, horizon_index]
        delta_index = int(downside[int(np.nanargmin(np.abs(rates - 0.30)))])
        return float(deltas[delta_index]), horizon

    @staticmethod
    def read_json(path: Path) -> dict:
        return read_json(path)

    @staticmethod
    def write_json(path: Path, payload: dict) -> None:
        write_json(path, payload)
