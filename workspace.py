"""Compatibility facade over workspace configuration, catalog, and artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from artifacts import ArtifactPaths, read_json, write_json
from catalog import NodeCatalog
from workspace_config import WorkspaceConfig
from workspace_plugins import load_workspace_plugin

ROOT = Path(__file__).resolve().parent
BASELINE_NODE = "baseline"


class Workspace:
    """One workspace's immutable declaration and artifact namespace."""

    def __init__(self, name: str):
        self.dir = ROOT / "workspaces" / name
        self.artifacts = ArtifactPaths(self.dir)
        self.catalog = NodeCatalog.load(self.artifacts.universe_path)
        self.config = WorkspaceConfig.from_meta(self.catalog.meta)
        load_workspace_plugin(self.dir)

    @property
    def universe_path(self) -> Path:
        return self.artifacts.universe_path

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
    def shift_deltas(self) -> np.ndarray:
        return self.config.deltas

    @property
    def shift_horizons(self) -> np.ndarray:
        return self.config.horizons

    # Compatibility aliases retain the artifact contract while callers migrate.
    @property
    def Δs(self) -> np.ndarray:
        return self.deltas

    @property
    def shift_Δs(self) -> np.ndarray:
        return self.shift_deltas

    @property
    def Δ_step(self) -> float:
        return self.config.delta_step

    @property
    def delta_step(self) -> float:
        return self.config.delta_step

    @property
    def n_bins(self) -> int:
        return self.config.n_bins

    @property
    def bayes_Δ(self) -> float | None:
        return self.config.bayes_delta

    @property
    def bayes_horizon(self) -> int | None:
        return self.config.bayes_horizon

    @property
    def bayes_folds(self) -> int:
        return self.config.bayes_folds

    @property
    def selection_top_k(self) -> int:
        return self.config.selection_top_k

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
    def cleared_path(self) -> Path:
        return self.artifacts.cleared_path

    @property
    def bayes_path(self) -> Path:
        return self.artifacts.bayes_path

    @property
    def bayes_summary_path(self) -> Path:
        return self.artifacts.bayes_summary_path

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
        horizons = self.shift_horizons
        horizon = self.bayes_horizon or int(horizons[len(horizons) // 2])
        if self.bayes_Δ is not None:
            return self.bayes_Δ, horizon
        horizon_index = int(np.flatnonzero(horizons == horizon)[0])
        deltas = self.shift_deltas
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
