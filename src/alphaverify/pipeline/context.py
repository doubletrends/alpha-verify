"""Shared runtime helpers for workspace-backed pipeline stages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaverify.domain import barrier
from alphaverify.domain.features import FeatureRegistry
from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.artifact_history import market_history_key
from alphaverify.infrastructure.market_data import WorkspaceData
from alphaverify.infrastructure.workspace import Workspace
from alphaverify.infrastructure.workspace_plugins import load_workspace_plugin


class RunContext:
    """Per-command dependencies: workspace declaration and cached source data."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.data = WorkspaceData(workspace)
        self.features = FeatureRegistry()
        self._data: dict[tuple[str, ...], pd.DataFrame] = {}
        self._outcomes: dict[tuple, dict] = {}
        self._feature_primitives: dict[int, dict] = {}
        load_workspace_plugin(workspace.dir, self.features)

    def load_data(self, sources: list[str]) -> pd.DataFrame:
        key = tuple(sources)
        if key not in self._data:
            self._data[key] = self.data.fetch(sources)
        return self._data[key]

    def node_feature(self, node: dict) -> tuple[pd.DataFrame, pd.Series]:
        """Return ``(data, feature series)`` for a node, or raise if unavailable."""
        data = self.load_data(node["data"])
        if data.empty:
            raise ValueError("empty data")
        primitives = self._feature_primitives.setdefault(id(data), {})
        feat = self.features.compute(data, node["feature"], node["params"], primitives).reindex(data.index)
        n_valid = int(feat.notna().sum())
        if n_valid < self.workspace.min_obs:
            raise ValueError(f"only {n_valid} valid observations")
        return data, feat

    def observed_outcomes(self, data: pd.DataFrame, barriers=None, horizons=None) -> dict:
        """Load or build the persisted, versioned observed outcomes of one history."""
        barriers = (
            self.workspace.barriers if barriers is None
            else np.asarray(barriers, dtype=float)
        )
        horizons = self.workspace.horizons if horizons is None else np.asarray(horizons, dtype=int)
        key = market_history_key(data)
        memory_key = (key, tuple(barriers), tuple(horizons))
        if memory_key in self._outcomes:
            return self._outcomes[memory_key]
        path = self.workspace.observed_cache_path(key)
        expected = {
            "artifact_schema_version": artifact_io.ARTIFACT_SCHEMA_VERSION,
            "history_key": key,
            "measurement_version": barrier.MEASUREMENT_VERSION,
            "barriers": barriers.tolist(),
            "horizons": horizons.tolist(),
        }
        cached = artifact_io.load_observed_cache(path) if path.exists() else {}
        if cached.get("meta") != expected:
            cached = barrier.observed_outcomes(data, barriers, horizons)
            artifact_io.save_observed_cache(cached, path, expected)
            cached["meta"] = expected
        self._outcomes[memory_key] = cached
        return cached


def baseline_surface(ws: Workspace) -> np.ndarray | None:
    """Unconditional probability surface, shaped ``barrier × horizon``."""
    path = ws.baseline_cube
    if not path.exists():
        return None
    return artifact_io.load_surface(path)["conditional_probability"][:, 0, :]


def materialized_shift(ws: Workspace, node_id: str) -> dict:
    """Hydrate a Stage 2 shift from its referenced Stage 1 arrays."""
    stored = artifact_io.load_shift(ws.shift_cube_path(node_id))
    meta = stored["meta"]
    for artifact, expected in (
        (ws.cube_path(node_id), meta.get("source_sha256")),
        (ws.baseline_cube, meta.get("baseline_sha256")),
    ):
        if expected and artifact_io.file_sha256(artifact) != expected:
            raise ValueError(f"Stage 1 source changed for {node_id}; rerun compare")
    surface = artifact_io.load_surface(ws.cube_path(node_id))
    baseline = baseline_surface(ws)
    if baseline is None:
        raise ValueError("baseline surface is unavailable")
    return {**surface, **stored, "baseline_probability": baseline}
