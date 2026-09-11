"""Shared runtime helpers for workspace-backed pipeline stages."""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from alphaverify.domain import barrier
from alphaverify.domain.features import FeatureRegistry, register_builtin_features
from alphaverify.infrastructure import artifact_io
from alphaverify.infrastructure.market_data import WorkspaceData
from alphaverify.infrastructure.workspace import BASELINE_NODE, Workspace
from alphaverify.infrastructure.workspace_plugins import load_workspace_plugin
from alphaverify.infrastructure.artifacts import market_history_key


class RunContext:
    """Per-command dependencies: workspace declaration and cached source data."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.data = WorkspaceData(workspace)
        self.features = FeatureRegistry()
        self._data: dict[tuple[str, ...], pd.DataFrame] = {}
        self._excursions: dict[tuple[tuple[str, ...], int], tuple[np.ndarray, np.ndarray]] = {}
        self._outcomes: dict[tuple, dict] = {}
        self._feature_primitives: dict[int, dict] = {}
        register_builtin_features(self.features)
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

    def forward_excursions(
        self, sources: list[str], data: pd.DataFrame, t_max: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reuse deterministic price excursions for nodes sharing a source panel."""
        key = (tuple(sources), int(t_max))
        if key not in self._excursions:
            self._excursions[key] = barrier.forward_extremes_upto(data, int(t_max))
        return self._excursions[key]

    def observed_outcomes(self, data: pd.DataFrame, barriers=None, horizons=None) -> dict:
        """Load or build the versioned source-level observed outcome cache."""
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
            downside_excursion, upside_excursion = barrier.forward_extremes_upto(
                data, int(horizons.max())
            )
            selected = horizons - 1
            downside_selected = downside_excursion[selected]
            upside_selected = upside_excursion[selected]
            price_eligible = np.isfinite(downside_selected) & np.isfinite(upside_selected)
            barrier_axis = barriers.reshape(1, 1, -1)
            touch_mask = np.where(
                barrier_axis < 0,
                downside_selected[:, :, None] <= barrier_axis,
                upside_selected[:, :, None] >= barrier_axis,
            ) & price_eligible[:, :, None]
            eligible_observation_count = price_eligible.sum(axis=1)
            hit_count = touch_mask.sum(axis=1)
            baseline_probability = np.where(
                eligible_observation_count[:, None] >= barrier.MIN_BIN_N,
                hit_count / np.maximum(eligible_observation_count[:, None], 1),
                np.nan,
            ).T
            cached = {
                "downside_excursion": downside_excursion,
                "upside_excursion": upside_excursion,
                "touch_mask": touch_mask,
                "baseline_probability": baseline_probability,
            }
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
    """Hydrate a thin Stage 2 result from its referenced Stage 1 arrays."""
    stored = artifact_io.load_shift(ws.shift_cube_path(node_id))
    if "conditional_probability" in stored:
        return stored
    meta = stored.get("meta", {})
    for artifact, expected in (
        (ws.cube_path(node_id), meta.get("source_sha256")),
        (ws.baseline_cube, meta.get("baseline_sha256")),
    ):
        if expected and hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Stage 1 source changed for {node_id}; rerun compare")
    surface = artifact_io.load_surface(ws.cube_path(node_id))
    baseline = baseline_surface(ws)
    if baseline is None:
        raise ValueError("baseline surface is unavailable")
    return {
        **surface, **stored, "baseline_probability": baseline,
        "meta": meta or surface.get("meta", {}),
    }
