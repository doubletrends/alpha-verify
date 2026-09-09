"""Shared runtime helpers for workspace-backed pipeline stages."""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from barrierlab.domain import barrier
from barrierlab.domain.features import FeatureRegistry, register_builtin_features
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.market_data import SourceRegistry, register_builtin_sources
from barrierlab.infrastructure.workspace import BASELINE_NODE, Workspace
from barrierlab.infrastructure.workspace_plugins import load_workspace_plugin
from barrierlab.infrastructure.artifacts import market_history_key


class RunContext:
    """Per-command dependencies: workspace declaration and cached source data."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.sources = SourceRegistry()
        self.features = FeatureRegistry()
        self._data: dict[tuple[str, ...], pd.DataFrame] = {}
        self._excursions: dict[tuple[tuple[str, ...], int], tuple[np.ndarray, np.ndarray]] = {}
        self._outcomes: dict[tuple, dict] = {}
        self._feature_primitives: dict[int, dict] = {}
        register_builtin_sources(self.sources)
        register_builtin_features(self.features)
        load_workspace_plugin(workspace.dir, self.sources, self.features)

    def load_data(self, sources: list[str]) -> pd.DataFrame:
        key = tuple(sources)
        if key not in self._data:
            self._data[key] = self.sources.fetch(
                sources,
                start=self.workspace.start_date,
                asset=self.workspace.asset,
            ).dropna(subset=["close"])
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

    def observed_outcomes(self, data: pd.DataFrame, deltas=None, horizons=None) -> dict:
        """Load or build the versioned source-level observed outcome cache."""
        deltas = self.workspace.deltas if deltas is None else np.asarray(deltas, dtype=float)
        horizons = self.workspace.horizons if horizons is None else np.asarray(horizons, dtype=int)
        key = market_history_key(data)
        memory_key = (key, tuple(deltas), tuple(horizons))
        if memory_key in self._outcomes:
            return self._outcomes[memory_key]
        path = self.workspace.observed_cache_path(key)
        expected = {
            "history_key": key,
            "measurement_version": barrier.MEASUREMENT_VERSION,
            "deltas": deltas.tolist(),
            "horizons": horizons.tolist(),
        }
        cached = artifact_io.load_observed_cache(path) if path.exists() else {}
        if cached.get("meta") != expected:
            low, high = barrier.forward_extremes_upto(
                data, int(horizons.max())
            )
            selected = horizons - 1
            lo, hi = low[selected], high[selected]
            price_ok = np.isfinite(lo) & np.isfinite(hi)
            delta_matrix = deltas.reshape(1, 1, -1)
            touches = np.where(
                delta_matrix < 0, lo[:, :, None] <= delta_matrix, hi[:, :, None] >= delta_matrix,
            ) & price_ok[:, :, None]
            counts = price_ok.sum(axis=1)
            hits = touches.sum(axis=1)
            baseline = np.where(
                counts[:, None] >= barrier.MIN_BIN_N,
                hits / np.maximum(counts[:, None], 1),
                np.nan,
            ).T
            cached = {
                "forward_low": low, "forward_high": high,
                "touches": touches, "baseline": baseline,
            }
            artifact_io.save_observed_cache(cached, path, expected)
            cached["meta"] = expected
        self._outcomes[memory_key] = cached
        return cached


def baseline_surface(ws: Workspace) -> np.ndarray | None:
    """The full baseline node's unconditional probability surface, shaped ``Δ x horizon``."""
    path = ws.baseline_cube
    if not path.exists():
        return None
    return artifact_io.load_surface(path)["prob"][:, 0, :]


def materialized_shift(ws: Workspace, node_id: str) -> dict:
    """Hydrate a thin Stage 2 result from its referenced Stage 1 arrays."""
    stored = artifact_io.load_shift(ws.shift_cube_path(node_id))
    if "prob" in stored:
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
    return {**surface, **stored, "base": baseline, "meta": meta or surface.get("meta", {})}
