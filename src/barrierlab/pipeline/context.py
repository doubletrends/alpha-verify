"""Shared runtime helpers for workspace-backed pipeline stages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from barrierlab.domain import barrier
from barrierlab.domain.features import FeatureRegistry, register_builtin_features
from barrierlab.infrastructure import artifact_io
from barrierlab.infrastructure.market_data import SourceRegistry, register_builtin_sources
from barrierlab.infrastructure.workspace import BASELINE_NODE, Workspace
from barrierlab.infrastructure.workspace_plugins import load_workspace_plugin


class RunContext:
    """Per-command dependencies: workspace declaration and cached source data."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.sources = SourceRegistry()
        self.features = FeatureRegistry()
        self._data: dict[tuple[str, ...], pd.DataFrame] = {}
        self._excursions: dict[tuple[tuple[str, ...], int], tuple[np.ndarray, np.ndarray]] = {}
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
        feat = self.features.compute(data, node["feature"], node["params"]).reindex(data.index)
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


def baseline_surface(ws: Workspace) -> np.ndarray | None:
    """The full baseline node's unconditional probability surface, shaped ``Δ x horizon``."""
    path = ws.baseline_cube
    if not path.exists():
        return None
    return artifact_io.load_surface(path)["prob"][:, 0, :]
