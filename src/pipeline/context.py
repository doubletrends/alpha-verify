"""Shared runtime helpers for workspace-backed pipeline stages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.features import FeatureRegistry, register_builtin_features
from domain import barrier, shift
from infrastructure.artifacts.store import feature_from_artifact, market_data_from_artifact
from infrastructure.market_data.fetcher import SourceRegistry, register_builtin_sources
from infrastructure.workspaces.plugins import load_workspace_plugin
from infrastructure.workspaces.workspace import BASELINE_NODE, Workspace


class RunContext:
    """Per-command dependencies: workspace declaration and cached source data."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.sources = SourceRegistry()
        self.features = FeatureRegistry()
        register_builtin_sources(self.sources)
        register_builtin_features(self.features)
        load_workspace_plugin(workspace.dir, self.sources, self.features)

    def load_data(self, sources: list[str]) -> pd.DataFrame:
        return self.sources.fetch(
            sources,
            start=self.workspace.start_date,
            asset=self.workspace.asset,
        ).dropna(subset=["close"])

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


def artifact_node_feature(cube: dict, min_obs: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Reconstruct ``(data, feature)`` from a Stage 2 shift artifact."""
    try:
        data = market_data_from_artifact(cube)
        feat = feature_from_artifact(cube, data.index)
    except ValueError as error:
        raise ValueError(f"{error} - run surface and shift again") from error
    n_valid = int(feat.notna().sum())
    if min_obs is not None and n_valid < min_obs:
        raise ValueError(f"only {n_valid} valid observations")
    return data, feat


def baseline_surface(ws: Workspace) -> np.ndarray | None:
    """The full baseline node's unconditional probability surface, shaped ``Δ x horizon``."""
    path = ws.baseline_cube
    if not path.exists():
        return None
    return barrier.load_cube(path)["prob"][:, 0, :]


def artifact_feature_panel(ws: Workspace) -> tuple[pd.DataFrame, dict, dict]:
    """
    Full candidate feature panel reconstructed from Stage 2 shift artifacts.

    This keeps composition downstream of the artifact chain after regeneration: Stage 1
    stores the ordered market and feature arrays, Stage 2 copies them into the shift
    artifacts, and Bayes reads those files instead of fetching data again.
    """
    base_path = ws.shift_cube_path("_base", BASELINE_NODE)
    if not base_path.exists():
        raise ValueError("missing baseline shift artifact - run shift first")
    base = shift.load(base_path)
    try:
        data = market_data_from_artifact(base)
    except ValueError as error:
        raise ValueError(f"baseline {error} - run surface and shift again") from error

    feats, fams = {}, {}
    for node in ws.catalog.all_nodes():
        if node["id"] == BASELINE_NODE:
            continue
        path = ws.shift_cube_path(node["family"], node["id"])
        if not path.exists():
            continue
        cube = shift.load(path)
        try:
            feats[node["id"]] = feature_from_artifact(cube, data.index)
        except ValueError as error:
            raise ValueError(
                f"{node['id']} {error} - run surface and shift again"
            ) from error
        fams[node["id"]] = node["family"]

    if not feats:
        raise ValueError("no feature values in shift artifacts")
    return data, feats, fams
