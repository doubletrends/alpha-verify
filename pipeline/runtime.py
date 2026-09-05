"""Shared runtime helpers for workspace-backed pipeline stages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from artifacts import feature_from_artifact, market_data_from_artifact
from data import features, fetcher
from engine import barrier, shift
from workspace import BASELINE_NODE, Workspace


class RunContext:
    """Per-command dependencies: workspace declaration and cached source data."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.catalog = workspace.catalog
        self.artifacts = workspace.artifacts
        self._data_cache: dict[tuple, pd.DataFrame] = {}

    def load_data(self, sources: list[str]) -> pd.DataFrame:
        key = tuple(sorted(sources))
        if key not in self._data_cache:
            self._data_cache[key] = fetcher.fetch(
                list(sources),
                start=self.workspace.start_date,
                asset=self.workspace.asset,
            ).dropna(subset=["close"])
        return self._data_cache[key]

    def node_feature(self, node: dict) -> tuple[pd.DataFrame, pd.Series]:
        """Return ``(data, feature series)`` for a node, or raise if unavailable."""
        data = self.load_data(node["data"])
        if data.empty:
            raise ValueError("empty data")
        feat = features.compute(data, node["feature"], node["params"]).reindex(data.index)
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
        raise ValueError(f"{error} - rerun --surface and --shift") from error
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
    Feature panel reconstructed from Stage 2 shift artifacts.

    This keeps composition downstream of the artifact chain after regeneration: Stage 1
    stores the ordered market and feature arrays, Stage 2 copies them into the shift
    artifacts, and Bayes reads those files instead of fetching data again.
    """
    base_path = ws.shift_cube_path("_base", BASELINE_NODE)
    if not base_path.exists():
        raise ValueError("missing baseline shift artifact - run --shift first")
    base = shift.load(base_path)
    try:
        data = market_data_from_artifact(base)
    except ValueError as error:
        raise ValueError(f"baseline {error} - rerun --surface and --shift") from error

    selected = [r for r in ws.read_json(ws.selection_path).get("selected", []) if ws.has_selection_array(r)]
    selected_ids = {r["node"] for r in selected}
    if not selected_ids:
        raise ValueError("missing selection artifact - run --selection first")

    feats, fams = {}, {}
    for node in ws.catalog.all_nodes():
        if node["id"] == BASELINE_NODE:
            continue
        if node["id"] not in selected_ids:
            continue
        path = ws.shift_cube_path(node["family"], node["id"])
        if not path.exists():
            continue
        cube = shift.load(path)
        try:
            feats[node["id"]] = feature_from_artifact(cube, data.index)
        except ValueError as error:
            raise ValueError(
                f"{node['id']} {error} - rerun --surface and --shift"
            ) from error
        fams[node["id"]] = node["family"]

    if not feats:
        raise ValueError("no feature values in shift artifacts")
    return data, feats, fams
