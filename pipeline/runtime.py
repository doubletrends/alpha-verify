"""Shared runtime helpers for workspace-backed pipeline stages."""

from __future__ import annotations

import pandas as pd
import numpy as np

from data import features, fetcher
from engine import barrier
from universe import all_nodes, load_universe
from workspace import BASELINE_NODE, Workspace


def loader(ws: Workspace):
    """Return a per-workspace data loader with process-local source-set caching."""
    cache: dict[tuple, pd.DataFrame] = {}

    def load(sources: list) -> pd.DataFrame:
        key = tuple(sorted(sources))
        if key not in cache:
            cache[key] = fetcher.fetch(
                list(sources),
                start=ws.start_date,
                asset=ws.asset,
            ).dropna(subset=["close"])
        return cache[key]

    return load


def node_feature(ws: Workspace, node: dict, get_data):
    """Return ``(data, feature series)`` for a node, or raise if unavailable."""
    data = get_data(node["data"])
    if data.empty:
        raise ValueError("empty data")
    feat = features.compute(data, node["feature"], node["params"]).reindex(data.index)
    n_valid = int(feat.notna().sum())
    if n_valid < ws.min_obs:
        raise ValueError(f"only {n_valid} valid observations")
    return data, feat


def baseline_surface(ws: Workspace) -> np.ndarray | None:
    """The baseline node's unconditional probability surface, shaped ``theta x horizon``."""
    path = ws.baseline_cube
    if not path.exists():
        return None
    return barrier.load_cube(path)["prob"][:, 0, :]


def feature_panel(ws: Workspace) -> tuple[pd.DataFrame, dict, dict]:
    """
    Every non-baseline node's feature on one shared index, with its family.

    The baseline contributes zero log-odds delta to every bar and only narrows the
    valid-row intersection, so it is excluded from the feature matrix. Its prior still
    enters the model through the baseline surface.
    """
    universe = load_universe(ws.universe_path)
    get_data = loader(ws)
    feats, fams, data = {}, {}, None
    for node in all_nodes(universe):
        if node["id"] == BASELINE_NODE:
            continue
        try:
            d, f = node_feature(ws, node, get_data)
        except Exception:
            continue
        if data is None or len(d) > len(data):
            data = d
        feats[node["id"]] = f
        fams[node["id"]] = node["family"]
    if data is None or not feats:
        raise ValueError("no computable features in this workspace")
    return data, {k: v.reindex(data.index) for k, v in feats.items()}, fams
