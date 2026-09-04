"""Shared runtime helpers for workspace-backed pipeline stages."""

from __future__ import annotations

import pandas as pd
import numpy as np

from data import features, fetcher
from engine import barrier, shift
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


def artifact_node_feature(cube: dict, min_obs: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Reconstruct ``(data, feature)`` from a Stage 2 shift artifact."""
    needed = ("index", "feature_values", "high", "low", "close")
    missing = [k for k in needed if k not in cube]
    if missing:
        raise ValueError(
            "shift artifact lacks ordered history "
            f"({', '.join(missing)}) - rerun --surface and --shift"
        )

    idx = pd.to_datetime(cube["index"])
    data = pd.DataFrame(
        {k: cube[k].astype(float) for k in ("open", "high", "low", "close", "volume") if k in cube},
        index=idx,
    )
    data.index.name = "Date"
    data = data.dropna(subset=["close"])
    if data.empty:
        raise ValueError("empty artifact history")

    feat = pd.Series(cube["feature_values"].astype(float), index=idx, name="feature").reindex(data.index)
    n_valid = int(feat.notna().sum())
    if min_obs is not None and n_valid < min_obs:
        raise ValueError(f"only {n_valid} valid observations")
    return data, feat


def baseline_surface(ws: Workspace) -> np.ndarray | None:
    """The full baseline node's unconditional probability surface, shaped ``theta x horizon``."""
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
    universe = load_universe(ws.universe_path)
    base_path = ws.shift_cube_path("_base", BASELINE_NODE)
    if not base_path.exists():
        raise ValueError("missing baseline shift artifact - run --shift first")
    base = shift.load(base_path)
    needed = ("index", "high", "low", "close")
    missing = [k for k in needed if k not in base]
    if missing:
        raise ValueError(
            "baseline shift artifact lacks ordered history "
            f"({', '.join(missing)}) - rerun --surface and --shift"
        )

    idx = pd.to_datetime(base["index"])
    data = pd.DataFrame(
        {k: base[k].astype(float) for k in ("open", "high", "low", "close", "volume") if k in base},
        index=idx,
    )
    data.index.name = "Date"

    selected = [r for r in ws.read_json(ws.selection_path).get("selected", []) if ws.has_selection_array(r)]
    selected_ids = {r["node"] for r in selected}
    if not selected_ids:
        raise ValueError("missing selection artifact - run --selection first")

    feats, fams = {}, {}
    for node in all_nodes(universe):
        if node["id"] == BASELINE_NODE:
            continue
        if node["id"] not in selected_ids:
            continue
        path = ws.shift_cube_path(node["family"], node["id"])
        if not path.exists():
            continue
        cube = shift.load(path)
        if "feature_values" not in cube or "index" not in cube:
            raise ValueError(
                f"{node['id']} shift artifact lacks ordered feature values - "
                "rerun --surface and --shift"
            )
        s = pd.Series(cube["feature_values"].astype(float), index=pd.to_datetime(cube["index"]))
        feats[node["id"]] = s.reindex(data.index)
        fams[node["id"]] = node["family"]

    if not feats:
        raise ValueError("no feature values in shift artifacts")
    return data, feats, fams
