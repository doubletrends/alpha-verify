"""Artifact paths and history reconstruction for one workspace."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_MARKET_COLUMNS = ("open", "high", "low", "close", "volume")
_REQUIRED_HISTORY_COLUMNS = ("index", "high", "low", "close")

STAGE_DIRECTORIES = {
    "surface": "01_surface",
    "shift": "02_shift",
    "validation": "03_validation",
}


class ArtifactPaths:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    @property
    def universe_path(self) -> Path:
        return self.workspace_dir / "universe.json"

    def stage_dir(self, stage: str) -> Path:
        return self.workspace_dir / STAGE_DIRECTORIES[stage]

    def cube_path(self, node_id: str) -> Path:
        return self.stage_dir("surface") / "array" / f"{node_id}.safetensors"

    def surface_path(self, node_id: str) -> Path:
        return self.stage_dir("surface") / "spreadsheet" / f"{node_id}.xlsx"

    def shift_cube_path(self, node_id: str) -> Path:
        return self.stage_dir("shift") / "array" / f"{node_id}.safetensors"

    def shift_surface_path(self, node_id: str) -> Path:
        return self.stage_dir("shift") / "spreadsheet" / f"{node_id}.xlsx"

    @property
    def validation_summary_path(self) -> Path:
        return self.stage_dir("validation") / "validation.json"

def market_data_from_artifact(artifact: dict) -> pd.DataFrame:
    """Restore the ordered OHLCV history retained in a pipeline artifact."""
    missing = [key for key in _REQUIRED_HISTORY_COLUMNS if key not in artifact]
    if missing:
        raise ValueError(f"artifact lacks ordered history ({', '.join(missing)})")
    index = pd.to_datetime(artifact["index"])
    data = pd.DataFrame(
        {key: artifact[key].astype(float) for key in _MARKET_COLUMNS if key in artifact},
        index=index,
    ).dropna(subset=["close"])
    data.index.name = "Date"
    if data.empty:
        raise ValueError("artifact history is empty")
    return data


def feature_from_artifact(artifact: dict, index: pd.Index) -> pd.Series:
    """Restore a feature series and align it to reconstructed market history."""
    if "feature_values" not in artifact or "index" not in artifact:
        raise ValueError("artifact lacks ordered feature values")
    values = pd.Series(
        artifact["feature_values"].astype(float),
        index=pd.to_datetime(artifact["index"]),
        name="feature",
    )
    return values.reindex(index)
