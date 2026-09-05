"""Artifact paths and small persistence helpers for one workspace."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


_MARKET_COLUMNS = ("open", "high", "low", "close", "volume")
_REQUIRED_HISTORY_COLUMNS = ("index", "high", "low", "close")


class ArtifactPaths:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    @property
    def universe_path(self) -> Path:
        return self.workspace_dir / "universe.json"

    def cube_path(self, family: str, node_id: str) -> Path:
        return self.workspace_dir / "01_surface" / family / f"{node_id}.npz"

    def surface_path(self, family: str, node_id: str) -> Path:
        return self.workspace_dir / "01_surface" / family / f"{node_id}.xlsx"

    def shift_cube_path(self, family: str, node_id: str) -> Path:
        return self.workspace_dir / "02_shift" / family / f"{node_id}.npz"

    def shift_surface_path(self, family: str, node_id: str) -> Path:
        return self.workspace_dir / "02_shift" / family / f"{node_id}.xlsx"

    @property
    def selection_path(self) -> Path:
        return self.workspace_dir / "03_selection" / "selection.json"

    def selection_array_path(self, row: dict) -> Path:
        return self._ranked_path("03_selection", row, ".npz")

    def selection_surface_path(self, row: dict) -> Path:
        return self._ranked_path("03_selection", row, ".xlsx")

    def validation_array_path(self, row: dict) -> Path:
        return self._ranked_path("04_validation", row, ".npz")

    def validation_surface_path(self, row: dict) -> Path:
        return self._ranked_path("04_validation", row, ".xlsx")

    def _ranked_path(self, stage: str, row: dict, suffix: str) -> Path:
        rank = int(row["rank"])
        return (
            self.workspace_dir / stage / row["family"]
            / f"rank_{rank:03d}__{row['node']}__bin_{int(row['bin_number']):02d}{suffix}"
        )

    @property
    def validation_summary_path(self) -> Path:
        return self.workspace_dir / "04_validation" / "validation.json"

    @property
    def redundancy_path(self) -> Path:
        return self.workspace_dir / "05_redundancy" / "redundancy.json"

    @property
    def redundancy_array_path(self) -> Path:
        return self.workspace_dir / "05_redundancy" / "redundancy.npz"

    @property
    def redundancy_workbook_path(self) -> Path:
        return self.workspace_dir / "05_redundancy" / "redundancy.xlsx"

    @property
    def bayes_path(self) -> Path:
        return self.workspace_dir / "06_bayes" / "bayes.npz"

    @property
    def bayes_summary_path(self) -> Path:
        return self.workspace_dir / "06_bayes" / "bayes.json"

    @property
    def bayes_workbook_path(self) -> Path:
        return self.workspace_dir / "06_bayes" / "bayes.xlsx"

    @property
    def bayes_shift_workbook_path(self) -> Path:
        return self.workspace_dir / "06_bayes" / "bayes_shift.xlsx"

    @property
    def result_dir(self) -> Path:
        return self.workspace_dir / "result"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


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
