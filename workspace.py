"""
A workspace is one asset, one sampling interval, and the feature universe tested on it.

universe.json is a pure declaration and is never written back. Progress is read off the
filesystem: a node has a surface iff its workbook exists. That keeps one source of truth
and means a partial run can always be resumed by looking at what is on disk.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

from tree.tree import load_tree


class Workspace:
    def __init__(self, name: str):
        self.dir = ROOT / 'workspaces' / name
        meta = load_tree(self.dir / 'universe.json')['meta']

        self.asset      = meta['asset']
        self.start_date = meta['start_date']
        self.min_obs    = meta.get('min_obs', 100)

        interval = self.asset.get('interval', '1d')
        self.horizon_unit = 'h' if interval.endswith('h') else 'd'

        # Horizons for the four sheets of Deliverable A.
        self.barrier_horizons = meta.get('barrier_horizons', [3, 7, 14, 30])

        # Barrier levels. These have to be scaled to the asset and the horizon: a 10%
        # barrier over 14 BTC days is reached about a fifth of the time, while the same
        # 10% over 24 NASDAQ hours is reached essentially never, so a range that suits
        # one asset is useless on the other.
        th = meta.get('theta', {})
        self.theta_min  = th.get('min', -0.20)
        self.theta_max  = th.get('max', 0.20)
        self.theta_step = th.get('step', 0.01)

        self.n_bins = meta.get('n_bins', 10)

        # Economic filter.
        ec = meta.get('evaluate', {})
        self.min_dev   = ec.get('min_dev', 10.0)
        self.min_bin_n = ec.get('min_bin_n', 50)
        self.min_run   = ec.get('min_run', 2)

        self._load_plugin()

    def _load_plugin(self) -> None:
        path = self.dir / 'plugin.py'
        if not path.exists():
            return
        spec = importlib.util.spec_from_file_location(f'_ws_plugin_{self.dir.name}', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    @property
    def thetas(self) -> np.ndarray:
        from engine import barrier
        return barrier.theta_levels(self.theta_min, self.theta_max, self.theta_step)

    # ── paths ─────────────────────────────────────────────────────────────────

    @property
    def tree_path(self) -> Path:
        return self.dir / 'universe.json'

    def surface_path(self, family: str, node_id: str) -> Path:
        return self.dir / 'surfaces' / family / f'{node_id}.xlsx'

    def has_surface(self, family: str, node_id: str) -> bool:
        return self.surface_path(family, node_id).exists()

    @property
    def eval_path(self) -> Path:
        return self.dir / 'evaluation.json'

    @property
    def validation_path(self) -> Path:
        return self.dir / 'validation.json'

    @property
    def cleared_path(self) -> Path:
        return self.dir / 'cleared.json'

    # ── json helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
