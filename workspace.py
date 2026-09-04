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

from universe import load_universe

# The unconditional rate is an ordinary node whose feature is constant, so every bar
# falls in a single bin. Naming it here rather than special-casing it in the engine is
# what keeps the base rate on exactly the same path as every condition.
BASELINE_NODE = 'baseline'


class Workspace:
    def __init__(self, name: str):
        self.dir = ROOT / 'workspaces' / name
        meta = load_universe(self.dir / 'universe.json')['meta']

        self.asset      = meta['asset']
        self.start_date = meta['start_date']
        self.min_obs    = meta.get('min_obs', 100)

        interval = self.asset.get('interval', '1d')
        self.horizon_unit = 'h' if interval.endswith('h') else 'd'

        # The cube measures every horizon in this ladder; the summary grid below
        # selects the few that everything is judged on.
        hz = meta.get('horizons', {})
        self.h_min = hz.get('min', 1)
        self.h_max = hz.get('max', 30)

        # Barrier levels. These have to be scaled to the asset and the horizon: a 10%
        # barrier over 14 BTC days is reached about a fifth of the time, while the same
        # 10% over 24 NASDAQ hours is reached essentially never, so a range that suits
        # one asset is useless on the other.
        th = meta.get('theta', {})
        self.theta_min  = th.get('min', -0.20)
        self.theta_max  = th.get('max', 0.20)
        self.theta_step = th.get('step', 0.01)

        self.n_bins = meta.get('n_bins', 10)

        # The summary is the grid everything is *judged* on: a coarse subset of the
        # full ladder, chosen so adjacent cells are genuinely different measurements.
        # theta is given as magnitudes and mirrored, which guarantees the ladder stays
        # symmetric and 0 appears exactly once.
        sm = meta.get('summary', {})
        self.summary_theta_abs = sm.get('theta_abs',
                                        [0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20])
        self.summary_h = sm.get('horizons', [1, 2, 3, 5, 7, 14, 30])

        # The composition stage's headline target. It is declared rather than derived so
        # the figure everyone reads is not silently re-pointed by a change to the grid;
        # `bayes_target` falls back to a rule when a workspace does not name one.
        by = meta.get('bayes', {})
        self.bayes_theta   = by.get('theta')
        self.bayes_horizon = by.get('horizon')
        self.bayes_folds   = by.get('folds', 5)

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
    def horizons(self) -> np.ndarray:
        return np.arange(self.h_min, self.h_max + 1)

    @property
    def summary_thetas(self) -> np.ndarray:
        a = sorted({round(abs(float(v)), 10) for v in self.summary_theta_abs})
        return np.array([-v for v in reversed(a) if v > 0] + [v for v in a], dtype=float)

    @property
    def summary_horizons(self) -> np.ndarray:
        return np.array(sorted(int(h) for h in self.summary_h), dtype=int)

    @property
    def thetas(self) -> np.ndarray:
        from engine import barrier
        return barrier.theta_levels(self.theta_min, self.theta_max, self.theta_step)

    # ── paths ─────────────────────────────────────────────────────────────────

    @property
    def universe_path(self) -> Path:
        return self.dir / 'universe.json'

    # Pipeline artifact folders are numbered by command:
    # surface 01, summary 02, validation 03.

    def cube_path(self, family: str, node_id: str) -> Path:
        return self.dir / '01_surface_array' / family / f'{node_id}.npz'

    def has_cube(self, family: str, node_id: str) -> bool:
        return self.cube_path(family, node_id).exists()

    def surface_path(self, family: str, node_id: str) -> Path:
        return self.dir / '01_surface_xlsx' / family / f'{node_id}.xlsx'

    def has_surface(self, family: str, node_id: str) -> bool:
        return self.surface_path(family, node_id).exists()

    def summary_cube_path(self, family: str, node_id: str) -> Path:
        return self.dir / '02_summary_array' / family / f'{node_id}.npz'

    def has_summary_cube(self, family: str, node_id: str) -> bool:
        return self.summary_cube_path(family, node_id).exists()

    def summary_surface_path(self, family: str, node_id: str) -> Path:
        return self.dir / '02_summary_xlsx' / family / f'{node_id}.xlsx'

    def has_summary_surface(self, family: str, node_id: str) -> bool:
        return self.summary_surface_path(family, node_id).exists()

    @property
    def eval_path(self) -> Path:
        return self.dir / 'evaluation.json'

    def validation_path(self, family: str, node_id: str) -> Path:
        return self.dir / '03_validation_array' / family / f'{node_id}.npz'

    def has_validation(self, family: str, node_id: str) -> bool:
        return self.validation_path(family, node_id).exists()

    def validation_sheet_path(self, family: str, node_id: str) -> Path:
        return self.dir / '03_validation_xlsx' / family / f'{node_id}.xlsx'

    def has_validation_sheet(self, family: str, node_id: str) -> bool:
        return self.validation_sheet_path(family, node_id).exists()

    @property
    def baseline_cube(self) -> Path:
        return self.summary_cube_path('_base', BASELINE_NODE)

    @property
    def cleared_path(self) -> Path:
        return self.dir / '04_gate.json'

    @property
    def bayes_path(self) -> Path:
        return self.dir / '05_bayes.npz'

    @property
    def bayes_summary_path(self) -> Path:
        return self.dir / '05_bayes.json'

    @property
    def result_dir(self) -> Path:
        return self.dir / 'result'

    def bayes_target(self, baseline: np.ndarray) -> tuple[float, int]:
        """
        The (θ, h) the composition figures are drawn at.

        Taken from universe.json when it names one. Otherwise chosen from the workspace's
        own grid: the middle summary horizon, and the downside barrier whose
        unconditional rate there is closest to 30% -- common enough that a walk forward
        has events to score, rare enough that predicting it is not trivial. Deriving it
        from the grid rather than hard-coding a percentage is what lets the same rule
        serve BTC and NASDAQ without either being a special case.
        """
        hz = self.summary_horizons
        h = int(self.bayes_horizon) if self.bayes_horizon is not None else int(hz[len(hz) // 2])
        if self.bayes_theta is not None:
            return float(self.bayes_theta), h

        th = self.summary_thetas
        j = int(np.flatnonzero(hz == h)[0])
        down = np.flatnonzero(th < 0)
        rates = baseline[down, j]
        i = int(down[int(np.nanargmin(np.abs(rates - 0.30)))])
        return float(th[i]), h

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
