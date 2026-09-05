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

        # The cube measures every horizon in this ladder. Stage 2 keeps this full grid
        # and subtracts the baseline surface from it.
        ts = meta.get('horizons', {})
        self.t_min = ts.get('min', 1)
        self.t_max = ts.get('max', 30)

        # Barrier levels. These have to be scaled to the asset and the horizon: a 10%
        # barrier over 14 BTC days is reached about a fifth of the time, while the same
        # 10% over 24 NASDAQ hours is reached essentially never, so a range that suits
        # one asset is useless on the other.
        th = meta.get('Δ', {})
        self.Δ_min  = th.get('min', -0.20)
        self.Δ_max  = th.get('max', 0.20)
        self.Δ_step = th.get('step', 0.01)

        self.n_bins = meta.get('n_bins', 10)

        # Legacy workspace declarations may still carry a `summary` section. Stage 2
        # now uses the full grid, so those values are kept only for reading older
        # workspace files and no longer define the judged surface.
        sm = meta.get('summary', {})
        self.summary_Δ_abs = sm.get('Δ_abs',
                                        [0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20])
        self.summary_t = sm.get('horizons', [1, 2, 3, 5, 7, 14, 30])

        # The composition stage's headline target. It is declared rather than derived so
        # the figure everyone reads is not silently re-pointed by a change to the grid;
        # `bayes_target` falls back to a rule when a workspace does not name one.
        by = meta.get('bayes', {})
        self.bayes_Δ   = by.get('Δ')
        self.bayes_horizon = by.get('horizon')
        self.bayes_folds   = by.get('folds', 5)

        # Selection and economic filter.
        se = meta.get('selection', {})
        self.selection_top_k = se.get('top_k', 20)

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
        return np.arange(self.t_min, self.t_max + 1)

    @property
    def Δs(self) -> np.ndarray:
        from engine import barrier
        return barrier.Δ_levels(self.Δ_min, self.Δ_max, self.Δ_step)

    @property
    def shift_Δs(self) -> np.ndarray:
        return self.Δs

    @property
    def shift_horizons(self) -> np.ndarray:
        return self.horizons

    # ── paths ─────────────────────────────────────────────────────────────────

    @property
    def universe_path(self) -> Path:
        return self.dir / 'universe.json'

    # Pipeline artifact folders are numbered by command:
    # surface 01, shift 02, selection 03, validation 04.

    def cube_path(self, family: str, node_id: str) -> Path:
        return self.dir / '01_surface_array' / family / f'{node_id}.npz'

    def has_cube(self, family: str, node_id: str) -> bool:
        return self.cube_path(family, node_id).exists()

    def surface_path(self, family: str, node_id: str) -> Path:
        return self.dir / '01_surface_xlsx' / family / f'{node_id}.xlsx'

    def has_surface(self, family: str, node_id: str) -> bool:
        return self.surface_path(family, node_id).exists()

    def shift_cube_path(self, family: str, node_id: str) -> Path:
        return self.dir / '02_shift_array' / family / f'{node_id}.npz'

    def has_shift_cube(self, family: str, node_id: str) -> bool:
        return self.shift_cube_path(family, node_id).exists()

    def shift_surface_path(self, family: str, node_id: str) -> Path:
        return self.dir / '02_shift_xlsx' / family / f'{node_id}.xlsx'

    def has_shift_surface(self, family: str, node_id: str) -> bool:
        return self.shift_surface_path(family, node_id).exists()

    @property
    def selection_path(self) -> Path:
        return self.dir / '03_selection_array' / 'selection.json'

    def selection_array_path(self, row: dict) -> Path:
        rank = int(row['rank'])
        return (
            self.dir / '03_selection_array' / row['family']
            / f'rank_{rank:03d}__{row["node"]}__bin_{int(row["bin_number"]):02d}.npz'
        )

    def has_selection_array(self, row: dict) -> bool:
        return self.selection_array_path(row).exists()

    def selection_surface_path(self, row: dict) -> Path:
        rank = int(row['rank'])
        return (
            self.dir / '03_selection_xlsx' / row['family']
            / f'rank_{rank:03d}__{row["node"]}__bin_{int(row["bin_number"]):02d}.xlsx'
        )

    def has_selection_surface(self, row: dict) -> bool:
        return self.selection_surface_path(row).exists()

    def validation_array_path(self, row: dict) -> Path:
        rank = int(row['rank'])
        return (
            self.dir / '04_validation_array' / row['family']
            / f'rank_{rank:03d}__{row["node"]}__bin_{int(row["bin_number"]):02d}.npz'
        )

    def validation_path(self, family: str, node_id: str) -> Path:
        return self.dir / '04_validation_array' / family / f'{node_id}.npz'

    def has_validation_array(self, row: dict) -> bool:
        return self.validation_array_path(row).exists()

    def has_validation(self, family: str, node_id: str) -> bool:
        return self.validation_path(family, node_id).exists()

    def validation_surface_path(self, row: dict) -> Path:
        rank = int(row['rank'])
        return (
            self.dir / '04_validation_xlsx' / row['family']
            / f'rank_{rank:03d}__{row["node"]}__bin_{int(row["bin_number"]):02d}.xlsx'
        )

    def validation_sheet_path(self, family: str, node_id: str) -> Path:
        return self.dir / '04_validation_xlsx' / family / f'{node_id}.xlsx'

    def has_validation_surface(self, row: dict) -> bool:
        return self.validation_surface_path(row).exists()

    def has_validation_sheet(self, family: str, node_id: str) -> bool:
        return self.validation_sheet_path(family, node_id).exists()

    @property
    def baseline_cube(self) -> Path:
        return self.cube_path('_base', BASELINE_NODE)

    @property
    def cleared_path(self) -> Path:
        return self.dir / '05_gate.json'

    @property
    def bayes_path(self) -> Path:
        return self.dir / '06_bayes.npz'

    @property
    def bayes_summary_path(self) -> Path:
        return self.dir / '06_bayes.json'

    @property
    def result_dir(self) -> Path:
        return self.dir / 'result'

    def bayes_target(self, baseline: np.ndarray) -> tuple[float, int]:
        """
        The (Δ, t) the composition figures are drawn at.

        Taken from universe.json when it names one. Otherwise chosen from the workspace's
        own grid: the middle summary horizon, and the downside barrier whose
        unconditional rate there is closest to 30% -- common enough that a walk forward
        has events to score, rare enough that predicting it is not trivial. Deriving it
        from the grid rather than hard-coding a percentage is what lets the same rule
        serve BTC and NASDAQ without either being a special case.
        """
        ts = self.shift_horizons
        t = int(self.bayes_horizon) if self.bayes_horizon is not None else int(ts[len(ts) // 2])
        if self.bayes_Δ is not None:
            return float(self.bayes_Δ), t

        th = self.shift_Δs
        j = int(np.flatnonzero(ts == t)[0])
        down = np.flatnonzero(th < 0)
        rates = baseline[down, j]
        i = int(down[int(np.nanargmin(np.abs(rates - 0.30)))])
        return float(th[i]), t

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
