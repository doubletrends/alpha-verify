import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from tree.tree import load_tree
from engine import outcomes


class Workspace:
    """
    One asset + one sampling interval + its feature universe + the barrier being measured.

    universe.json is a pure declaration and is never written to. A node's progress is
    read off the filesystem instead: it is 'tested' for an event iff its xlsx exists
    under that event's directory. That keeps one source of truth, and means two
    barriers (or two thresholds) can be swept independently without a shared status
    field drifting out of sync with what is actually on disk.
    """

    def __init__(self, name: str):
        self.dir  = ROOT / 'workspaces' / name
        tree      = load_tree(self.dir / 'universe.json')
        meta      = tree['meta']

        self.horizons     = meta['horizons']
        self.start_date   = meta['start_date']
        self.asset        = meta['asset']
        self.n_thresholds = meta.get('n_thresholds', 30)

        interval = self.asset.get('interval', '1d')
        self.horizon_unit = 'h' if interval.endswith('h') else 'd'

        self.sample_freq  = meta['sample_freq']
        self.min_obs      = meta.get('min_obs', 100)
        self.read_min_dev = meta.get('read_min_dev', 10.0)
        self.read_min_n   = meta.get('read_min_n', 50)
        self._display_nums = meta['display_horizons']

        self._load_plugin()

        # The outcome is loaded after the plugin so a workspace can register its own.
        # It must be declared: the barrier θ has to be scaled to the asset and horizon
        # (10% over 14 BTC days is a 21.7% event, over 24 NASDAQ hours a 0.4% one), so
        # inheriting a default from elsewhere is exactly the mistake worth preventing.
        oc = meta.get('outcome')
        if not oc or not oc.get('name'):
            raise ValueError(
                f"workspace '{name}': universe.json meta has no 'outcome' block.\n"
                f"  Add one, e.g.  \"outcome\": {{\"name\": \"drawdown\", \"params\": {{\"threshold\": 0.10}}}}\n"
                f"  Registered outcomes: {', '.join(outcomes.available())}"
            )
        self.set_outcome(oc['name'], oc.get('params') or {})

    def set_outcome(self, name: str, params: dict | None = None) -> None:
        """Point the workspace at a different barrier (used by --outcome / --threshold)."""
        self.outcome        = name
        self.outcome_params = params or {}
        self.outcome_spec   = outcomes.get(name)
        self.event          = outcomes.event_label(name, self.outcome_params)
        self.outcome_expr   = outcomes.describe(name, self.outcome_params)
        self.outcome_sign   = self.outcome_spec.get('sign', -1)

    @property
    def mirror_outcome(self) -> str | None:
        """The opposite barrier, for the skew readout."""
        return outcomes.mirror(self.outcome)

    def _load_plugin(self) -> None:
        plugin_path = self.dir / 'plugin.py'
        if not plugin_path.exists():
            return
        spec = importlib.util.spec_from_file_location(
            f'_ws_plugin_{self.dir.name}', plugin_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    @property
    def tree_path(self) -> Path:
        return self.dir / 'universe.json'

    # ── on-disk layout ────────────────────────────────────────────────────────
    #
    #   <family>/<event>/<node_id>.xlsx   the surface
    #   <family>/<event>/log.jsonl        run log for that family+event
    #   <family>/<event>/skipped.json     nodes that could not be run, with reasons
    #   validation.<event>.json           null-test results
    #   findings.<event>.json             per-family verdicts
    #
    # Every path is scoped by event, so no two barriers or thresholds collide.

    def output_dir(self, family: str) -> Path:
        return self.dir / family / self.event

    def output_path(self, family: str, node_id: str) -> Path:
        return self.output_dir(family) / f'{node_id}.xlsx'

    def family_log(self, family: str) -> Path:
        return self.output_dir(family) / 'log.jsonl'

    def skip_path(self, family: str) -> Path:
        return self.output_dir(family) / 'skipped.json'

    @property
    def validation_path(self) -> Path:
        return self.dir / f'validation.{self.event}.json'

    @property
    def findings_path(self) -> Path:
        return self.dir / f'findings.{self.event}.json'

    # ── status, derived from disk ─────────────────────────────────────────────

    def is_tested(self, family: str, node_id: str) -> bool:
        return self.output_path(family, node_id).exists()

    def load_skips(self, family: str) -> dict:
        """{node_id: reason} for nodes this workspace could not run under this event."""
        path = self.skip_path(family)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return {}

    def record_skip(self, family: str, node_id: str, reason: str) -> None:
        """
        Remember that a node could not be run, so --next does not retry it forever.

        Recorded per event: a node with too few observations for a 20% barrier may
        be perfectly runnable at 5%.
        """
        skips = self.load_skips(family)
        skips[node_id] = reason
        path = self.skip_path(family)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(skips, indent=2, ensure_ascii=False), encoding='utf-8')

    def clear_skip(self, family: str, node_id: str) -> None:
        skips = self.load_skips(family)
        if skips.pop(node_id, None) is not None:
            self.skip_path(family).write_text(
                json.dumps(skips, indent=2, ensure_ascii=False), encoding='utf-8'
            )

    def node_status(self, family: str, node_id: str) -> str:
        """'tested' / 'skipped' / 'pending', read from disk for the current event."""
        if self.is_tested(family, node_id):
            return 'tested'
        if node_id in self.load_skips(family):
            return 'skipped'
        return 'pending'

    # ── horizons ──────────────────────────────────────────────────────────────

    @property
    def key_horizons(self) -> list[str]:
        return [f'+{h}{self.horizon_unit}' for h in self.horizons]

    @property
    def pivot_horizon(self) -> str:
        h = self.horizons
        return f'+{h[(len(h) - 1) // 2]}{self.horizon_unit}'

    @property
    def display_horizons(self) -> list[str]:
        return [f'+{n}{self.horizon_unit}' for n in self._display_nums]
