"""Loading boundary for workspace-local registrations."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_workspace_plugin(workspace_dir: Path, sources, features) -> None:
    """Load one workspace plugin and register its explicit extensions."""
    path = workspace_dir / "plugin.py"
    if not path.exists():
        return
    spec = importlib.util.spec_from_file_location(f"_workspace_plugin_{workspace_dir.name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load workspace plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    register = getattr(module, "register", None)
    if register is None:
        raise ValueError(f"workspace plugin must define register(sources, features): {path}")
    register(sources, features)
