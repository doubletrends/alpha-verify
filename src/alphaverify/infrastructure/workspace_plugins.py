"""Loading boundary for workspace-local registrations."""

from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys
from types import ModuleType


def load_workspace_module(workspace_dir: Path, name: str, *, required=False):
    """Load local code with relative imports isolated by absolute workspace root."""
    workspace_dir = workspace_dir.resolve()
    path = workspace_dir / f"{name}.py"
    if not path.exists():
        if required:
            raise ValueError(f"workspace must provide {path.name}: {path}")
        return None
    root_name = "_alphaverify_workspaces_" + hashlib.sha256(
        str(workspace_dir.parent).encode()
    ).hexdigest()[:16]
    package_name = root_name + "." + workspace_dir.name
    for package, directory in ((root_name, workspace_dir.parent), (package_name, workspace_dir)):
        if package not in sys.modules:
            module = ModuleType(package)
            module.__path__ = [str(directory)]
            module.__package__ = package
            sys.modules[package] = module
    module_name = package_name + "." + name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load workspace module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_workspace_plugin(workspace_dir: Path, features) -> None:
    """Register optional workspace features; data preparation lives in data.py."""
    module = load_workspace_module(workspace_dir, "plugin")
    if module is None:
        return
    register = getattr(module, "register", None)
    if register is None:
        raise ValueError(f"workspace plugin must define register(features): {workspace_dir}")
    register(features)
