"""Copying a shipped workspace declaration into a caller-owned workspaces directory."""

from __future__ import annotations

from pathlib import Path
import shutil

from alphaverify.infrastructure.workspace import WORKSPACES_DIRNAME

# The declaration files a workspace owns; everything else below it is generated.
DECLARATION_FILES = ("universe.json", "data.py", "plugin.py")
SHARED_DIRNAME = "_shared"

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
# An installed wheel carries the declarations; a source checkout uses the live ones.
_BUNDLED_TEMPLATES = _PACKAGE_DIR / "templates"
_CHECKOUT_TEMPLATES = _PACKAGE_DIR.parents[1] / WORKSPACES_DIRNAME


def templates_dir() -> Path:
    """The directory holding the shipped workspace declarations."""
    for candidate in (_BUNDLED_TEMPLATES, _CHECKOUT_TEMPLATES):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"no shipped workspaces alongside the package: {_BUNDLED_TEMPLATES}")


def available_templates() -> list[str]:
    return sorted(
        path.name for path in templates_dir().iterdir()
        if (path / "universe.json").exists()
    )


def copy_workspace(name: str, destination_root: Path) -> list[Path]:
    """Copy one shipped workspace and the ``_shared`` helpers it imports.

    Only declaration files are copied: generated artifacts belong to the run that
    produced them and are never portable between workspaces.
    """
    source = templates_dir() / name
    if not (source / "universe.json").exists():
        raise ValueError(
            f"no shipped workspace named {name!r}; available: {', '.join(available_templates())}"
        )
    target = destination_root / name
    if target.exists():
        raise FileExistsError(f"workspace directory already exists: {target}")

    written = []
    target.mkdir(parents=True)
    for filename in DECLARATION_FILES:
        origin = source / filename
        if origin.exists():
            shutil.copy2(origin, target / filename)
            written.append(target / filename)

    shared_source = templates_dir() / SHARED_DIRNAME
    shared_target = destination_root / SHARED_DIRNAME
    if shared_source.is_dir():
        shared_target.mkdir(exist_ok=True)
        for origin in sorted(shared_source.glob("*.py")):
            copy = shared_target / origin.name
            if not copy.exists():
                shutil.copy2(origin, copy)
                written.append(copy)
    return written
