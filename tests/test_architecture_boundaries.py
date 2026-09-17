from __future__ import annotations

import ast
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "alphaverify"
# Each layer may import only the layers listed here; everything else points upward or sideways.
ALLOWED_LAYER_DEPENDENCIES = {
    "cli": {"pipeline", "presentation", "infrastructure", "domain"},
    "pipeline": {"presentation", "infrastructure", "domain"},
    "presentation": {"domain"},
    "infrastructure": set(),
    "domain": set(),
}


def imported_layers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    layers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            continue

        for module in modules:
            parts = module.split(".")
            if parts[0] == "alphaverify" and len(parts) > 1:
                layers.add(parts[1])
            else:
                layers.add(parts[0])
    return layers


def module_name(path: Path) -> str:
    parts = ("alphaverify",) + path.relative_to(PACKAGE_ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def module_graph() -> dict[str, set[str]]:
    """Internal import edges between modules, including deferred imports."""
    modules = {module_name(path): path for path in PACKAGE_ROOT.rglob("*.py")}
    graph = {}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        targets = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    submodule = f"{node.module}.{alias.name}"
                    targets.add(submodule if submodule in modules else node.module)
            elif isinstance(node, ast.Import):
                targets.update(alias.name for alias in node.names)
        graph[name] = {target for target in targets if target in modules and target != name}
    return graph


def layer(module: str) -> str:
    """The top-level package below ``alphaverify``; empty for the root package itself."""
    return module.partition(".")[2].split(".")[0]


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_core_has_no_provider_implementations(self) -> None:
        forbidden = {"yfinance", "urllib", "requests", "httpx"}
        for path in PACKAGE_ROOT.rglob("*.py"):
            self.assertFalse(imported_layers(path) & forbidden, path)

    def test_internal_imports_use_the_alphaverify_namespace(self) -> None:
        legacy_roots = {"domain", "infrastructure", "pipeline", "presentation"}
        violations = []
        for path in PACKAGE_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    if module.split(".", 1)[0] in legacy_roots:
                        violations.append((path, node.lineno, module))
        self.assertEqual(violations, [])

    def test_infrastructure_and_presentation_are_flat_packages(self) -> None:
        for package in (
            PACKAGE_ROOT / "infrastructure",
            PACKAGE_ROOT / "presentation",
        ):
            nested_modules = [path for path in package.rglob("*.py") if path.parent != package]
            self.assertEqual(nested_modules, [], package)

    def test_layers_depend_only_on_declared_lower_layers(self) -> None:
        violations = [
            (module, target)
            for module, targets in module_graph().items()
            for target in targets
            if layer(module) != layer(target)
            and layer(target) not in ALLOWED_LAYER_DEPENDENCIES[layer(module)]
        ]
        self.assertEqual(violations, [])

    def test_module_graph_has_no_cycles(self) -> None:
        graph = module_graph()
        finished, active = set(), []

        def visit(module: str) -> None:
            if module in active:
                self.fail(" -> ".join(active[active.index(module):] + [module]))
            if module in finished:
                return
            active.append(module)
            for target in sorted(graph[module]):
                visit(target)
            active.pop()
            finished.add(module)

        for module in sorted(graph):
            visit(module)

    def test_stages_import_only_earlier_stages(self) -> None:
        def stage(module: str) -> int | None:
            leaf = module.rsplit(".", 1)[-1]
            return int(leaf[5:7]) if leaf.startswith("step_") else None

        violations = [
            (module, target)
            for module, targets in module_graph().items()
            if layer(module) == "pipeline"
            for target in targets
            if stage(target) is not None
            and (stage(module) is None or stage(target) >= stage(module))
        ]
        self.assertEqual(violations, [])

    def test_domain_has_no_persistence_dependencies_or_entry_points(self) -> None:
        forbidden_dependencies = {"datetime", "json", "pathlib"}
        forbidden_functions = {
            "load",
            "load_cube",
            "load_selected_node",
            "save",
            "save_cube",
            "save_selected_node",
        }
        for path in (PACKAGE_ROOT / "domain").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            public_functions = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            self.assertFalse(imported_layers(path) & forbidden_dependencies, path)
            self.assertFalse(public_functions & forbidden_functions, path)

    def test_npz_persistence_is_owned_by_infrastructure(self) -> None:
        owner = PACKAGE_ROOT / "infrastructure" / "artifact_io.py"
        violations = []
        for path in PACKAGE_ROOT.rglob("*.py"):
            if path == owner:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "np"
                    and node.func.attr in {"load", "save", "savez", "savez_compressed"}
                ):
                    violations.append((path, node.lineno, node.func.attr))
        self.assertEqual(violations, [])

    def test_validation_consumes_shift_artifacts_without_selection(self) -> None:
        source = (
            PACKAGE_ROOT / "pipeline" / "step_03_validation.py"
        ).read_text(encoding="utf-8")
        self.assertIn("shift_cube_path", source)
        self.assertNotIn("selection_array_path", source)


if __name__ == "__main__":
    unittest.main()
