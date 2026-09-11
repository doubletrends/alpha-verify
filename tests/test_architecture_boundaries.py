from __future__ import annotations

import ast
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "alphaverify"


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

    def test_domain_has_no_outward_layer_dependencies(self) -> None:
        forbidden = {"infrastructure", "pipeline", "presentation"}
        for path in (PACKAGE_ROOT / "domain").glob("*.py"):
            self.assertFalse(imported_layers(path) & forbidden, path)

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

    def test_presentation_does_not_depend_on_pipeline_or_cli(self) -> None:
        forbidden = {"cli", "pipeline"}
        for path in (PACKAGE_ROOT / "presentation").rglob("*.py"):
            self.assertFalse(imported_layers(path) & forbidden, path)

    def test_validation_consumes_shift_artifacts_without_selection(self) -> None:
        source = (
            PACKAGE_ROOT / "pipeline" / "step_03_validation.py"
        ).read_text(encoding="utf-8")
        self.assertIn("shift_cube_path", source)
        self.assertNotIn("selection_array_path", source)


if __name__ == "__main__":
    unittest.main()
