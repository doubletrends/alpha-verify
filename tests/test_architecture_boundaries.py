from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1] / "src"


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
    return roots


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_domain_has_no_outward_layer_dependencies(self) -> None:
        forbidden = {"infrastructure", "pipeline", "presentation"}
        for path in (ROOT / "domain").glob("*.py"):
            self.assertFalse(imported_roots(path) & forbidden, path)

    def test_presentation_does_not_depend_on_pipeline_or_cli(self) -> None:
        forbidden = {"cli", "pipeline"}
        for path in (ROOT / "presentation").rglob("*.py"):
            self.assertFalse(imported_roots(path) & forbidden, path)

    def test_validation_consumes_selected_artifacts_not_shift_artifacts(self) -> None:
        source = (ROOT / "pipeline" / "step_04_validation.py").read_text(encoding="utf-8")
        self.assertIn("selection_array_path", source)
        self.assertNotIn("shift_cube_path", source)


if __name__ == "__main__":
    unittest.main()
