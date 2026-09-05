"""Validated traversal of a workspace's feature declaration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NodeCatalog:
    """The immutable node and family declaration loaded from ``universe.json``."""

    raw: dict

    @classmethod
    def load(cls, path: Path) -> "NodeCatalog":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw.get("meta"), dict) or not isinstance(raw.get("families"), dict):
            raise ValueError(f"invalid workspace declaration: {path}")
        return cls(raw)

    @property
    def meta(self) -> dict:
        return self.raw["meta"]

    def all_nodes(self) -> list[dict]:
        return [node for nodes in self.raw["families"].values() for node in nodes]

    def in_family(self, family: str) -> list[dict]:
        return self.raw["families"].get(family, [])

    def find(self, node_id: str) -> dict:
        for node in self.all_nodes():
            if node["id"] == node_id:
                return node
        raise ValueError(f"Node '{node_id}' not found in universe.json")


def find_node(universe: dict, node_id: str) -> dict:
    """Compatibility helper for report code that consumes a raw declaration."""
    return NodeCatalog(universe).find(node_id)
