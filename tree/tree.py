"""
Node traversal over universe.json.

universe.json is a pure declaration and is never written back: a node's progress is
derived from the filesystem instead (see Workspace.node_status), so the status-based
helpers this module used to carry — and save_tree with them — are gone.
"""

import json
from pathlib import Path


def load_tree(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def all_nodes(tree: dict) -> list[dict]:
    return [n for nodes in tree['families'].values() for n in nodes]


def find_node(tree: dict, node_id: str) -> dict:
    for node in all_nodes(tree):
        if node['id'] == node_id:
            return node
    raise ValueError(f"Node '{node_id}' not found in universe.json")


def all_in_family(tree: dict, family: str) -> list[dict]:
    return tree['families'].get(family, [])
