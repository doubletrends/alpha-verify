"""Helpers for traversing a workspace's ``universe.json`` declaration."""

import json
from pathlib import Path


def load_universe(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def all_nodes(universe: dict) -> list[dict]:
    return [n for nodes in universe['families'].values() for n in nodes]


def find_node(universe: dict, node_id: str) -> dict:
    for node in all_nodes(universe):
        if node['id'] == node_id:
            return node
    raise ValueError(f"Node '{node_id}' not found in universe.json")


def all_in_family(universe: dict, family: str) -> list[dict]:
    return universe['families'].get(family, [])
