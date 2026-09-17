"""Canonical mathematical names shared across tensor boundaries."""

from __future__ import annotations

from alphaverify.domain.notation import OhlcvComponent


def test_ohlcv_component_order_is_explicit() -> None:
    assert tuple(component.name.lower() for component in OhlcvComponent) == (
        "open", "high", "low", "close", "volume",
    )
