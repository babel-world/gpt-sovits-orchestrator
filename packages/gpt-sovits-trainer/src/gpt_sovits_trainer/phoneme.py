"""Map pre-g2p phoneme tokens from data_09 manifest to model integer IDs."""

from __future__ import annotations

import os

from gpt_sovits_trainer.vendor_env import bootstrap_vendor

_SYMBOL_TO_ID: dict[str, int] | None = None


def _load_symbol_table() -> dict[str, int]:
    global _SYMBOL_TO_ID
    if _SYMBOL_TO_ID is not None:
        return _SYMBOL_TO_ID

    bootstrap_vendor()
    from text import symbols2

    _SYMBOL_TO_ID = {symbol: idx for idx, symbol in enumerate(symbols2.symbols)}
    return _SYMBOL_TO_ID


def phones_to_ids(phones: list[str], *, version: str | None = None) -> list[int]:
    """Convert space-separated phoneme tokens to integer IDs (v2Pro uses symbols2)."""
    if version is None:
        version = os.environ.get("version", "v2Pro")
    if version == "v1":
        bootstrap_vendor()
        from text import symbols

        table = {symbol: idx for idx, symbol in enumerate(symbols.symbols)}
    else:
        table = _load_symbol_table()
    return [table[symbol] for symbol in phones]
