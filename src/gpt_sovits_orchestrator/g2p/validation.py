from gpt_sovits_orchestrator.g2p.constants.symbols2 import symbols

_SYMBOL_SET = set(symbols)


def find_unknown_symbols(phones: list[str]) -> list[str]:
    """Return symbols that are not in the symbols2 vocabulary."""
    return [phone for phone in phones if phone not in _SYMBOL_SET]
