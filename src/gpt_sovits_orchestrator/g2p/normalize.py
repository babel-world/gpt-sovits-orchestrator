import re

from gpt_sovits_orchestrator.g2p.constants.symbols2 import punctuation


def normalize_japanese(text: str) -> str:
    """Normalize Japanese text before G2P (at minimum: dedupe consecutive punctuation)."""
    punctuations = "".join(re.escape(symbol) for symbol in punctuation)
    pattern = rf"([{punctuations}])([{punctuations}])+"
    return re.sub(pattern, r"\1", text)
