"""Japanese text to VITS-compatible phoneme pipeline."""

from __future__ import annotations

import re
from collections.abc import Callable

from gpt_sovits_orchestrator.g2p.utils.symbol_alignment import align_to_vits_symbols

_japanese_characters = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)
_japanese_marks = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)


def text_to_vits_phones(text: str, g2p_fn: Callable[[str], list[str]]) -> list[str]:
    """Split, stitch, and align Japanese text into VITS-compatible phonemes."""
    text = text.lower()

    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)

    raw_phones: list[str] = []

    for index, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            raw_phones.extend(g2p_fn(sentence))

        if index < len(marks):
            if marks[index] == " ":
                continue
            raw_phones.append(marks[index].replace(" ", ""))

    return align_to_vits_symbols(raw_phones)
