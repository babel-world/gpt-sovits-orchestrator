from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SLICE_CHUNK_FILENAME_RE = re.compile(
    r"^(.+)_(\d{4})_(\d{10})-(\d{10})\.wav$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SliceChunkFilenameParts:
    filename: str
    base_name: str
    chunk_index: int
    start_sample: int
    end_sample: int


def parse_slice_chunk_filename(filename: str) -> SliceChunkFilenameParts:
    name = Path(filename).name
    match = SLICE_CHUNK_FILENAME_RE.match(name)
    if not match:
        raise ValueError(
            f"Filename does not match slice convention: {name!r}, "
            "expected {base_name}_{chunk_index:04d}_{start:010d}-{end:010d}.wav"
        )
    base_name, idx, start, end = match.groups()
    return SliceChunkFilenameParts(
        filename=name,
        base_name=base_name,
        chunk_index=int(idx, 10),
        start_sample=int(start, 10),
        end_sample=int(end, 10),
    )
