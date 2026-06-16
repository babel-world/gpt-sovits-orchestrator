from __future__ import annotations

import zipfile
from pathlib import Path

from prefect import task

from gpt_sovits_orchestrator.wav32k.preprocess import (
    int16_array_to_wav_bytes,
    wav_bytes_to_wav32k_int16,
)
from gpt_sovits_orchestrator.wav32k.zip_paths import wav32k_zip_path


def _list_wav_entries(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".wav") and not name.endswith("/")
        ]
    if not names:
        raise ValueError(f"No WAV entries found in ZIP: {zip_path}")
    return sorted(names, key=lambda name: Path(name).name)


@task(name="wav32k-from-zip", log_prints=True)
def wav32k_from_zip(
    zip_path: Path,
    output_dir: Path,
) -> Path:
    """Convert slice WAV entries to 32 kHz int16 target audio ZIP (SoVITS s2 / 5-wav32k)."""
    zip_path = zip_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.is_file():
        raise FileNotFoundError(f"Slice ZIP not found: {zip_path}")

    out_zip_path = wav32k_zip_path(zip_path, output_dir)
    wav_names = _list_wav_entries(zip_path)
    written = 0
    skipped = 0
    total = len(wav_names)

    with zipfile.ZipFile(zip_path, "r") as in_archive, zipfile.ZipFile(
        out_zip_path,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as out_archive:
        for index, name in enumerate(wav_names, start=1):
            wav_key = Path(name).name
            target = wav_bytes_to_wav32k_int16(in_archive.read(name), wav_key)
            if target is None:
                skipped += 1
                print(f"SKIP [{index}/{total}] {wav_key}")
                continue
            out_archive.writestr(wav_key, int16_array_to_wav_bytes(target))
            written += 1
            print(f"OK   [{index}/{total}] {wav_key}")

    if written == 0:
        raise RuntimeError(f"No wav32k entries generated from ZIP: {zip_path}")

    print(
        f"Saved wav32k ZIP: {out_zip_path} "
        f"({written} entries, {skipped} skipped)"
    )
    return out_zip_path
