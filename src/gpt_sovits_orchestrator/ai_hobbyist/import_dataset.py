from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path

from gpt_sovits_orchestrator.workspace.slug import slices_zip_stem, validate_speaker_slug
from gpt_sovits_orchestrator.config import MANIFEST_FIELDS


@dataclass(frozen=True)
class ImportSummary:
    speaker: str
    zip_path: Path
    manifest_path: Path
    paired: int
    skipped_no_lab: int
    skipped_empty_text: int


def _read_lab_text(lab_path: Path) -> str:
    return lab_path.read_text(encoding="utf-8-sig").strip()


def import_ai_hobbyist_dataset(
    inbox_dir: Path,
    speaker: str,
    *,
    output_zip_dir: Path,
    output_manifest_dir: Path,
    language: str = "ja",
    probability: float = 0.99,
) -> tuple[Path, Path, ImportSummary]:
    """Pack wav+lab pairs into data_02 ZIP and data_03 manifest CSV."""
    slug = validate_speaker_slug(speaker)
    inbox_dir = inbox_dir.resolve()
    output_zip_dir = output_zip_dir.resolve()
    output_manifest_dir = output_manifest_dir.resolve()
    output_zip_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_dir.mkdir(parents=True, exist_ok=True)

    if not inbox_dir.is_dir():
        raise FileNotFoundError(f"AI-Hobbyist inbox not found: {inbox_dir}")

    wav_paths = sorted(inbox_dir.glob("*.wav"), key=lambda path: path.name.lower())
    if not wav_paths:
        raise ValueError(f"No .wav files in inbox: {inbox_dir}")

    zip_path = output_zip_dir / f"{slices_zip_stem(slug)}.zip"
    manifest_path = output_manifest_dir / f"{slug}_manifest.csv"

    rows: list[dict[str, str | float]] = []
    skipped_no_lab = 0
    skipped_empty_text = 0

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for wav_path in wav_paths:
            lab_path = wav_path.with_suffix(".lab")
            if not lab_path.is_file():
                skipped_no_lab += 1
                print(f"SKIP (no .lab) {wav_path.name}")
                continue

            text = _read_lab_text(lab_path)
            if not text:
                skipped_empty_text += 1
                print(f"SKIP (empty .lab) {wav_path.name}")
                continue

            archive.write(wav_path, arcname=wav_path.name)
            rows.append(
                {
                    "filename": wav_path.name,
                    "speaker": slug,
                    "language": language,
                    "text": text,
                    "probability": probability,
                }
            )
            print(f"OK   {wav_path.name}")

    if not rows:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"No wav+lab pairs imported from {inbox_dir} "
            f"(skipped_no_lab={skipped_no_lab}, skipped_empty_text={skipped_empty_text})"
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = ImportSummary(
        speaker=slug,
        zip_path=zip_path,
        manifest_path=manifest_path,
        paired=len(rows),
        skipped_no_lab=skipped_no_lab,
        skipped_empty_text=skipped_empty_text,
    )
    print(
        f"AI-Hobbyist import: speaker={slug} paired={summary.paired} "
        f"skipped_no_lab={skipped_no_lab} skipped_empty_text={skipped_empty_text}"
    )
    print(f"Saved slice ZIP: {zip_path}")
    print(f"Saved manifest CSV: {manifest_path}")
    return zip_path, manifest_path, summary
