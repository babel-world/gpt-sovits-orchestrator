from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from gpt_sovits_orchestrator.assemble.paths import assemble_manifest_path
from gpt_sovits_orchestrator.assemble.sources import (
    load_npz_keys,
    load_wav32k_entry_sizes,
    modality_audit,
)
from gpt_sovits_orchestrator.config import ASSEMBLE_OUTPUT_COLUMNS, G2P_OUTPUT_COLUMNS
from gpt_sovits_orchestrator.hubert.npz import hubert_npz_paths
from gpt_sovits_orchestrator.semantic.npz import semantic_npz_paths
from gpt_sovits_orchestrator.sv.npz import sv_npz_paths
from gpt_sovits_orchestrator.wav32k.zip_paths import wav32k_zip_path


def _read_g2p_rows(g2p_csv_path: Path) -> list[dict[str, str]]:
    text = g2p_csv_path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError(f"Empty G2P CSV: {g2p_csv_path}")
    if tuple(reader.fieldnames) != G2P_OUTPUT_COLUMNS:
        raise ValueError(
            f"Unexpected G2P header in {g2p_csv_path}: {reader.fieldnames!r}"
        )
    return list(reader)


def compute_eligible_filenames(
    g2p_rows: list[dict[str, str]],
    *,
    hubert_keys: set[str],
    sv_keys: set[str],
    semantic_keys: set[str],
    wav32k_keys: set[str],
) -> list[str]:
    ok_filenames = {row["filename"] for row in g2p_rows if row.get("status") == "ok"}
    eligible = ok_filenames & hubert_keys & sv_keys & semantic_keys & wav32k_keys
    return sorted(eligible)


def build_assemble_manifest(
    zip_path: Path,
    g2p_csv_path: Path,
    output_dir: Path,
    *,
    hubert_data_dir: Path,
    sv_data_dir: Path,
    semantic_data_dir: Path,
    wav32k_data_dir: Path,
) -> tuple[Path, int]:
    zip_path = zip_path.resolve()
    g2p_csv_path = g2p_csv_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    g2p_rows = _read_g2p_rows(g2p_csv_path)
    rows_by_filename = {row["filename"]: row for row in g2p_rows}

    _, hubert_out_npz = hubert_npz_paths(zip_path, hubert_data_dir)
    _, sv_out_npz = sv_npz_paths(zip_path, sv_data_dir)
    _, semantic_out_npz = semantic_npz_paths(hubert_out_npz, semantic_data_dir)
    wav32k_zip = wav32k_zip_path(zip_path, wav32k_data_dir)

    hubert_keys = load_npz_keys(hubert_out_npz)
    sv_keys = load_npz_keys(sv_out_npz)
    semantic_keys = load_npz_keys(semantic_out_npz)
    wav32k_sizes = load_wav32k_entry_sizes(wav32k_zip)
    wav32k_keys = set(wav32k_sizes)

    eligible = compute_eligible_filenames(
        g2p_rows,
        hubert_keys=hubert_keys,
        sv_keys=sv_keys,
        semantic_keys=semantic_keys,
        wav32k_keys=wav32k_keys,
    )
    if not eligible:
        raise RuntimeError("Assemble intersection is empty")

    output_path = assemble_manifest_path(g2p_csv_path, output_dir)
    assembled_rows: list[dict[str, str | int]] = []

    with np.load(hubert_out_npz) as hubert_out, np.load(sv_out_npz) as sv_out, np.load(
        semantic_out_npz
    ) as semantic_out:
        for filename in eligible:
            base_row = rows_by_filename[filename]
            audit = modality_audit(
                filename,
                hubert_out=hubert_out,
                semantic_out=semantic_out,
                sv_out=sv_out,
                wav32k_sizes=wav32k_sizes,
            )
            assembled_rows.append(
                {
                    **{col: base_row[col] for col in G2P_OUTPUT_COLUMNS},
                    **audit,
                    "eligible_gpt": "true",
                    "eligible_sovits": "true",
                }
            )

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=ASSEMBLE_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(assembled_rows)

    return output_path, len(assembled_rows)
