from __future__ import annotations

import csv
import io
from pathlib import Path

import httpx
from prefect import task

from gpt_sovits_orchestrator.config import (
    G2P_JA_API_PATH,
    G2P_MODE,
    G2P_OUTPUT_COLUMNS,
    NLP_SERVER_BASE_URL,
)
from gpt_sovits_orchestrator.g2p.manifest import (
    process_manifest_csv,
    summarize_manifest_csv,
)


def _infer_base_name(manifest_path: Path, csv_text: str) -> str:
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        speaker = (row.get("speaker") or "").strip()
        if speaker:
            return speaker
    return manifest_path.stem.removesuffix("_manifest") or "audio"


@task(name="g2p-manifest", log_prints=True)
def g2p_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    base_url: str = NLP_SERVER_BASE_URL,
    timeout: float = 60.0,
) -> Path:
    """Run G2P on manifest CSV rows via ``POST /api/g2p/ja`` and write training CSV."""
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    csv_text = manifest_path.read_text(encoding="utf-8-sig")
    base_name = _infer_base_name(manifest_path, csv_text)
    output_path = output_dir / f"{base_name}_manifest_g2p.csv"

    url = f"{base_url.rstrip('/')}{G2P_JA_API_PATH}"

    with httpx.Client(timeout=timeout) as client:

        def g2p_fn(text: str) -> list[str]:
            response = client.post(url, json={"text": text, "mode": G2P_MODE})
            response.raise_for_status()
            return response.json()["phones"]

        result_csv = process_manifest_csv(csv_text, g2p_fn)

    output_path.write_text(result_csv, encoding="utf-8", newline="\n")

    summary = summarize_manifest_csv(result_csv)
    print(
        f"G2P summary: ok={summary['ok']} skip={summary['skip']} "
        f"error={summary['error']} total={summary['total']}"
    )
    print(f"Saved G2P manifest CSV: {output_path}")

    header = next(csv.reader(io.StringIO(result_csv)))
    if tuple(header) != G2P_OUTPUT_COLUMNS:
        raise ValueError(
            f"Unexpected CSV header: {header!r}, expected {G2P_OUTPUT_COLUMNS!r}"
        )

    return output_path
