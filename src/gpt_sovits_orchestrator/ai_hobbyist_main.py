from __future__ import annotations

import sys
from pathlib import Path

from gpt_sovits_orchestrator.ai_hobbyist.flow import ai_hobbyist_orchestrator_flow


def _print_results(
    g2p_csv_path: Path,
    hubert_paths: tuple[Path, Path],
    sv_paths: tuple[Path, Path],
    semantic_paths: tuple[Path, Path],
    wav32k_zip: Path,
    assemble_csv: Path,
) -> None:
    hubert_in_npz, hubert_out_npz = hubert_paths
    sv_in_npz, sv_out_npz = sv_paths
    semantic_in_npz, semantic_out_npz = semantic_paths
    print(f"data_04 ready: {g2p_csv_path}")
    print(f"data_05 ready: {hubert_in_npz}")
    print(f"data_05 ready: {hubert_out_npz}")
    print(f"data_06 ready: {sv_in_npz}")
    print(f"data_06 ready: {sv_out_npz}")
    print(f"data_07 ready: {semantic_in_npz}")
    print(f"data_07 ready: {semantic_out_npz}")
    print(f"data_08 ready: {wav32k_zip}")
    print(f"data_09 ready: {assemble_csv}")


def main() -> None:
    speaker = sys.argv[1] if len(sys.argv) > 1 else None
    results = ai_hobbyist_orchestrator_flow(speaker=speaker)
    _print_results(*results)


if __name__ == "__main__":
    main()
