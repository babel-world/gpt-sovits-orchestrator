from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpt_sovits_orchestrator.ai_hobbyist.flow import ai_hobbyist_orchestrator_flow
from gpt_sovits_orchestrator.dry_vocal.flow import dry_vocal_orchestrator_flow
from gpt_sovits_orchestrator.trainer_run import run_trainer_subprocess
from gpt_sovits_orchestrator.workspace import PIPELINE_AI_HOBBYIST, PIPELINE_CHOICES, PIPELINE_DRY_VOCAL


def _print_data_results(
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpt-sovits-orchestrator",
        description="GPT-SoVITS pipeline: data preparation (data/02–09) then training (data/10)",
    )
    parser.add_argument(
        "pipeline",
        choices=PIPELINE_CHOICES,
        help="pipeline to run: dry-vocal or ai-hobbyist",
    )
    parser.add_argument(
        "speaker",
        help="speaker slug (e.g. manbo, jinxi)",
    )
    parser.add_argument(
        "--trainer-only",
        action="store_true",
        help="skip data/02–09; assume assemble manifest exists and run training only",
    )
    return parser


def run_orchestrator_flow(pipeline: str, speaker: str):
    if pipeline == PIPELINE_DRY_VOCAL:
        return dry_vocal_orchestrator_flow(speaker=speaker)
    if pipeline == PIPELINE_AI_HOBBYIST:
        return ai_hobbyist_orchestrator_flow(speaker=speaker)
    raise ValueError(f"unsupported pipeline: {pipeline!r}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.trainer_only:
        results = run_orchestrator_flow(args.pipeline, args.speaker)
        _print_data_results(*results)
    run_trainer_subprocess(args.pipeline, args.speaker)


if __name__ == "__main__":
    main(sys.argv[1:])
