from __future__ import annotations

from unittest.mock import patch

import pytest

from gpt_sovits_orchestrator.main import main as orchestrator_main
from gpt_sovits_orchestrator.trainer_run import run_trainer_subprocess


def test_run_trainer_subprocess_invokes_uv() -> None:
    with patch("gpt_sovits_orchestrator.trainer_run.subprocess.run") as run:
        with patch("gpt_sovits_orchestrator.trainer_run.shutil.which", return_value="uv"):
            run_trainer_subprocess("dry-vocal", "manbo")
    run.assert_called_once()
    cmd = run.call_args.args[0]
    assert cmd[:4] == ["uv", "run", "gpt-sovits-trainer", "dry-vocal"]
    assert cmd[4] == "manbo"


def test_main_runs_trainer_after_orchestrator() -> None:
    fake_results = (
        "g2p.csv",
        ("h_in", "h_out"),
        ("s_in", "s_out"),
        ("m_in", "m_out"),
        "wav32k.zip",
        "assemble.csv",
    )
    with patch("gpt_sovits_orchestrator.main.run_orchestrator_flow", return_value=fake_results):
        with patch("gpt_sovits_orchestrator.main.run_trainer_subprocess") as train:
            orchestrator_main(["dry-vocal", "manbo"])
    train.assert_called_once_with("dry-vocal", "manbo")


def test_main_trainer_only_skips_orchestrator() -> None:
    with patch("gpt_sovits_orchestrator.main.run_orchestrator_flow") as flow:
        with patch("gpt_sovits_orchestrator.main.run_trainer_subprocess") as train:
            orchestrator_main(["dry-vocal", "manbo", "--trainer-only"])
    flow.assert_not_called()
    train.assert_called_once_with("dry-vocal", "manbo")
