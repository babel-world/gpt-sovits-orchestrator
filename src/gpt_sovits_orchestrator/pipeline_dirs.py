from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineStageDirs:
    """Per-stage output directories for orchestrator data_02 … data_10."""

    data_02: Path
    data_03: Path
    data_04: Path
    data_05: Path
    data_06: Path
    data_07: Path
    data_08: Path
    data_09: Path
    data_10: Path

    @classmethod
    def under_workspace_data(cls, workspace_root: Path) -> PipelineStageDirs:
        data = workspace_root / "data"
        return cls(
            data_02=data / "02",
            data_03=data / "03",
            data_04=data / "04",
            data_05=data / "05",
            data_06=data / "06",
            data_07=data / "07",
            data_08=data / "08",
            data_09=data / "09",
            data_10=data / "10",
        )

    def ensure_exists(self) -> None:
        for path in (
            self.data_02,
            self.data_03,
            self.data_04,
            self.data_05,
            self.data_06,
            self.data_07,
            self.data_08,
            self.data_09,
            self.data_10,
        ):
            path.mkdir(parents=True, exist_ok=True)
