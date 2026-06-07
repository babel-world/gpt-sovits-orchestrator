"""Compute and optionally delete vendor files not reachable from s1/s2 training seeds."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "GPT_SoVITS"

SEEDS = [
    VENDOR / "AR/models/t2s_model.py",
    VENDOR / "AR/modules/optim.py",
    VENDOR / "AR/modules/lr_schedulers.py",
    VENDOR / "AR/data/bucket_sampler.py",
    VENDOR / "module/models.py",
    VENDOR / "module/losses.py",
    VENDOR / "module/mel_processing.py",
    VENDOR / "module/commons.py",
    VENDOR / "text/symbols.py",
    VENDOR / "text/symbols2.py",
]

KEEP_DATA = {
    VENDOR / "configs/s2v2Pro.json",
    VENDOR / "configs/.gitignore",
}

LOCAL_PREFIXES = ("AR.", "module.", "text.", "f5_tts.", "GPT_SoVITS.")


def module_to_path(module: str) -> Path | None:
    if module.startswith("GPT_SoVITS."):
        module = module[len("GPT_SoVITS.") :]
    if not any(module.startswith(prefix) for prefix in ("AR.", "module.", "text.", "f5_tts.")):
        if not module.startswith("GPT_SoVITS."):
            return None
    parts = module.split(".")
    candidate = VENDOR.joinpath(*parts)
    if candidate.with_suffix(".py").is_file():
        return candidate.with_suffix(".py")
    init = candidate / "__init__.py"
    if init.is_file():
        return init
    return None


def imports_in_file(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    found: set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def collect_reachable() -> set[Path]:
    queue = [p for p in SEEDS if p.is_file()]
    seen: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for mod in imports_in_path(path):
            child = module_to_path(mod)
            if child and child not in seen:
                queue.append(child)
    return seen | KEEP_DATA


def imports_in_path(path: Path) -> set[str]:
    return imports_in_file(path)


def main() -> int:
    reachable = collect_reachable()
    all_files = {p for p in VENDOR.rglob("*") if p.is_file()}
    to_delete = sorted(all_files - reachable)

    print(f"reachable: {len(reachable)}")
    print(f"delete: {len(to_delete)}")
    for p in sorted(reachable):
        print(f"  keep {p.relative_to(VENDOR)}")

    if "--apply" in sys.argv:
        vendor_root = ROOT / "vendor"
        if (vendor_root / "tools").exists():
            import shutil

            shutil.rmtree(vendor_root / "tools", ignore_errors=True)
            print("deleted vendor/tools")
        for p in to_delete:
            p.unlink()
            print(f"deleted {p.relative_to(VENDOR)}")
        for d in sorted({p.parent for p in VENDOR.rglob("*") if p.is_dir()}, reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
