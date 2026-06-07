# Sync upstream GPT-SoVITS, then prune to s1/s2 training subset
$ErrorActionPreference = "Stop"
$upstream = if ($env:UPSTREAM_GPT_SOVITS_ROOT) { $env:UPSTREAM_GPT_SOVITS_ROOT } else { "E:\WorkSpace\Projects_GitHub\RVC-Boss\GPT-SoVITS" }
$root = Join-Path $PSScriptRoot ".."
$vendor = Join-Path $root "vendor"
$gpt = Join-Path $vendor "GPT_SoVITS"
New-Item -ItemType Directory -Force -Path $gpt | Out-Null
robocopy (Join-Path $upstream "GPT_SoVITS") $gpt /E /NFL /NDL /NJH /NJS /nc /ns /np
Push-Location $root
uv run python scripts/prune_vendor.py --apply
Pop-Location
Write-Host "vendor synced and pruned from $upstream"
