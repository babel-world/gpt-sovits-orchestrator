# GPT-SoVITS Orchestrator

A personal project for orchestrating GPT-SoVITS workflows with [Prefect](https://www.prefect.io/). It deconstructs GPT-SoVITS's opaque pipeline into explicit, manageable steps.

[简体中文](README.zh-Hans.md)

## Quick Start

```bash
cp .env.example .env
uv sync

# Full pipeline: data/02–09 in-process, then training (data/10) via subprocess
uv run gpt-sovits-orchestrator dry-vocal manbo
uv run gpt-sovits-orchestrator ai-hobbyist jinxi
```

Workspaces: `.local/{pipeline}/{speaker}/inbox` and `data/02 … data/10`.

Configurable parameters are in the root [`.env.example`](.env.example) (shared by orchestrator and trainer).

## Documentation

- [Data pipeline](docs/data-pipeline.md) — APIs, artifacts, and naming conventions for each stage from source through train
