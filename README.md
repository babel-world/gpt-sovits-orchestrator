# GPT-SoVITS Orchestrator

A personal project for orchestrating GPT-SoVITS workflows with [Prefect](https://www.prefect.io/). It deconstructs GPT-SoVITS's opaque pipeline into explicit, manageable steps.

[简体中文](README.zh-Hans.md)

## Quick Start

```bash
cp .env.example .env
uv sync
uv run gpt-sovits-orchestrator
```

Configurable parameters are in the root [`.env.example`](.env.example) (orchestrator only; defaults target local smoke validation).

## Documentation

- [Data pipeline](docs/data-pipeline.md) — APIs, artifacts, and naming conventions for each stage from source through train
