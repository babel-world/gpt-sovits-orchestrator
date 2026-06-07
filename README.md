# GPT-SoVITS Orchestrator

A personal project for orchestrating GPT-SoVITS workflows with [Prefect](https://www.prefect.io/). It aims to deconstruct GPT-SoVITS's opaque pipeline into explicit, manageable steps.

## Quick Start

```bash
cp .env.example .env
uv sync
uv run gpt-sovits-orchestrator
```

可调参数见根目录 [`.env.example`](.env.example)（仅 orchestrator；默认值为本地 smoke 验收配置）。
