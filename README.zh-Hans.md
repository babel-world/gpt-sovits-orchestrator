# GPT-SoVITS Orchestrator

基于 [Prefect](https://www.prefect.io/) 编排 GPT-SoVITS 工作流的个人项目，将作者原项目中隐式的训练流水线拆解为可管理的显式步骤。

[English](README.md)

## 快速开始

```bash
cp .env.example .env
uv sync
uv run gpt-sovits-orchestrator
```

可调参数见根目录 [`.env.example`](.env.example)（仅 orchestrator；默认值为本地 smoke 验收配置）。

## 文档

- [数据流水线](docs/data-pipeline.md) — source → slice → … → train 各阶段 API、产物与命名约定
