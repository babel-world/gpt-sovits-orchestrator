# GPT-SoVITS Orchestrator

基于 [Prefect](https://www.prefect.io/) 编排 GPT-SoVITS 工作流的个人项目，将作者原项目中隐式的训练流水线拆解为可管理的显式步骤。

[English](README.md)

## 快速开始

```bash
cp .env.example .env
uv sync

# 一步到位：data/02–09 同进程，data/10 训练 subprocess 调 trainer
uv run gpt-sovits-orchestrator dry-vocal manbo
uv run gpt-sovits-orchestrator ai-hobbyist jinxi
```

工作区：`.local/{pipeline}/{speaker}/inbox` 与 `data/02 … data/10`。

可调参数见根目录 [`.env.example`](.env.example)（orchestrator 与 trainer 共用）。

## 文档

- [数据流水线](docs/data-pipeline.md) — 各阶段 API、产物与命名约定
