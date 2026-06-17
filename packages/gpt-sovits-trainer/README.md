# GPT-SoVITS Trainer

独立 Python 3.10 子项目：s1/s2 训练黑盒。通常由 `gpt-sovits-orchestrator` 在 data/09 完成后 **subprocess** 调用，无需手动运行。

**环境变量**：与 orchestrator 共用仓库根目录 [`.env`](../../.env.example)。

## 前置

1. orchestrator 已产出工作区 `data/09`（或运行完整 `gpt-sovits-orchestrator {pipeline} {speaker}`）
2. 底模：`uv run gpt-sovits-download-models`
3. NVIDIA GPU + CUDA（推荐）

## 单独运行（重训 / 调试）

```powershell
cd packages/gpt-sovits-trainer
uv sync
uv run gpt-sovits-trainer dry-vocal manbo
```

`pipeline` / `speaker` 与 orchestrator 相同；工作区自动推导为 `.local/dry-vocal/manbo/data/`。  
epoch、batch 等超参读根 `.env` 的 `TRAINER_*`。

## 代码分层

| 层 | 路径 | 说明 |
|----|------|------|
| 训练入口 | `src/gpt_sovits_trainer/main.py` | CLI + trainer_flow |
| s1 / s2 | `train_s1.py`, `train_s2.py` | 训练循环 |
| 数据 | `datasets/` | assemble manifest + NPZ/ZIP |
| 模型定义 | `vendor/GPT_SoVITS/` | 精简 vendor |

详见 [vendor/README.md](vendor/README.md)。
