# GPT-SoVITS Trainer

独立 Python 3.10 子项目：读取仓库根 `.local/data_09` assemble manifest 与 data_05~08 NPZ/ZIP，用**自研训练循环** + **精简 vendor 模型定义**完成 s1/s2 微调，产出写入 `.local/data_10/`。

## 前置

1. orchestrator 已跑通至 `data_09/manbo_manifest_assemble.csv`
2. （可选）从上游重新同步并裁剪 vendor：`powershell -File scripts/sync_vendor.ps1`
3. 下载底模：`uv run gpt-sovits-download-models`
4. NVIDIA GPU + CUDA（推荐；CPU 可 smoke 但极慢）

## 运行

```powershell
cd packages/gpt-sovits-trainer
cp .env.example .env
uv sync
uv run gpt-sovits-trainer
```

训练参数见本目录 [`.env.example`](.env.example)（默认 2+2 epoch smoke；Colab 正式训练见注释）。

验收产物：

- `.local/data_10/manbo.ckpt`（GPT s1）
- `.local/data_10/manbo.pth`（SoVITS s2 v2Pro）

## 代码分层

| 层 | 路径 | 说明 |
|----|------|------|
| 训练入口 | `src/gpt_sovits_trainer/main.py` | Prefect flow |
| s1 循环 | `src/gpt_sovits_trainer/train_s1.py` | 原生 PyTorch + AR 模型 |
| s2 循环 | `src/gpt_sovits_trainer/train_s2.py` | GAN 交替训练 + module 模型 |
| 数据 | `src/gpt_sovits_trainer/datasets/` | assemble manifest + NPZ/ZIP |
| 音素 | `src/gpt_sovits_trainer/phoneme.py` | manifest `phones` → ID |
| 模型定义 | `vendor/GPT_SoVITS/` | **仅** AR / module / f5_tts / symbols（约 30 文件） |
| 底模 | `.models/pretrained/` | HuggingFace 预训练权重 |

详见 [vendor/README.md](vendor/README.md)。
