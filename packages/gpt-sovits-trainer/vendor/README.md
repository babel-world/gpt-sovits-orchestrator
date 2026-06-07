# vendor / GPT-SoVITS（精简版）

本目录**不是**上游项目的完整拷贝，而是仅保留 s1/s2 训练所需的模型定义与配置。

## 目录结构（约 30 个文件）

| 路径 | 用途 |
|------|------|
| `GPT_SoVITS/AR/` | GPT s1：`Text2SemanticDecoder`、Transformer、ScaledAdam、BucketSampler |
| `GPT_SoVITS/module/` | SoVITS s2：生成器/判别器、VITS 损失、梅尔谱 |
| `GPT_SoVITS/f5_tts/` | v2Pro 生成器中的 DiT 骨干 |
| `GPT_SoVITS/text/symbols*.py` | 音素词表（供 s2 `text_embedding` 维度；训练时 phoneme→id 在 `gpt_sovits_trainer.phoneme`） |
| `GPT_SoVITS/configs/s2v2Pro.json` | s2 默认超参 |

## 不在 vendor 内的逻辑（本仓库自研）

| 模块 | 说明 |
|------|------|
| `src/gpt_sovits_trainer/train_s1.py` | s1 原生 PyTorch 训练循环 |
| `src/gpt_sovits_trainer/train_s2.py` | s2 GAN 训练循环 |
| `src/gpt_sovits_trainer/datasets/` | 读取 data_09 + NPZ/ZIP 的 Dataset / Collate / Sampler |
| `src/gpt_sovits_trainer/phoneme.py` | manifest `phones` → 整数 ID（不再依赖 g2p/归一化） |

## 从上游重新同步

若需对齐上游新版本：

```powershell
powershell -File scripts/sync_vendor.ps1
uv run python scripts/prune_vendor.py --apply
```

`prune_vendor.py` 会从 s1/s2 训练入口做 import 可达性分析，删除 inference、BigVGAN、prepare_datasets、tools 等无关文件。

## 相对上游的微调

| 文件 | 变更 |
|------|------|
| `f5_tts/model/__init__.py` | `GPT_SoVITS.f5_tts...` → `f5_tts...` |
| `f5_tts/model/backbones/dit.py` | 同上 |

其余保留文件与 upstream 一致（仅裁剪，不改算法）。
