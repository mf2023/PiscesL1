# Pisces L1

A 0.5B multimodal MoE model supporting text/image/audio/document, can run inference on single RTX 4090.

## Features

- **Multimodal**: Text, image, audio, and document understanding
- **MoE Architecture**: Mixture of Experts for efficient scaling
- **Lightweight**: 0.5B parameters, suitable for consumer GPUs
- **Modern**: RMSNorm, RoPE, Grouped-Query Attention

## Quick Start

### GPU Check

Before training, check your GPU status:
```bash
python tools/check_gpu.py
```

### Installation & Usage (One Command)

```bash
# 1. Setup environment and install all dependencies (auto venv)
python run.py setup

# 2. Start training (auto-downloads dataset)
python run.py

# 3. Inference
python run.py infer --ckpt ckpt/pisces_tiny_stories_epoch1.pt --prompt "Hello"
```

**Note**: No need to run any shell or PowerShell scripts. All environment setup and dependency installation are handled by `python run.py setup`.

**依赖兼容性说明**：
> Pisces 现已全面适配最新版依赖（见 requirements.txt），所有依赖均为 2024 年最新稳定版，代码已针对新版本做兼容性修正，无需担心 ImportError 或依赖冲突，可直接一键安装和运行。

## Model Architecture

- **Base Model**: 0.5B parameters, 24 layers
- **MoE**: 64 experts, top-2 routing
- **Attention**: 16 heads, 4 KV heads (GQA)
- **Context**: 8K tokens
- **Multimodal**: CLIP (vision), AST (audio), LayoutLMv3 (documents)

## Datasets

- **TinyStories**: Text generation
- **COCO Few-shot**: Image-text few-shot learning
- **AudioSet Captions**: Audio caption generation
