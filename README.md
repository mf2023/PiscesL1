# Pisces L1

A 0.5B multimodal MoE model supporting text/image/audio/document, can run inference on single RTX 4090.

## Features

- **Multimodal**: Text, image, audio, and document understanding
- **MoE Architecture**: Mixture of Experts for efficient scaling
- **Lightweight**: 0.5B parameters, suitable for consumer GPUs
- **Modern**: RMSNorm, RoPE, Grouped-Query Attention

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Inference
```bash
python scripts/infer_demo.py --ckpt path/to/checkpoint.pt --prompt "Hello world"
```

### Training
```bash
# Download datasets first
python data/download.py

# Start training
python trainer/train.py --config configs/0.5B.json --dataset tiny
```

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
