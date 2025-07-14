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

### Installation

#### Option 1: Using Virtual Environment (Recommended)

**On Windows:**
```powershell
# Run the setup script
.\setup_env.ps1

# Or manually:
python -m venv pisces_env
.\pisces_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

**On Linux/macOS:**
```bash
# Run the setup script
chmod +x setup_env.sh
./setup_env.sh

# Or manually:
python3 -m venv pisces_env
source pisces_env/bin/activate
pip install -r requirements.txt
```

#### Option 2: Direct Installation (Not Recommended)
```bash
pip install -r requirements.txt
```

**Note**: Running pip as root can cause permission issues. Use a virtual environment instead.

### 🚀 One-Click Usage

Pisces provides unified training and inference:

#### Training
```bash
# Start training (auto-downloads dataset)
python run.py train

# Or just
python run.py
```

#### Inference
```bash
# Text inference
python run.py infer --ckpt ckpt/pisces_tiny_stories_epoch1.pt --prompt "Hello"

# Multimodal inference
python run.py infer --ckpt ckpt/pisces_tiny_stories_epoch1.pt --prompt "Describe this image" --image image.jpg
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
