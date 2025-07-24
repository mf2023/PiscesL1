

Pisces L1
=====

Pisces L1 是一个基于深度学习的多模态模型训练和推理框架，支持从本地缓存加载数据，适用于多种硬件环境，包含一系列高效训练和推理工具。

🚀 特性
-----
- 支持多模态输入（文本、图像、音频、文档）
- 提供不同规模的模型配置（0.5B 到 70B）
- 支持 QLoRA 训练，可在 24GB GPU 上进行 70B 模型训练
- 提供模型量化、设备设置、数据预处理等工具

🛠️ 安装与环境
-----
### 快速搭建
1. 克隆仓库：
   ```bash
   git clone https://gitee.com/dunimd/piscesl1.git
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 下载数据集：
   ```bash
   python tools/arrow.py
   ```

⚡ 命令行使用
-----
### 主要命令
- 启动训练：
  ```bash
  python tools/train.py --config configs/70B.json
  ```
- 推理：
  ```bash
  python tools/infer.py --checkpoint path_to_ckpt
  ```
- 模型量化：
  ```bash
  python tools/quantize.py --checkpoint path_to_ckpt --save_path quantized_ckpt
  ```

🧠 模型架构与配置
-----
Pisces L1 的模型架构包含以下关键组件：
- **PiscesConfig**: 模型配置类
- **PiscesModel**: 主模型结构
- **RMSNorm**: 归一化层
- **RotaryEmbedding**: RoPE 位置嵌入
- **YaRNRotaryEmbedding**: Yarn RoPE 位置嵌入扩展
- **VisionEncoder, AudioEncoder, DocEncoder**: 多模态编码器
- **MoELayer, DynamicMoELayer**: 混合专家系统模块

📦 数据集
-----
- 数据加载由 `PiscesDataset` 类实现
- 提供数据下载工具：`download_datasets`
- 支持数据集分割：`build_splits`

🏆 Extreme 70B 训练（24GB GPU）
-----
### 70B QLoRA 训练示例
使用 QLoRA 技术可在 24GB GPU 上训练 70B 大模型，具体步骤：
1. 设置量化配置
2. 加载预训练模型
3. 使用 `train.py` 开始训练

❓ FAQ
-----
- **如何选择合适的模型配置？** 根据硬件资源选择合适的 `.json` 配置文件（如 70B 需要分布式或量化设置）
- **是否支持多 GPU？** 是，可通过配置实现多 GPU 或混合精度训练
- **如何扩展支持新的模态？** 继承 `VisionEncoder`, `AudioEncoder` 等类并实现新模态编码器

📄 许可证
-----
本项目遵循开源许可证（具体请查看 LICENSE 文件）

🌏 社区与引用
-----
如果使用本项目进行研究或开发，请适当引用我们的工作。

欢迎加入我们的社区，一起推动多模态大模型的发展！