# ch01-02: 环境准备

## 硬件要求

### GPU配置（推荐）

| 配置级别 | GPU显存 | 适用场景 |
|----------|---------|----------|
| 最低 | 8GB | 基础实验、小模型训练 |
| 推荐 | 16GB | 123M模型完整训练 |
| 理想 | 24GB+ | 更大模型、批量训练 |

### 存储空间

- 代码: ~1GB
- 数据集: ~10GB
- 模型checkpoint: ~1GB
- 推荐总空间: 50GB+

## 软件环境

### 基础环境

```bash
# Python版本 (推荐3.10-3.12)
python --version  # 应该是3.10+

# 创建虚拟环境
python -m venv llm_env
source llm_env/bin/activate  # Linux/Mac
# 或
llm_env\Scripts\activate  # Windows
```

### 核心依赖

```bash
# PyTorch (CUDA版本，根据你的CUDA版本选择)
pip install torch==2.10.0

# Transformers (HuggingFace)
pip install transformers

# Tokenizers (BPE分词器)
pip install tokenizers

# 其他常用库
pip install datasets accelerate
```

### 验证安装

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## 推荐的IDE

| IDE | 优点 | 适用场景 |
|-----|------|----------|
| VS Code | 轻量、插件丰富 | 日常开发 |
| PyCharm | 调试能力强 | 大型项目 |
| Jupyter | 交互式 | 实验探索 |

## 目录结构

```
LLMs-from-scratch/
├── learning/              # 学习笔记和代码
│   ├── ch01-overview/    # 教程概述
│   ├── ch02-text-data/  # 文本处理
│   ├── ch03-attention/  # 注意力机制
│   ├── ch04-gpt-model/  # GPT架构
│   ├── ch05-pretraining/ # 预训练
│   ├── ch06-finetuning/ # 微调
│   ├── ch07-instruction-finetuning/ # 指令微调
│   ├── ch08-rlhf/       # 人类反馈强化学习
│   └── ch09-moe/        # MoE架构
├── data/                 # 训练数据
└── model/               # 保存的模型
```

## 快速测试

运行以下代码验证环境：

```python
# test_environment.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 检查GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 测试GPT2模型加载
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 测试文本生成
input_text = "Hello, world!"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0]))

print("环境验证通过！")
```

## 下一步

环境准备好后，开始学习第2章：文本数据处理
