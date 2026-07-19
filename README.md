# LLM-Transformer-Beginner (原 Machine-Learning-Beginner)

一个从零开始手写代码学习机器学习、神经网络和大语言模型的**训练技术**项目。

> 📌 **项目定位**：本项目专注于 **LLM 训练、微调、对齐与衍生技术**方向——从 `y=wx` 入门，到手撸 Llama3 大模型，再到训练自己的中文 GPT。
>
> 应用方向（Agent、RAG、Embedding、Skill/MCP 等）的学习请移步姊妹项目：
> - 🤖 基于 LLM 的 Agent 开发（含 Skill/MCP）→ [LLM-Agent-Beginner](https://github.com/oiuv/LLM-Agent-Beginner)
> - 🧬 Embedding / RAG 等应用层 → [LLM-Embedding-Beginner](https://github.com/oiuv/LLM-Embedding-Beginner)

## 🎯 项目简介

做为一个传统的程序员，在使用AI并开始学习机器学习和神经网络相关的内容后，一直存在一些疑惑，比如：

1. 机器学习编程和传统编程的核心区别是什么？有没有最简单直观的代码能让我感受感受？
2. 机器到底是怎么学习的？有没有最简单直观的代码能让我感受感受？
3. 为什么说大模型是一个函数？这个函数长什么样的？
4. 为什么要学习率，它到底起个什么作用？为什么学习率小了收敛速度慢，大了又可能震荡或发散？
5. 反向传播是怎么传播的？梯度下降又是什么鬼？
6. 为什么没有激活函数的神经网络只能表达线性关系？激活函数是怎么做非线性变换的？
7. 神经元和隐藏层到底都起的什么作用？
8. ……

是的，我就想看有没有代码让我能直观的理解这些概念。本项目完全从零开始撸代码，从 `y=wx` 开始入门，到手撸 Llama3 大模型，再到训练自己的中文 GPT，循序渐进地理解机器学习的本质。

---

## 📚 学习路径

本项目按照 **基础 → 进阶 → 实践** 的路径组织：

### 🌱 01-basics - 基础篇：机器学习入门
完全从零手写代码，不使用机器学习框架，直观理解核心概念。

| 章节 | 内容 | 文件 |
|------|------|------|
| [01-linear-model](01-basics/01-linear-model/) | 从 `y=wx` 开始了解机器是怎么学习的 | `tutorial.ipynb` |
| [02-gradient-descent](01-basics/02-gradient-descent/) | 从均方误差感受梯度下降的具体实现 | `tutorial.ipynb`, `tutorial.py` |
| [03-activation](01-basics/03-activation/) | 用激活函数感受非线性变换的效果 | `tutorial.ipynb`, `tutorial.py` |
| [04-neural-network](01-basics/04-neural-network/) | 手撸神经网络感受深度学习 | `tutorial.ipynb` |

**你将学到：**
- 机器学习与传统编程的核心区别
- 参数、特征、标签、超参数等概念
- 学习率、梯度下降、反向传播的原理
- 激活函数的非线性变换
- 神经网络的前向传播和反向传播

---

### 🚀 02-llm-from-scratch - 进阶篇：从零实现大模型
不依赖 PyTorch/TensorFlow 的高级封装，逐行手写实现大语言模型。

> 📌 **致谢**：本章节 Llama3 教程基于 [naklecha/llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch) 改编，感谢原作者的精彩实现！

| 章节 | 内容 | 文件 |
|------|------|------|
| [fundamentals](02-llm-from-scratch/fundamentals/) | **LLM 基础教程** - 从文本处理到 RLHF 的完整学习路径（9章） | 多章节 |
| [llama3-step-by-step](02-llm-from-scratch/llama3-step-by-step/) | **Llama3 分解式教学** - 10个渐进式课程，从零实现每个组件 | 10个lesson |
| [llama3](02-llm-from-scratch/llama3/) | **Llama3 完整实现** - 单个notebook完整实现Llama3-8B | `tutorial.ipynb` |

**学习路径建议：**
1. **零基础系统学习** → `fundamentals/`（9章通用教程）
2. **深入Llama3每个组件** → `llama3-step-by-step/`（10课分解教学）
3. **快速概览整体架构** → `llama3/`（完整notebook）

**你将学到：**
- 大语言模型的内部架构（Tokenizer / Embedding / Attention / FFN / Norm）
- Tokenizer（BPE 分词器训练与对齐）
- RMSNorm 归一化、RoPE 旋转位置编码、Grouped Query Attention、SwiGLU 前馈网络
- KV-Cache 推理优化原理、Flash Attention 的工程动机
- 现代对齐流程：SFT → RLHF (PPO 概念) → DPO/ORPO 等新一代偏好优化
- MoE（混合专家）：Top-K 路由、负载均衡、Switch/Mixtral 风格架构
- 长上下文技术原理：RoPE 外推、YaRN、Sliding Window Attention
- 数据工程基础、BPE/SentencePiece 选型、混合精度训练（bf16/fp16）

---

### 🛠️ 03-practice - 实践篇：训练自己的模型
使用成熟框架（PyTorch + Transformers）训练实用的中文 GPT。

| 章节 | 内容 | 文件 |
|------|------|------|
| [chinese-gpt](03-practice/chinese-gpt/) | 训练中文小说 GPT | `train.py` |

**你将学到：**
- 使用 Transformers 库构建 GPT
- 训练自定义 BPE 分词器
- 数据预处理和 Dataset 构建
- 模型训练和验证（含 bf16 混合精度、Cosine Schedule + Warmup、梯度裁剪）
- 断点续训、早停机制、显存监控
- 文本生成（自回归 / Temperature / Top-p / Repetition Penalty）

---

### 🧪 tests - 实验与测试
自由实验和测试代码的 playground。

---
### 🔮 04-future - 未来扩展

`04-future/` 目录预留扩展空间。下面这些**仍属于本项目定位（LLM 训练/微调技术）**，按重要性排序：

**📦 参数高效微调（PEFT）**
- LoRA / QLoRA / AdaLoRA / DoRA / IA³
- 用 PEFT 库微调开源模型（不写一行训练循环）

**⚡ 推理优化与量化**
- KV-Cache / Grouped-Query Attention 的实现细节
- Flash Attention 原理与 torch.nn.functional.scaled_dot_product_attention
- PagedAttention（vLLM 思想）/ Speculative Decoding
- 量化：bitsandbytes int4/int8 / GPTQ / AWQ

**🎯 现代对齐（2024+）**
- DPO / ORPO / SimPO / KTO（替代 PPO 的新一代偏好优化）
- RLAIF / Constitutional AI / Process Reward Model
- Reasoning 模型后训练：DeepSeek-R1 / GRPO / o1-style 训练循环

**🔧 高级预训练工程**
- 数据工程：清洗、deduplication、质量过滤
- 分布式训练：DDP / FSDP / DeepSpeed
- 显存优化：Gradient Checkpointing / Activation Offloading
- 混合精度进阶：fp16 + GradScaler / fp8（Hopper）
- 评估：lm-eval-harness / MMLU / GSM8K / HumanEval / Perplexity

**🧩 衍生与组合**
- Model Merging（TIES / DARE）
- Knowledge Distillation（Teacher-Student）
- 模型转换与导出（ONNX / GGUF / llama.cpp）

**📐 长上下文与稀疏架构**
- RoPE 外推 / YaRN / Sliding Window Attention
- MoE 实战：Switch Transformer / Expert Parallel / All-to-All 通信

**🎨 多模态训练（可选方向）**
- Vision-Language Model（如 LLaVA）训练原理
- CLIP-style 双塔对比学习

> ⚠️ 备注：`Agent / RAG / Tool Use / Skill / MCP / Embedding 应用` 等**应用方向**不属于本项目，请见开头 [姊妹项目](#) 链接。

---

## 🚀 快速开始

### 基础篇
```bash
# 进入第一章
jupyter notebook 01-basics/01-linear-model/tutorial.ipynb
```

### 进阶篇
```bash
# 需要先下载 Llama3 权重
jupyter notebook 02-llm-from-scratch/llama3/tutorial.ipynb
```

### 实践篇
```bash
# 训练中文 GPT
python 03-practice/chinese-gpt/train.py -d ../../data/小说.txt
```

---

## 📖 推荐学习顺序

1. **完全零基础**：从 `01-basics/01-linear-model/` 开始，按顺序学习
2. **有基础想系统学习 LLM**：`02-llm-from-scratch/fundamentals/`（9章完整教程）
3. **想深入 Llama3 每个组件**：`02-llm-from-scratch/llama3-step-by-step/`（10课分解教学）⭐
4. **有基础想快速概览 LLM**：直接跳到 `02-llm-from-scratch/llama3/`
5. **想快速上手项目**：从 `03-practice/chinese-gpt/` 开始

---

## 💡 学习建议

- 本教程过于基础，相当于在教你数学的四则运算
- 建议配合大模型（如 ChatGPT）学习并扩展知识
- 每个章节都包含详细的代码和图解，建议边运行边理解
- 不要只看，一定要动手跑代码！

---

## 📂 项目结构

```
LLM-Transformer-Beginner/
├── README.md                    # 本文件
├── AGENTS.md                    # 给 AI 协作者的工程说明
├── LICENSE
├── .gitignore
│
├── 01-basics/                   # 🌱 基础篇
│   ├── 01-linear-model/
│   ├── 02-gradient-descent/
│   ├── 03-activation/
│   └── 04-neural-network/
│
├── 02-llm-from-scratch/         # 🚀 进阶篇
│   ├── fundamentals/            # 通用 LLM 基础教程（9章）
│   ├── llama3-step-by-step/     # Llama3 分解式教学（10课）⭐
│   └── llama3/                  # Llama3 完整实现（notebook）
│
├── 03-practice/                 # 🛠️ 实践篇
│   ├── chinese-gpt/             # 中文小说 GPT
│   └── chinese-gpt-fundamentals.md  # 基础教程项目链接
│
├── 04-future/                   # 🔮 未来扩展（PEFT / 量化 / DPO / 推理优化等）
│
├── scripts/                     # 一键训练脚本 + 编码转换工具
│
└── tests/                       # 🧪 playground
```

**🔗 姊妹项目（应用方向）**
- [LLM-Agent-Beginner](https://github.com/oiuv/LLM-Agent-Beginner) — Agent / Skill / MCP
- [LLM-Embedding-Beginner](https://github.com/oiuv/LLM-Embedding-Beginner) — Embedding / RAG 应用

**🤝 推荐互补项目**
- [MiniMind](https://github.com/jingyaogong/minimind) — [项目主页](https://jingyaogong.github.io/minimind/)
  从零训练一个 64M 参数的完整小模型（3 块钱 + 2 小时）。覆盖 Pretrain → SFT → LoRA → DPO/PPO/GRPO → Agent 训练 → 蒸馏全链路，含 Web Demo 和 API 服务。
  本项目侧重"理解原理"（逐行手写讲解），MiniMind 侧重"动手训练"（端到端训出一个能对话的模型）。建议先学完本项目理解原理，再用 MiniMind 跑通完整训练流程。

---

## 🙏 致谢

- [naklecha/llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch) - 本项目的 Llama3 教程改编自该仓库，感谢原作者 Naklecha 的精彩实现和分享！

---

## 📝 License

MIT License

---

**Happy Learning! 🎉**
