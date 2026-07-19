# Chinese GPT - 中文小说生成模型

使用 PyTorch + Transformers 训练中文 GPT 模型，可以生成小说风格的文本。

## 📖 功能特性

- ✅ 自定义 BPE 分词器（适配中文）
- ✅ 基于 GPT2 架构
- ✅ 支持 txt 和 jsonl 数据格式（兼容开源数据集）
- ✅ 完整的训练流程
- ✅ 断点续训支持
- ✅ 早停机制
- ✅ 验证集评估
- ✅ 训练日志记录
- ✅ 自动生成样例
- ✅ 文本生成功能

## 🚀 快速开始

### 1. 准备数据

支持三种数据格式：

```bash
# 方式一：txt 文件（传统方式）
data/小说.txt

# 方式二：jsonl 文件（推荐，兼容开源数据集）
dataset/pretrain_data.jsonl   # 每行 {"text": "..."}

# 方式三：目录（自动扫描 .txt 和 .jsonl）
data/
├── 小说1.txt
├── 小说2.txt
└── pretrain_data.jsonl
```

### 2. 训练模型

```bash
# 默认训练（自动使用 dataset/pretrain_t2t_mini.jsonl）
python train.py

# 指定 txt 文件
python train.py -d data/小说.txt

# 指定 jsonl 文件
python train.py -d dataset/pretrain_data.jsonl

# 多文件训练（传入目录，自动合并所有 txt 和 jsonl）
python train.py -d data/

# 自定义参数
python train.py -d data/小说.txt -e 5 -b 4

# 小模型快速实验
python train.py -d data/小说.txt -c 256 -E 512 -L 6 -b 4
```

### 3. 生成文本

```bash
# 基础生成
python generate.py --model output/model --prompt "第一章"

# 自定义参数
python generate.py --model output/model --prompt "第一章" --length 1000 --temperature 0.9

# 低温度（更确定）
python generate.py --model output/model --prompt "第一章" --temperature 0.5

# 高温度（更随机）
python generate.py --model output/model --prompt "第一章" --temperature 1.2
```

## 📂 目录结构

```
chinese-gpt/
├── train.py              # 训练脚本
├── generate.py           # 生成脚本
├── README.md             # 本文件
└── configs/              # 配置文件（可选）
```

## ⚙️ 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-d` | 数据文件路径或目录（支持 txt/jsonl） | `dataset/pretrain_t2t_mini.jsonl` |
| `-o` | 输出目录 | `./output` |
| `-V` | 词表大小 | 50000 |
| `-C` | 上下文长度 | 512 |
| `-E` | 嵌入维度 | 768 |
| `-H` | 注意力头数 | 12 |
| `-L` | Transformer 层数 | 12 |
| `-b` | 批次大小 | 8 |
| `-lr` | 学习率 | 5e-4 |
| `-e` | 训练轮数 | 20 |
| `-s` | 验证集比例 | 0.05 |

## 📝 生成参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` / `-m` | 模型路径 | 必需 |
| `--prompt` / `-p` | 生成提示文本 | 必需 |
| `--length` / `-l` | 生成token数量 | 500 |
| `--temperature` / `-t` | 温度参数（越高越随机） | 0.8 |
| `--top_p` | Top-p采样（控制多样性） | 0.9 |
| `--repetition_penalty` | 重复惩罚 | 1.1 |

## 📊 模型配置示例

### 小模型（快速实验）
```bash
python train.py -d data.txt -c 256 -E 512 -L 6 -H 8 -b 8 -e 10
```
- 上下文：256
- 维度：512
- 层数：6
- 头数：8

### 标准模型
```bash
python train.py -d data.txt -c 512 -E 768 -L 12 -H 12 -b 4 -e 20
```
- 上下文：512
- 维度：768
- 层数：12
- 头数：12

### 大模型（需要更多显存）
```bash
python train.py -d data.txt -c 1024 -E 1024 -L 16 -H 16 -b 2 -e 30
```

## 🎯 训练流程

```
[阶段1] 加载数据
    ↓
[阶段2] 训练 BPE 分词器
    ↓
[阶段3] 创建数据集
    ↓
[阶段4] 创建 GPT 模型
    ↓
[阶段5] 训练模型
    ├── 训练集训练
    ├── 验证集评估
    ├── 保存最佳模型
    └── 早停检查
```

## 💾 输出文件

```
output/
├── tokenizer.json          # 分词器
├── config.json             # 训练配置（超参数、最佳损失等）
├── training.log            # 训练日志（每轮损失记录）
└── model/
    ├── config.json         # 模型配置
    ├── pytorch_model.bin   # 模型权重
    └── checkpoint.pt       # 训练断点（临时，训练完成后删除）
```

### training.log 格式
```
1 2.3456 2.1234
2 1.9876 1.8765
...
```
每行：`epoch train_loss val_loss`

## 🔍 监控训练

训练过程中会显示：
- 训练损失
- 验证损失
- 当前轮数
- 显存使用
- 最佳模型保存提示

## 🛠️ 故障排除

### CUDA 内存不足
- 减小批次大小：`-b 2` 或 `-b 1`
- 减小上下文长度：`-c 256`
- 减小模型维度：`-E 512`
- 使用梯度累积（需修改代码）

### 训练速度太慢
- 增加批次大小（如果显存允许）
- 使用更小的验证集比例：`-s 0.01`
- 使用 GPU 而不是 CPU

### 生成文本质量差
- 增加训练轮数：`-e 50`
- 增加数据量
- 调整学习率：`-lr 1e-4`
- 使用更大的模型

## 📚 进阶功能

### 断点续训
如果训练中断，重新运行相同命令会自动从断点继续。

### 早停机制
验证损失连续 3 轮不下降时自动停止训练，防止过拟合。

## 🎉 完整示例

```bash
# 1. 快速开始（使用默认 jsonl 数据集）
python train.py

# 2. 使用自己的小说数据
mkdir -p data
# 将你的小说txt文件放入 data/ 目录
python train.py -d data/小说.txt -e 10 -b 4

# 3. 使用 jsonl 数据集（将 jsonl 文件放入 dataset/ 目录）
python train.py -d dataset/pretrain_data.jsonl

# 4. 查看训练日志
cat output/training.log

# 5. 生成文本
python generate.py --model output/model --prompt "第一章" --length 500

# 6. 尝试不同风格
python generate.py --model output/model --prompt "话说" --temperature 1.0
python generate.py --model output/model --prompt "江湖" --temperature 0.7
```

---

**Previous ← [02-llm-from-scratch](../../02-llm-from-scratch/)**
