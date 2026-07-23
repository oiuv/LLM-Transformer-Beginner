# Chinese GPT - 中文小说生成模型

使用 PyTorch + Transformers 训练中文 GPT 模型，可以生成小说风格的文本。

## 📖 功能特性

- ✅ 自定义 BPE 分词器（适配中文）
- ✅ 基于 GPT2 架构
- ✅ 支持 txt 和 jsonl 数据格式（兼容开源数据集）
- ✅ per-sample 样本组织 + bos/eos 边界（与 minimind PretrainDataset 对齐）
- ✅ AdamW betas=(0.9, 0.95) + weight decay 分组（与 minimind 训练配方对齐）
- ✅ 小模型 dropout=0（HF GPT-2 默认 0.1 在 <1 亿参数规模上属过正则化；如需恢复用 `-dp 0.1`）
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
python train.py -d data/小说.txt -C 256 -E 512 -L 6 -b 4
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
| `-V` | 词表大小 | 6400 |
| `-C` | 上下文长度 | 340 |
| `-E` | 嵌入维度 | 768 |
| `-H` | 注意力头数 | 8 |
| `-L` | Transformer 层数 | 8 |
| `-b` | 批次大小 | 32 |
| `-acc` | 梯度累积步数 | 8（等效 batch=256） |
| `-dp` | Dropout 概率 | 0.0（小模型推荐；HF GPT-2 默认 0.1）|
| `-lr` | 学习率 | 5e-4 |
| `-e` | 训练轮数 | 2 |
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
python train.py -d data.txt -C 256 -E 512 -L 6 -H 8 -b 8 -e 10
```
- 上下文：256
- 维度：512
- 层数：6
- 头数：8

### 标准模型（默认配置，对齐 minimind）

```bash
python train.py -d dataset/pretrain_t2t_mini.jsonl
```

默认配置：`vocab=6400, 上下文=340, 嵌入=768, 头数=8, 层数=8, batch=32, acc=8（等效 256）, epochs=2`

直接 `python train.py` 即可使用这套默认配置训练，详见下方"默认配置说明"。

### 大模型（需要更多显存）
```bash
python train.py -d data.txt -C 1024 -E 1024 -L 16 -H 16 -b 2 -e 30
```

## 🎯 训练流程

### 默认配置说明

`train.py` 的默认值已对齐 minimind 的 pretrain 默认配置：

| 参数 | chinese-gpt 旧默认 | 新默认（对齐 minimind） |
|---|---|---|
| 词表大小 `-V` | 50000 | **6400** |
| 上下文长度 `-C` | 512 | **340** |
| 注意力头数 `-H` | 12 | **8** |
| Transformer 层数 `-L` | 12 | **8** |
| 批次大小 `-b` | 8 | **32** |
| 梯度累积 `-acc` | 1（不累积） | **8（等效 256）** |
| 训练轮数 `-e` | 20 | **2** |
| 分词器预处理器 | `Whitespace()` | **`ByteLevel()`** |

> 注：旧默认值 50000/512/12/12/8/20/1 是教程最初版的"通用 GPT-2 默认"思路——词表大、模型大、epoch 多。新默认针对中文小说场景做了精简，对齐 minimind 实战配置。

这样改是为了让 chinese-gpt 的训练效果能直接和 minimind 对照参考——**架构仍然是 GPT-2**（不是 Llama），所以和 minimind 仍有结构性差距；改这些默认值拉齐的是"数据吞吐、tokenization、训练步数"等可对标的部分。架构差异详见各文件的 docstring。

直接 `python train.py` 即可使用新默认配置训练。

### 训练阶段

```
[阶段1] 加载数据
    ↓
[阶段2] 训练 BPE 分词器（ByteLevel 预处理器，见下方说明）
    ↓
[阶段3] 创建数据集（per-sample + bos/eos，见下方说明）
    ↓
[阶段4] 创建 GPT 模型
    ↓
[阶段5] 训练模型
    ├── 训练集训练
    ├── 验证集评估
    ├── 保存最佳模型
    └── 早停检查
```

### 关于分词器（ByteLevel vs Whitespace）

`train.py` 用 `ByteLevel(add_prefix_space=False)` 作为 BPE 的预处理器（pre_tokenizer），与 minimind 对齐。**这两种都是合法的 BPE 训练前置切分方式，不是新概念——BPE 本身的训练/合并/解码流程完全一样**，只是"先怎么切"的选择：

| 预处理器 | 切分依据 | 中文表现 | 典型代表 |
|---|---|---|---|
| `Whitespace()` | 空格/换行 | 中文无空格 → 整段汉字当作一个"词" → BPE 学不到字符级合并 | 早期教程通用做法 |
| `ByteLevel()` | UTF-8 字节 | 中文字符固定 3 字节 → BPE 在字节上做合并 → 常用字/偏旁成为 token，压缩率约 1.5–1.7 char/token | GPT-2、minimind |

对**生成时的副作用**：ByteLevel 用 Ġ (U+0120) 代替 ASCII 空格标记"词首"，中文输出里这些字符没有语义，会污染结果。`generate.py` 的 `decode()` 已经做了 `text.replace("Ġ", " ")` 后处理，无需手动干预。

如果想对比效果，可以手动切回 `Whitespace()`：`train.py` 第 226 行 `tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)` 改为 `Whitespace()`，删除已有 `output/tokenizer.json` 后重新训练即可（vocab_size 也要相应调高到 8000+ 才合理）。

### 关于数据加载方式（per-sample + bos/eos vs 滑动切块）

`train.py` 用 **per-sample + bos+text+eos** 的方式构造训练样本，对齐 minimind 的 `PretrainDataset`。这是中文小说 / 通用中文 pretrain 场景下更现代的做法。

旧版（chinese-gpt 早期）的做法：把所有文本 `"".join` 成一个超长字符串 → 按 `context_length` 滑动切块 → 相邻块互相重叠，没有显式的样本边界。

#### 两种做法对比

| 维度 | 旧版：滑动切块 | 新版：per-sample + bos/eos（当前） |
|---|---|---|
| 一个训练样本 = | 一段 context_length 长的 token 块 | 一条 jsonl / 一个 txt 文件 |
| 相邻样本关系 | 第 N 块和第 N+1 块有 ~339 token 重叠 | 互相独立，每条样本都有自己的 bos 和 eos |
| 样本边界 | 无明确标记，模型分不清"段尾"和"新一段开头" | `bos` 之后是新样本，`eos` 处必停，模型对样本边界有清晰归纳偏置 |
| 位置编码学到的是 | "在第 N 个 token"（跨样本混合的绝对位置） | "在第 N 条样本内的第 M 个 token"（独立位置） |
| label 计算 | `chunk[:-1] / chunk[1:]`（chunk 内直接截位） | `labels[i] = input_ids[i+1]` + `labels[pad] = -100` |
| 缓存 | 预编码整个语料到 `.npy`（首次几秒→重启秒启动） | 不缓存，每次 `__getitem__` 重新 tokenize（首次启动稍慢，无磁盘占用） |

#### 为什么 per-sample 在中文小说场景更优

1. **样本边界可学习**：bos/eos 是模型能"看见"的特殊 token，LM 学到 eos 后必停 → 生成时撞到 eos 自然结束，无需依赖长度截断。旧版切块方式相邻两块共用同一段文本结尾/开头，模型对"这是一段的结尾 / 那是另一段的开头"无归纳偏置。
2. **位置编码稳定**：每条样本的 bos 都在位置 0，eos 都在最末位 → 位置编码里"开头"和"结尾"的信号一致；旧版里第 N 块的开头可能在绝对位置 10000、第 N+1 块的开头在绝对位置 10340，位置编码被稀释成"全文第几 token"信息。
3. **与开源 pretrain 配方对齐**：minimind / nanoGPT / HuggingFace `DataCollatorForLanguageModeling` 教科书都使用 per-sample + bos/eos，是事实标准。
4. **对 truncation 友好**：`tokenizer.encode(text, max_length=context_length-2, truncation=True)` 在 tokenizer 内部做截断，单条样本过长不会污染下游样本边界。旧版按字符切块到 context_length 时截断位置毫无语义。

#### 代码位置 & 验证

- `train.py:NovelDataset.__getitem__` — per-sample 编码 + bos/eos 包裹 + pad 的实际写法
- `train.py:NovelDataset.__init__` — 不再接收 `text` 和 `cache_path`，只接收 `samples: List[str]`
- `train.py:load_and_preprocess_data` — 返回 `(samples, paragraphs)`；`samples` 给 `NovelDataset` 训练，`paragraphs` 给 BPE 学习 token 合并
- 训练循环 / optimizer / scheduler / early-stop **完全未动** —— 数据组织只影响 `Dataset` 这一层

注：这个改动属于"教程外"知识点。BPE、Dataset、DataLoader 在前置教程里都讲过，但"per-sample vs 滑动切块"这种 pretrain 数据组织设计选型没有专门讲过 —— 这里补的就是这个 gap。

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
- 减小批次大小：`-b 2` 或 `-b 1`（**这才会真降显存**）
- 减小上下文长度：`-C 256`
- 减小模型维度：`-E 512`
- 用 `-acc` 加大梯度累积步数（**不会降显存**，只是用更小的 micro-batch 模拟更大的等效 batch，便于对齐 minimind 等项目的等效 batch 配置）

### 训练速度太慢
- 增加批次大小（如果显存允许）
- 使用更小的验证集比例：`-s 0.01`
- 使用 GPU 而不是 CPU

### 生成文本质量差
- 增加训练轮数：`-e 50`
- 增加数据量
- 调整学习率：`-lr 1e-4`
- 使用更大的模型

### 加载旧 checkpoint 时 `param_groups` 不匹配
升级到 weight-decay 分组 + betas=(0.9, 0.95) 的 AdamW 后，旧的 `output/model/checkpoint.pt` 里 `optimizer_state_dict` 的 `param_groups` 形状不同，`load_state_dict` 会报错。  
解决：删除 `output/model/checkpoint.pt` 从头重训（建议同时让模型权重跟着新配方走，避免 loss 曲线在新 optimizer 上出现 spike）。

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
