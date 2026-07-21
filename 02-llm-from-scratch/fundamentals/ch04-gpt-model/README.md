# 第4章：GPT 模型架构

## 概述

第4章将前面学到的所有组件组装成完整的 GPT 模型。我们将逐层构建架构。

**新概念（共6个）：**
1. **LayerNorm（层归一化）** - 归一化激活值，稳定训练
2. **GELU** - 平滑激活函数（比 ReLU 更适合 Transformer）
3. **FeedForward（前馈网络）** - 两层神经网络，进行非线性变换
4. **TransformerBlock（Transformer块）** - 结合注意力 + 前馈网络 + 残差连接
5. **GPTModel（GPT模型）** - 完整架构，包含嵌入层和 Transformer 块
6. **Text Generation（文本生成）** - 自回归地逐词元生成

## 架构总览

```
输入 Token IDs
      ↓
┌─────────────────────────────────┐
│  Token Embedding + Position Emb │
│  词元嵌入 + 位置嵌入              │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│  Transformer Block 1            │
│  ├─ LayerNorm → Attention → 残差
│  └─ LayerNorm → FeedForward → 残差
└─────────────────────────────────┘
      ↓
      ... (重复 n_layers 次)
      ↓
┌─────────────────────────────────┐
│  Transformer Block 12           │
└─────────────────────────────────┘
      ↓
   Final LayerNorm（最终层归一化）
      ↓
   Linear → Logits (vocab_size)
   线性层 → logits（词表大小）
      ↓
   预测下一个 token
```

---

## 1. LayerNorm（层归一化）

### 作用
将嵌入维度上的值归一化，使其均值为0、方差为1。

### 为什么重要
- 通过将值保持在一致范围内来稳定训练
- 在注意力和前馈层**之前**应用（Pre-LN）
- 有可学习参数（scale 和 shift），模型可以调整

### 代码
```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除以零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

### 关键点
- `dim=-1` 表示沿最后一个维度（嵌入维度）归一化
- `keepdim=True` 保持维度以便广播
- `scale` 和 `shift` 在训练过程中学习（不是固定的！）

### 示例
```
输入 (batch=2, emb_dim=5):
[[-0.11, 0.12, -0.37, -0.24, -1.20],
 [ 0.21, -0.97, -0.76,  0.32, -0.11]]

每行均值: [-0.36, -0.26]
每行方差: [0.20, 0.27]

LayerNorm 后:
[[ 0.55,  1.07, -0.02,  0.27, -1.87],
 [ 0.91, -1.38, -0.96,  1.13,  0.29]]

归一化后均值≈0, 方差≈1
```

---

## 2. GELU（高斯误差线性单元）

### 作用
一种平滑的激活函数，近似于：`x * P(X ≤ x)`，其中 X ~ N(0,1)

### 为什么重要
- 比 ReLU 更平滑（在0处没有尖锐的角）
- 允许小的负值通过（不像 ReLU 会将负值归零）
- GPT、BERT 和大多数现代 Transformer 的标准配置

### 代码
```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

### 与 ReLU 的比较
```
输入:     [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
ReLU:     [ 0.0,  0.0,  0.0, 0.0, 1.0, 2.0, 3.0]  # 负值全变0
GELU:     [-0.004, -0.045, -0.159, 0.0, 0.841, 1.955, 2.996]  # 允许小负值
```

**注意：** GELU 在 0 附近是平滑曲线，ReLU 是尖锐的折线。

---

## 3. FeedForward（前馈网络）

### 作用
对每个 token 位置独立应用的两层神经网络。

### 结构
```
输入 (emb_dim=768) 
    → Linear → (4 × 768 = 3072)  # 扩展4倍
    → GELU 
    → Linear → (768)             # 压缩回原尺寸
```

### 代码
```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 扩展
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 压缩
        )

    def forward(self, x):
        return self.layers(x)
```

### 为什么要扩展4倍？
- 给模型更多的"思考空间"来处理复杂变换
- GPT-2 small 中是 768 → 3072 → 768
- 大部分计算发生在这个扩展层中

### 参数数量
```
Linear(768 → 3072): 768 × 3072 + 3072 = 2,362,368
Linear(3072 → 768): 3072 × 768 + 768 = 2,360,064
总计: 约 4.72M 参数
```

---

## 4. TransformerBlock（Transformer 块）

### 作用
结合多头注意力和前馈网络，并加入残差连接。

### 结构
```
输入 (x)
  │
  ├──────────────────────────┐
  ↓                          │
LayerNorm                    │
  ↓                          │
Multi-Head Attention         │
  ↓                          │
Dropout                      │
  ↓                          │
  + ←────────────────────────┘ (残差连接 = x + attention_output)
  │
  ├──────────────────────────┐
  ↓                          │
LayerNorm                    │
  ↓                          │
FeedForward                  │
  ↓                          │
Dropout                      │
  ↓                          │
  + ←────────────────────────┘ (残差连接 = x + ffn_output)
  │
  ↓
输出
```

### 代码
```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(...)  # 第3章学的
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 注意力块 + 残差
        shortcut = x
        x = self.norm1(x)      # Pre-LN: 先归一化
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut       # 加回原始输入（残差）

        # 前馈块 + 残差
        shortcut = x
        x = self.norm2(x)      # Pre-LN: 先归一化
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut       # 加回原始输入（残差）

        return x
```

### 关键点

#### Pre-LN vs Post-LN
- **Pre-LN**（GPT 使用）：LayerNorm 在 Attention/FFN **之前**
- **Post-LN**：LayerNorm 在 Attention/FFN **之后**
- Pre-LN 训练更稳定

#### 残差连接（Residual Connection）
```python
x = x + shortcut  # 或写作 x = x + attention_output
```
- 帮助梯度在反向传播时流动
- 允许模型"跳过"某些层
- 缓解深层网络的梯度消失问题

### 单个 TransformerBlock 的参数
```
MultiHeadAttention: 2,360,064 参数
FeedForward:        4,722,432 参数
LayerNorm (×2):     3,072 参数
总计:               7,085,568 参数 (约 7M)
```

---

## 5. GPTModel（完整 GPT 架构）

### 结构
```
Token IDs (batch, seq_len)
      ↓
┌─────────────────────────────┐
│ tok_emb: Embedding          │  词元嵌入 (vocab_size, emb_dim)
│ pos_emb: Embedding          │  位置嵌入 (context_length, emb_dim)
│ x = tok_emb + pos_emb       │  相加
│ x = dropout(x)              │  dropout
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│ trf_blocks: Sequential      │  12个 TransformerBlock
│   TransformerBlock × 12     │
└─────────────────────────────┘
      ↓
   final_norm: LayerNorm      最终层归一化
      ↓
   out_head: Linear           输出头
      ↓
   logits (batch, seq_len, vocab_size)
```

### 代码
```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)                    # (batch, seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len))     # (seq_len, emb_dim)
        x = tok_embeds + pos_embeds                          # 广播相加
        x = self.drop_emb(x)
        x = self.trf_blocks(x)                               # 通过12个块
        x = self.final_norm(x)
        logits = self.out_head(x)                            # (batch, seq_len, vocab_size)
        return logits
```

### GPT-2 124M 配置
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # BPE 分词器的词表大小
    "context_length": 1024,  # 最大序列长度
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 12,           # 注意力头数
    "n_layers": 12,          # Transformer 块数
    "drop_rate": 0.1,        # Dropout 率
    "qkv_bias": False        # Q/K/V 投影不加偏置
}
```

### 参数统计
```
词元嵌入 (Token embedding):  50257 × 768 = 38,597,376 (约 38.6M)
位置嵌入 (Position embedding): 1024 × 768 = 786,432 (约 0.8M)
Transformer 块:             12 × 7M = 85,026,816 (约 85M)
最终 LayerNorm:              768 × 2 = 1,536
输出头 (Output head):        **与词元嵌入共享权重**（权重绑定）

不重复计算的参数总数: 约 124M
```

### n_ctx 与 Position Embedding 的关系

**`n_ctx`（上下文长度）决定了 Position Embedding 表的大小：**

| 参数 | 含义 | 决定什么 |
|------|------|---------|
| `n_ctx` | 上下文长度 | Position Embedding 表的行数 |
| `emb_dim` | 嵌入维度 | Position Embedding 表的列数 |
| `n_ctx × emb_dim` | - | Position Embedding 表的总参数量 |

```
n_ctx = 1024
Position Embedding 表:
┌────────────────────────────────┐
│ pos=0    → [768维向量]        │
│ pos=1    → [768维向量]        │
│ pos=2    → [768维向量]        │
│ ...                             │
│ pos=1023 → [768维向量]        │ ← 最多只能表示 1024 个位置
└────────────────────────────────┘
                 │
                 └── 表大小: 1024 × 768 = 786,432
```

**为什么不能超过 n_ctx？**
- 如果输入序列长度 > n_ctx，Position Embedding 表没有对应的位置向量
- 模型无法表示超出训练长度位置的信息

### n_ctx 与 Attention 计算的关系

**`n_ctx` 决定了 Attention 计算的序列长度上限：**

```
n_ctx = 1024 表示最多能处理 1024 个 token 的序列

Attention 计算量 = seq_len × seq_len = n_ctx²

n_ctx=512:  512²  = 262,144 次计算
n_ctx=1024: 1024² = 1,048,576 次计算  ← 4倍！
```

**显存消耗也随 n_ctx² 增长：**
- Attention 权重矩阵: [seq_len, seq_len]
- n_ctx=512:  512 × 512 × 4字节 = 1MB
- n_ctx=1024: 1024 × 1024 × 4字节 = 4MB

### 权重绑定（Weight Tying）
```python
# out_head 的权重与 tok_emb 共享
self.out_head.weight = self.tok_emb.weight
```

**为什么这样做：**
1. **参数量减半**：`vocab_size × emb_dim` 这一大块参数（124M 模型里约 38M）只算一次，不重复计入输出头
2. **语义一致性**：输入端把 token id 映射到向量，输出端把向量映射回 vocab 概率——本质是同一套语义空间的正向/逆向映射，共享权重让两者学得更协调
3. **GPT-2 / GPT-3 的标准做法**，原始 GPT 论文也采用此设计

---

## 6. Text Generation（文本生成）

### 作用
通过预测下一个 token，逐个生成文本。

### 过程
```
"Hello" → 模型 → logits → argmax → "world"
"Hello world" → 模型 → logits → argmax → "!"
"Hello world!" → 模型 → logits → argmax → "How"
...
```

### 代码
```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx: (batch, seq_len) - 当前的 token 序列
    for _ in range(max_new_tokens):
        # 如果太长，只保留最后 context_size 个 token
        idx_cond = idx[:, -context_size:]
        
        # 获取模型预测
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 只关注最后一个位置的 logits
        logits = logits[:, -1, :]  # (batch, vocab_size)
        
        # 贪婪解码：选择概率最高的 token
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
        
        # 将新 token 添加到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
```

### 关键点

#### 贪婪解码（Greedy Decoding）
- 总是选择概率最高的 token（argmax）
- 简单但可能不是最优的（可能陷入重复）

#### 上下文截断
```python
idx_cond = idx[:, -context_size:]  # 只保留最后 1024 个 token
```
- GPT-2 最大支持 1024 个 token 的上下文

#### 自回归生成
- 输出成为下一步的输入
- 逐步生成，每次一个 token

### 其他解码方法（后续章节）
- **温度采样（Temperature sampling）**：增加随机性
- **Top-k 采样**：从概率最高的 k 个 token 中选择
- **Top-p（核）采样**：从累积概率达到 p 的最小集合中选择

### 生成示例
```
输入文本: "Hello, I am"
编码后: [15496, 11, 314, 716]

生成 10 个新 token 后:
输出 token IDs: [15496, 11, 314, 716, 27018, 24086, 47843, ...]
解码文本: "Hello, I am Featureiman Byeswickattribute argue logger..."

注意：输出是随机的，因为模型还没有训练！
预训练后会生成连贯的文本。
```

---

## 数据流总结

### 形状变化
```
输入: (batch=2, seq_len=4)
Token IDs: [[15496, 11, 314, 716], [25371, 42188, 47556, 8856]]

tok_emb: (2, 4, 768)       # 词元嵌入
pos_emb: (4, 768)          # 位置嵌入（广播到 batch）
x = tok + pos: (2, 4, 768)

经过每个 TransformerBlock: (2, 4, 768)  # 形状保持不变！

final_norm 后: (2, 4, 768)
out_head 后: (2, 4, 50257)  # 每个位置的 logits

生成时，使用 logits[:, -1, :] → (2, 50257) → argmax → (2, 1)
```

---

## Q&A 洞察

### Q1: LayerNorm 的 scale 和 shift 参数有什么用？

**问题**：LayerNorm 有两个可学习参数 scale (γ) 和 shift (β)，它们有什么作用？

**回答**：
- **没有参数**：归一化后永远是 mean=0, var=1，限制了表达能力
- **有了参数**：模型可以学习恢复或调整分布

```python
output = scale * norm_x + shift
# scale 控制拉伸/压缩（改变方差）
# shift 控制平移（改变均值）
```

例如：学习到 scale=2, shift=5 → 变成均值5，方差4

**总结**：给模型灵活性，可以选择保持归一化，也可以调整分布。

### Q2: 为什么生成的文本是"随机"的？

**问题**：说模型输出是随机的，但多次运行结果相同？

**回答**：
- **模型权重随机**：初始化时随机生成，所以输出无意义（乱码）
- **运行结果固定**：代码里设置了 `torch.manual_seed(123)`，保证可重复

```python
torch.manual_seed(123)  # 固定随机种子
# 每次运行得到相同的"乱码"
```

**区别**：
- "随机输出" = 模型没训练，权重随机
- "结果相同" = 设了种子，方便调试

---

## 运行代码

```bash
cd 02-llm-from-scratch/fundamentals/ch04-gpt-model
python ch04_code.py
```

---

## 下一步

理解本章后：
- 第5章：预训练（损失函数、训练循环）
- 第6章：针对特定任务的微调
