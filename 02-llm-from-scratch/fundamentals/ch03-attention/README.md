# 第3章：注意力机制 (Attention Mechanism)

本章你将理解 LLM 的核心 - 注意力机制是如何工作的。

---

## 核心概念

```
Self-Attention: 让每个词都能"看到"所有其他词

     Query/Key/Value: 三种角色，计算注意力权重
        Causal Mask: 遮住未来，防止"作弊"
      Multi-Head: 多个头，捕获不同关系
```

---

## 3.1 为什么需要注意力机制？

### 问题

第2章把文本转换成了向量序列，但这些向量之间**没有交互**，不知道彼此的关系。

```
输入: [Hello, world]
向量: [[0.1, 0.2], [0.3, 0.4]]
      ↓
模型不知道 "Hello" 和 "world" 有关系！
```

### 解决方案

注意力机制让每个位置都能"看到"所有其他位置：

```
输入序列: "Your journey starts with one step"
              ↓      ↓      ↓     ↓    ↓    ↓
          [x1]   [x2]   [x3]  [x4] [x5] [x6]
              ↓      ↓      ↓     ↓    ↓    ↓
          [z1]   [z2]   [z3]  [z4] [z5] [z6]

每个 z[i] 都是所有 x 的加权和
z[2] = α1*x1 + α2*x2 + α3*x3 + α4*x4 + α5*x5 + α6*x6
       ↑    ↑    ↑    ↑    ↑    ↑
     注意力权重(和为1)
```

---

## 3.2 简单自注意力（无训练权重）

### 三步计算

```
Step 1: 计算注意力分数（未归一化）
    ω_ij = x[i] · x[j]^T    # 点积

Step 2: 归一化（softmax）
    α_ij = softmax(ω_ij)    # 转换成概率

Step 3: 计算上下文向量
    z[i] = Σ α_ij * x[j]      # 加权求和
```

### 代码示例

```python
import torch

# 6个词，每个词用3维向量表示
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your     (x^1)
    [0.55, 0.87, 0.66],  # journey  (x^2)
    [0.57, 0.85, 0.64],  # starts   (x^3)
    [0.22, 0.58, 0.33],  # with     (x^4)
    [0.77, 0.25, 0.10],  # one      (x^5)
    [0.05, 0.80, 0.55],  # step     (x^6)
])  # shape: [6, 3]

# Step 1: 计算注意力分数（点积）
attn_scores = inputs @ inputs.T  # [6, 6]

# Step 2: 归一化（softmax）
attn_weights = torch.softmax(attn_scores, dim=-1)  # [6, 6]

# Step 3: 计算上下文向量
context_vecs = attn_weights @ inputs  # [6, 3]
```

### 为什么用点积？

```
点积 = 相似度

x · y 越大 → 越相似 → 越应该关注
```

**点积的本质**：
```python
# 两个向量的点积
a = [1, 2, 3]
b = [1, 2, 3]

a · b = 1*1 + 2*2 + 3*3 = 14  # 完全相同，点积大

a = [1, 2, 3]
b = [-1, -2, -3]

a · b = -1 -4 -9 = -14  # 完全相反，点积小（负）
```

### ⚠️ 重要说明

这个简单版本**没有可训练权重**，只用于演示！实际模型使用可训练的 QKV 投影。

---

## 3.3 带可训练权重的自注意力

### 核心改进：引入 Query, Key, Value

```
输入 x → 三个可训练的投影 → Query, Key, Value

Query (查询): "我要找什么？"
Key (键):     "我是什么？"
Value (值):  "我的内容是什么？"
```

### 比喻：图书馆找书

```
Query = 搜索关键词 ("动物")
Key = 书的分类标签 ("动物"标签)  
Value = 书的实际内容

搜索时：
  1. 用 Query ("动物") 
  2. 匹配 Key (标签)
  3. 返回 Value (内容)
```

### 数学公式

```
# 1. 投影到 Q, K, V 空间
q[i] = x[i] @ W_query    # 可训练矩阵
k[i] = x[i] @ W_key      # 可训练矩阵
v[i] = x[i] @ W_value    # 可训练矩阵

# 2. 计算注意力分数
attn_scores = q[i] · k[j]^T

# 3. 缩放（稳定训练）
attn_scores = attn_scores / sqrt(d_k)

# 4. 归一化
attn_weights = softmax(attn_scores)

# 5. 加权求和
output = Σ attn_weights * v[j]
```

### 为什么需要缩放？

```python
# 不缩放的问题：
# 大的 d_k 会导致点积值很大
# → softmax 输出接近 one-hot（0或1）
# → 梯度很小，训练困难

# 缩放后：保持数值稳定
d_k = keys.shape[-1]
attn_weights = softmax(attn_scores / d_k**0.5)
```

### 代码实现

```python
import torch.nn as nn

d_in = 3    # 输入维度
d_out = 2   # 输出维度

# 三个可训练的权重矩阵（随机初始化）
W_query = nn.Parameter(torch.rand(d_in, d_out))
W_key = nn.Parameter(torch.rand(d_in, d_out))
W_value = nn.Parameter(torch.rand(d_in, d_out))

# 投影到 Q, K, V 空间
queries = inputs @ W_query  # [6, 2]
keys = inputs @ W_key      # [6, 2]
values = inputs @ W_value    # [6, 2]

# 计算注意力
d_k = keys.shape[-1]
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
output = attn_weights @ values
```

---

## 3.4 训练过程

### 关键问题：训练过程中调整什么？

| 参数 | 初始值 | 训练中 |
|------|-------|--------|
| Token Embedding 向量 | 随机 | ✅ 调整 |
| W_query 矩阵 | 随机 | ✅ 调整 |
| W_key 矩阵 | 随机 | ✅ 调整 |
| W_value 矩阵 | 随机 | ✅ 调整 |

### 训练流程

```
训练目标：预测下一个词

输入: [The, cat, sat]
目标: 预测 "mat"

1. 查找 embedding → 向量
2. 通过 QKV 计算注意力
3. 预测下一个词
4. 计算损失（预测错了吗？）
5. 反向传播 → 调整：
   - embedding 向量
   - W_query
   - W_key
   - W_value

重复数百万次后：
- embedding 向量变得有意义（相似的词向量接近）
- QKV 矩阵学会"什么关系是重要的"
```

### 训练结果

```
训练前:
  "cat" → [0.1, 0.2, 0.3]  # 随机
  "dog" → [0.8, 0.9, 0.1]  # 随机
  
训练后:
  "cat" → [0.5, 0.3, 0.8]  # 学到"动物"语义
  "dog" → [0.6, 0.4, 0.7]  # 学到"动物"语义
  
→ "cat" 和 "dog" 向量接近（因为上下文相似）
```

---

## 3.5 因果注意力（Causal Attention）

### 问题

在生成任务中，不能让模型"看到未来"！

```
当前: "The cat sat on the"
预测: "mat"

模型在预测时：
  ❌ 不应该看到 "mat"（这是要预测的答案）
  ✓ 只能看到前面的词
```

### 解决方案：Causal Mask

```
输入: "Your journey starts with one step"
位置:   [1]   [2]     [3]    [4]  [5]  [6]

无掩码（所有位置互相可见）:
      [1] [2] [3] [4] [5] [6]
[1]  [ ✓   ✓   ✓   ✓   ✓   ✓ ]
[2]  [ ✓   ✓   ✓   ✓   ✓   ✓ ]
...

有掩码（只能看到过去）:
      [1] [2] [3] [4] [5] [6]
[1]  [ ✓   ✗   ✗   ✗   ✗   ✗ ]  ← 位置1只能看位置1
[2]  [ ✓   ✓   ✗   ✗   ✗   ✗ ]  ← 位置2能看位置1-2
[3]  [ ✓   ✓   ✓   ✗   ✗   ✗ ]  ← 位置3能看位置1-3
[4]  [ ✓   ✓   ✓   ✓   ✗   ✗ ]
[5]  [ ✓   ✓   ✓   ✓   ✓   ✗ ]
[6]  [ ✓   ✓   ✓   ✓   ✓   ✓ ]  ← 位置6能看所有

✓ = 可见，✗ = 被遮住
```

### 代码实现

```python
# 1. 创建上三角掩码
context_length = 6
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
# tensor([[0, 1, 1, 1, 1, 1],
#         [0, 0, 1, 1, 1, 1],
#         [0, 0, 0, 1, 1, 1],
#         ...])

# 2. 应用掩码（用 -inf 填充）
attn_scores.masked_fill_(mask.bool(), -torch.inf)

# 3. Softmax（-inf 的位置变成 0）
attn_weights = torch.softmax(attn_scores, dim=-1)
```

### 为什么用 -inf？

```
softmax([1, 2, -inf]) = [0.119, 0.881, 0.000]
                              ↑
                    被遮住的位置权重为0

-inf 经过 softmax 后变成 0，完全不关注被遮住的位置！
```

---

## 3.6 多头注意力（Multi-Head Attention）

### 为什么需要多个头？

**问题**：一个注意力头只能学到一种关系

**例子**：
```
句子: "The animal didn't cross the street because it was too tired"

"it" 这个词：
  - 头1 可能关注 "animal"（"it"指的是"animal"）
  - 头2 可能关注 "tired"（"it"累了）
  - 头3 可能关注 "street"（和位置相关）

多个头 = 捕获不同类型的关系！
```

### 比喻

```
一个头 = 一个"专家"

专家1: 专门学语法关系
专家2: 专门学语义关系
专家3: 专门学位置关系

多个专家一起工作 = 更全面地理解句子！
```

### 结构

```
输入 x
   ↓
   ├─→ Head 1: (Q1, K1, V1) → Attention → output1
   ├─→ Head 2: (Q2, K2, V2) → Attention → output2
   ├─→ Head 3: (Q3, K3, V3) → Attention → output3
   └─→ ...
   ↓
Concat [output1, output2, output3, ...]
   ↓
Linear Projection
   ↓
最终输出
```

### ⚠️ 重要：多头的内容是随机的！

**问题**：每个头学的内容是固定的吗？比如头1一定学语法关系？

**答案**：❌ 不是！是模型训练过程中自然形成的。

```
初始状态：
  - 每个头的权重都是随机的
  - 不知道头1会学什么，头2会学什么

训练过程：
  - 模型自动调整不同头的权重
  - 不同头自然分化出不同的"专长"

训练结果（可能）：
  - 头1可能恰好学到了语法关系
  - 头2可能恰好学到了语义关系
  - 但这是"自然形成的"，不是固定的！

例子：
  训练3次GPT模型：
    模型1: 头1→语法, 头2→语义, 头3→位置
    模型2: 头1→语义, 头2→位置, 头3→语法
    模型3: 头1→位置, 头2→语法, 头3→语义
  
  不同模型学到的东西可能不同！
  但都能很好地完成任务。
```

### 代码实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads):
        super().__init__()
        
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度
        
        # Q, K, V 投影
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        
        # 输出投影
        self.out_proj = nn.Linear(d_out, d_out)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        # 1. 投影到 Q, K, V
        queries = self.W_query(x)  # [b, num_tokens, d_out]
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # 2. 分割成多个头
        # [b, num_tokens, d_out] → [b, num_tokens, num_heads, head_dim]
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # 3. 转置: [b, num_heads, num_tokens, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # 4. 计算注意力（每个头独立）
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        
        # 5. 加权求和
        context_vec = attn_weights @ values
        
        # 6. 合并所有头
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(b, num_tokens, self.d_out)
        
        # 7. 最终投影
        output = self.out_proj(context_vec)
        
        return output
```

---

## 完整流程图

```
输入 Embeddings [batch, seq_len, d_in]
   ↓
┌────────────────────────────────────────┐
│    Multi-Head Attention            │
│                                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  │W_query │  │ W_key  │  │W_value │
│  └────┬────┘  └────┬────┘  └────┬────┘
│       ↓           ↓           ↓       │
│    Query       Key        Value     │
│       │           │           │       │
│       └─────┬─────┴─────┬─────┘       │
│             ↓             ↓           │
│      点积 (相似度)                   │
│             ↓                        │
│      缩放 ÷ sqrt(d_k)               │
│             ↓                        │
│      Causal Mask (遮住未来)          │
│             ↓                        │
│      Softmax (归一化)                │
│             ↓                        │
│      加权求和 (注意力输出)            │
│             ↓                        │
│      合并所有头 (Concat)             │
│             ↓                        │
│      线性投影 (Linear)               │
└────────────────────────────────────────┘
   ↓
输出 [batch, seq_len, d_out]
```

---

## Q&A 洞察

### Q1: Q/K/V 代码都一样，为什么能学到不同功能？

**问题**：Q、K、V 的代码结构相同（都是 Linear 层），怎么确保它们分别学到"我在找什么"、"我包含什么"、"我的内容"？

**回答**：
- **没有显式指定**：模型不知道 Q/K/V 应该是什么意思
- **训练中学会**：通过损失函数的反向传播，三个矩阵自然分化出不同功能
- **类比**：三个学生一起做作业，初始都是空的脑袋。如果配合对了（预测准），就保持；如果错了，一起调整。多次训练后自然分工。

```
初始：随机权重 → 随机输出
    ↓
损失函数"惩罚"错误
    ↓
权重往减少损失的方向调整
    ↓
最终：Q学会"找相关词"，K学会"展示自己"，V学会"传递信息"
```

**关键**：不是"指定" Q/K/V 的角色，而是通过训练让它们**必须配合**才能减少损失，配合过程中自然分化。

### Q2: 多头注意力每个头学什么？

**问题**：12个头是固定的角色吗？比如头1学语法，头2学语义？

**回答**：
- **不是固定的**：每个头的内容是**随机学习**的
- **训练决定**：模型自己决定每个头关注什么
- **可能的结果**：某些头关注语法，某些关注语义，某些关注位置关系等
- **每次训练不同**：重新训练可能得到不同的分配

---

## 关键要点总结

| 概念 | 作用 | 关键点 |
|------|------|--------|
| **Self-Attention** | 让每个词看到所有词 | 点积 → Softmax → 加权求和 |
| **Query/Key/Value** | 三种角色 | 可训练的投影矩阵 |
| **Scaled Dot-Product** | 稳定训练 | 除以 sqrt(d_k) |
| **Causal Mask** | 防止看到未来 | 用 -inf 遮住 |
| **Multi-Head** | 捕获不同关系 | 多个头，内容随机形成 |

---

## 训练过程总结

```
初始：所有权重都是随机的
  - Token Embedding 向量
  - W_query, W_key, W_value 矩阵
  - 多头注意力的权重

训练：通过反向传播调整
  - 目标：预测下一个词
  - 错误 → 调整所有权重

最终：学到有意义的表示
  - Embedding 向量：相似词向量接近
  - QKV 矩阵：学会重要的关系
  - 多头：自然分化出不同专长
```

---

## 运行代码

```bash
cd 02-llm-from-scratch/fundamentals/ch03-attention
python ch03_code.py
```

---

## 下一章预告

第4章将使用本章的注意力机制构建完整的 GPT 模型！