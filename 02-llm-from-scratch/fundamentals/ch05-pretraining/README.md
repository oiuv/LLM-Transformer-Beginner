# 第5章：预训练（Pretraining）

## 概述

第5章让我们把第4章的 GPT 模型训练起来！训练后模型就能生成有意义的文本。

**核心概念（共5个）：**
1. **损失函数（Loss Function）** - 衡量模型预测有多差
2. **训练循环（Training Loop）** - 前向 → 损失 → 反向 → 更新
3. **评估（Evaluation）** - 监控训练损失和验证损失
4. **优化器（Optimizer）** - AdamW，决定如何更新权重
5. **生成样本** - 训练过程中查看生成效果

---

## 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│                      预训练流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 准备数据                                                 │
│     文本 → Token IDs → DataLoader（批次训练数据）             │
│                                                             │
│  2. 初始化模型                                               │
│     GPTModel（随机权重）                                      │
│                                                             │
│  3. 训练循环（重复多次）                                      │
│     ┌─────────────────────────────────────┐                 │
│     │  前向传播: input → model → logits   │                 │
│     │  计算损失: logits vs target → loss  │                 │
│     │  反向传播: loss.backward() → 梯度   │                 │
│     │  更新权重: optimizer.step()         │                 │
│     └─────────────────────────────────────┘                 │
│                                                             │
│  4. 评估 & 生成                                              │
│     监控损失下降，生成文本查看效果                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. 损失函数（Loss Function）

### 作用
衡量模型预测和真实答案的差距。

### 交叉熵损失（Cross-Entropy Loss）

```
模型输出 logits: [0.1, 2.3, -0.5, ...]  (50257个值)
真实答案 token:  15496

损失 = -log(模型预测15496的概率)

预测概率越高 → 损失越低
预测概率越低 → 损失越高
```

### 代码

```python
import torch.nn.functional as F

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    # 前向传播
    logits = model(input_batch)  # (batch, seq_len, vocab_size)
    
    # 计算交叉熵损失
    # flatten(0, 1) 把 batch 和 seq_len 合并
    loss = F.cross_entropy(
        logits.flatten(0, 1),      # (batch*seq_len, vocab_size)
        target_batch.flatten()     # (batch*seq_len,)
    )
    
    return loss
```

### 为什么需要 flatten？

```
原始形状:
  logits: (batch=2, seq_len=4, vocab_size=50257)
  target: (batch=2, seq_len=4)

Flatten 后:
  logits: (8, 50257)   ← 把前两个维度合并
  target: (8,)         ← 对应8个预测任务

CrossEntropyLoss 要求:
  predictions: (N, num_classes)
  targets: (N,)
```

---

## 2. 训练循环（Training Loop）

### 核心四步

```
┌──────────────────────────────────────────────┐
│  1. optimizer.zero_grad()  # 清空梯度        │
│  2. loss = calc_loss(...)  # 计算损失        │
│  3. loss.backward()        # 反向传播算梯度  │
│  4. optimizer.step()       # 更新权重        │
└──────────────────────────────────────────────┘
```

### 代码

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()  # 训练模式（启用dropout）
        
        for input_batch, target_batch in train_loader:
            
            # ===== 核心训练四步 =====
            optimizer.zero_grad()                      # 1. 清空梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                            # 3. 反向传播
            optimizer.step()                           # 4. 更新权重
            # ========================
        
        # 每个 epoch 后评估
        train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 生成样本看看效果
        generate_and_print_sample(model, tokenizer, device, start_context)
    
    return train_losses, val_losses
```

### 四步详解

| 步骤 | 代码 | 作用 |
|------|------|------|
| **1. 清空梯度** | `optimizer.zero_grad()` | 上一批的梯度不要了 |
| **2. 计算损失** | `loss = calc_loss(...)` | 看看预测有多差 |
| **3. 反向传播** | `loss.backward()` | 计算每个参数的梯度 |
| **4. 更新权重** | `optimizer.step()` | 根据梯度调整权重 |

---

## 3. 评估（Evaluation）

### 训练损失 vs 验证损失

```
训练集: 用于训练模型（90%数据）
验证集: 用于检查效果（10%数据）
```

### 代码

```python
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 评估模式（关闭dropout）
    
    with torch.no_grad():  # 不计算梯度（省内存）
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    
    model.train()  # 切回训练模式
    return train_loss, val_loss
```

### 损失曲线解读

```
Loss
  │
  │  Train ────────
  │       ╲
  │        ╲
  │         ╲────── Val
  │               ╲
  │                ╲
  └─────────────────── Epochs

理想情况: 两者都下降，且接近
过拟合:   Train↓ Val↑（记住训练数据，不能泛化）
欠拟合:   两者都很高（模型太简单或训练不够）
```

---

## 4. 优化器（Optimizer）

### AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,           # 学习率
    weight_decay=0.1   # 权重衰减（正则化）
)
```

### 参数解释

| 参数 | 值 | 作用 |
|------|-----|------|
| **lr (learning_rate)** | 5e-4 = 0.0005 | 步长，太大不稳定，太小太慢 |
| **weight_decay** | 0.1 | 防止过拟合，惩罚大权重 |

### 为什么用 AdamW？

- **Adam**: 自适应学习率，每个参数有自己的学习率
- **AdamW**: 改进版的 Adam，权重衰减实现更正确
- 现代 Transformer 训练的标准选择

---

## 5. 生成样本

### 训练过程中查看效果

```python
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=50,
            context_size=256
        )
    
    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    print(decoded_text)
    
    model.train()
```

### 训练前后对比

```
训练前:
"Every effort moves you" → "Every effort moves you Featureiman Byeswick..."
                                    ↑ 随机乱码

训练后:
"Every effort moves you" → "Every effort moves you, I know, the picture..."
                                    ↑ 有意义的文本！
```

---

## 完整训练配置

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,    # 缩短到256（原1024），加快训练
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

OTHER_SETTINGS = {
    "learning_rate": 5e-4,    # 学习率
    "num_epochs": 10,         # 训练轮数
    "batch_size": 2,          # 批次大小
    "weight_decay": 0.1       # 权重衰减
}
```

---

## 数据准备

```python
# 读取文本
with open("the-verdict.txt", "r") as f:
    text_data = f.read()

# 划分训练/验证集
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))

train_data = text_data[:split_idx]    # 前90%
val_data = text_data[split_idx:]      # 后10%

# 创建 DataLoader
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=256,
    stride=256,
    shuffle=True
)
```

---

## 训练过程可视化

```
Epoch 1 (Step 000000): Train loss 9.781, Val loss 9.933
Every effort moves you Featureiman Byeswickattribute argue...

Epoch 2 (Step 000005): Train loss 8.147, Val loss 8.259
Every effort moves you, I know the picture...

Epoch 5 (Step 000020): Train loss 1.214, Val loss 1.352
Every effort moves you, I know, the picture still...

Epoch 10 (Step 000045): Train loss 0.391, Val loss 0.482
Every effort moves you, I know, the picture still 
remains faintly upon the retina of my consciousness...
```

**损失从 9.78 → 0.39，生成的文本变得有意义！**

---

## 关键概念总结

| 概念 | 作用 | 关键点 |
|------|------|--------|
| **损失函数** | 衡量预测差距 | 交叉熵，概率越高损失越低 |
| **训练循环** | 核心训练流程 | 清梯度→算损失→反向→更新 |
| **评估** | 监控效果 | 训练损失 vs 验证损失 |
| **优化器** | 更新权重 | AdamW，自适应学习率 |
| **生成样本** | 查看效果 | 训练中生成文本观察 |

---

## 与第4章的联系

```
第4章: 搭建了 GPT 模型（随机权重，输出乱码）
         ↓
第5章: 训练模型（通过损失函数和优化器调整权重）
         ↓
结果: 模型权重被优化，能生成有意义的文本
```

**理解了训练过程，回头看第4章的组件（LayerNorm、Attention、FeedForward）就知道它们是如何被训练的了！**

---

## Q&A 洞察

### Q1: 训练效果不好怎么办？

**观察**：训练50轮后，生成文本还是有重复，不够流畅。

**原因分析**：

| 因素 | 我们的演示 | 真正的 GPT |
|------|-----------|-----------|
| 数据量 | 20KB | 数百 GB |
| 模型大小 | 124M 参数 | 124M ~ 175B |
| 训练时间 | 几分钟 | 数周/数月 |

**结论**：模型容量 >> 数据量，模型开始"背诵"而非"学习"。

### Q2: 什么是过拟合？怎么发现？

**定义**：模型死记硬背训练数据，不能泛化到新数据。

**表现**：
```
训练损失: 4.40 → 0.05  ↓↓↓ 持续下降
验证损失: 6.76 → 6.20 → 7.60  ↓↓↑ 先降后升
         ↑
         验证损失开始上升 = 过拟合信号
```

**解决方法**：
- 早停（Early Stopping）：在验证损失最低点停止
- 增加数据
- 减小模型
- 正则化（weight_decay, dropout）

### Q3: 最佳模型在哪里？

**实验结果**：
```
Epoch 13: 验证损失 = 6.1998 ← 最低点，最佳模型
Epoch 50: 验证损失 = 7.6026 ← 过拟合了
```

**实践**：保存验证损失最低的模型，而非训练到最后。

---

## 运行代码

```bash
cd 02-llm-from-scratch/fundamentals/ch05-pretraining
python ch05_code.py
```

---

## 下一步

- 第6章：微调（Fine-tuning）针对特定任务
