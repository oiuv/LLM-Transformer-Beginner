# 第6章：微调（Fine-tuning）

## 概述

**微调** = 用预训练好的模型 + 少量有标签数据 → 完成特定任务

```
预训练（第5章）: 学会语言（预测下一个词）
微调（第6章）:   学会任务（分类、问答、摘要等）
```

---

## 预训练 vs 微调

| 对比 | 预训练 | 微调 |
|------|--------|------|
| **目标** | 预测下一个词 | 完成特定任务 |
| **数据** | 海量无标签文本 | 少量有标签数据 |
| **时间** | 数周/数月 | 数小时/数天 |
| **成本** | 非常高 | 相对低 |

---

## 本章任务：垃圾短信分类

```
输入: 短信内容
输出: "垃圾短信" 或 "正常短信"

例子:
"WINNER!! You won $1000!" → spam (垃圾)
"Hey, are you coming?"    → ham  (正常)
```

---

## 微调流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. 加载预训练权重                                          │
│     download OpenAI GPT-2 weights → load into our model     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 修改输出层                                              │
│     原输出: (batch, seq, 50257) 预测下一个词                │
│     新输出: (batch, 2) 分类（垃圾/正常）                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 准备分类数据                                            │
│     短信文本 → token IDs → 标签 (0/1)                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. 微调训练                                                │
│     分类损失 → 反向传播 → 更新权重                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  5. 评估                                                    │
│     准确率、精确率、召回率                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. 加载预训练权重

### 为什么用预训练权重？

```
从零训练:
- 需要海量数据
- 需要很长时间
- 效果不一定好

用预训练权重:
- OpenAI 已经训练好了 GPT-2
- 已经学会语言知识
- 只需要"微调"到特定任务
```

### 代码

```python
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt

# 下载 OpenAI 的 GPT-2 权重
params, params_hparams = download_and_load_gpt2(
    model_size="124M", 
    models_dir="gpt2"
)

# 加载到我们的模型
model = GPTModel(GPT_CONFIG_124M)
load_weights_into_gpt(model, params)

# 现在模型已经"懂语言"了！
```

---

## 2. 修改输出层

### 原始 GPT 输出

```python
# 原始: 预测下一个词
logits = model(input_ids)  # (batch, seq_len, 50257)
# 用最后一个位置的 logits 预测
```

### 分类任务输出

```python
# 方案1: 用最后一个 token 的输出
last_token_output = model output[:, -1, :]  # (batch, 768)

# 方案2: 添加分类头
classifier = nn.Linear(768, 2)  # 2类: 垃圾/正常
logits = classifier(last_token_output)  # (batch, 2)
```

### 代码

```python
class GPTClassifier(nn.Module):
    def __init__(self, gpt_model, num_classes=2):
        super().__init__()
        self.gpt = gpt_model
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids):
        # GPT 输出
        with torch.no_grad():  # 可以冻结 GPT 权重
            hidden_states = self.gpt(input_ids)
        
        # 用最后一个 token
        last_hidden = hidden_states[:, -1, :]
        
        # 分类
        logits = self.classifier(last_hidden)
        return logits
```

---

## 3. 准备分类数据

### 数据格式

```csv
Label,Text
ham,"Go until jurong point, crazy.."
spam,"WINNER!! You won $1000!"
ham,"Hey, are you coming tonight?"
```

### 数据处理

```python
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length):
        self.data = pd.read_csv(csv_file)
        
        # 编码文本
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        
        # 截断/填充到固定长度
        self.encoded_texts = [
            tokens[:max_length] + [50256] * (max_length - len(tokens))
            for tokens in self.encoded_texts
        ]
        
        # 标签: ham=0, spam=1
        self.labels = torch.tensor(
            [1 if label == "spam" else 0 for label in self.data["Label"]]
        )
    
    def __getitem__(self, idx):
        return torch.tensor(self.encoded_texts[idx]), self.labels[idx]
```

---

## 4. 微调训练

### 训练循环（和第5章类似）

```python
def train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        
        for input_ids, labels in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            logits = model(input_ids)
            
            # 分类损失
            loss = F.cross_entropy(logits, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
        
        # 评估
        train_acc = calc_accuracy(model, train_loader)
        val_acc = calc_accuracy(model, val_loader)
        print(f"Epoch {epoch+1}: Train Acc {train_acc:.2%}, Val Acc {val_acc:.2%}")
```

### 冻结 vs 微调全部

| 策略 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| **冻结 GPT** | 只训练分类头 | 快，防过拟合 | 可能不够好 |
| **微调全部** | 训练所有参数 | 效果更好 | 慢，可能过拟合 |

```python
# 冻结 GPT 权重
for param in model.gpt.parameters():
    param.requires_grad = False

# 只训练分类头
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=5e-5)
```

---

## 5. 评估指标

### 准确率（Accuracy）

```
准确率 = 正确预测数 / 总预测数
```

### 精确率、召回率、F1

```
精确率(Precision) = 预测为垃圾且真的是垃圾 / 预测为垃圾
召回率(Recall)    = 预测为垃圾且真的是垃圾 / 真实垃圾数
F1 = 2 × (精确率 × 召回率) / (精确率 + 召回率)
```

### 代码

```python
def calc_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for input_ids, labels in data_loader:
            logits = model(input_ids)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return correct / total
```

---

## 微调策略

### 1. 特征提取（Feature Extraction）

```python
# 冻结 GPT，只训练分类头
for param in model.gpt.parameters():
    param.requires_grad = False
```

**适用**：数据少，防过拟合

### 2. 微调最后几层

```python
# 冻结大部分层，只训练最后2-3层
for param in model.gpt.trf_blocks[:-2].parameters():
    param.requires_grad = False
```

**适用**：中等数据量

### 3. 微调全部

```python
# 所有参数都训练
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
```

**适用**：数据多，追求最好效果

---

## 与前几章的联系

```
第2章: 文本 → Token IDs
第3章: 注意力机制
第4章: GPT 架构
第5章: 预训练（学会语言）
第6章: 微调（学会任务）← 你在这里
```

---

## 关键概念总结

| 概念 | 说明 |
|------|------|
| **预训练权重** | 用 OpenAI 训练好的 GPT-2 |
| **分类头** | 替换输出层，适配分类任务 |
| **有监督学习** | 用标签数据训练 |
| **冻结** | 固定某些层的权重 |
| **微调策略** | 根据数据量选择冻结程度 |

---

## Q&A 洞察

### Q1: 为什么微调只需少量数据，而预训练需要海量数据？

**问题**：预训练要万亿 token，微调只要几千条短信就能学会分类，差距为什么这么大？

**回答**：
- **预训练阶段**：模型从零开始学语言——语法、词义、世界知识。这些是**通用能力**，必须靠海量数据才能学全。
- **微调阶段**：模型已经懂语言了，只是不知道"什么是垃圾短信"。微调本质是**在已有能力上加一层任务适配**，不需要重新学语言。
- **类比**：预训练像读完九年义务教育，微调像上岗培训。前者要十几年，后者可能只要几天。

```
预训练：学会"读"和"理解"中文（海量数据，无标签）
    ↓
微调：学会"判断垃圾短信"这个具体任务（少量数据，有标签）
```

**关键**：微调不是从零学起，而是把预训练模型的通用能力"引导"到具体任务上。

### Q2: 分类头（classifier）为什么是新建的，不能复用 lm_head？

**问题**：预训练模型的 `lm_head` 输出 vocab_size 维（如 50000），分类任务只需要 2 维（spam/ham）。为什么不直接取 lm_head 的前 2 维？

**回答**：
- **语义不对应**：lm_head 的 50000 维对应 50000 个 token 的概率，前 2 维只是"第 1、2 个 token 的概率"，和"垃圾/正常"毫无关系。
- **新建分类头**：`nn.Linear(emb_dim, 2)` 是一个全新的随机初始化层，从零学"特征向量 → 2 类概率"的映射。
- **结构**：复用的是**前面的 transformer body**（提取特征），不是**最后的输出头**（任务相关）。

```python
# 预训练模型：input → transformer body → lm_head → 50000 维 token 概率
# 微调模型：  input → transformer body → classifier → 2 维类别概率
#                          ↑复用            ↑新建
```

**关键**：transformer body 是"特征提取器"，可跨任务复用；输出头是"任务适配器"，每个任务独立。

### Q3: 冻结 vs 全参数微调，怎么选？

**问题**：什么时候该冻结 GPT 只训分类头，什么时候该全参数微调？

**回答**：
- **数据量小（< 1万条）**：**冻结 GPT，只训分类头**。原因：数据少时全参数微调容易过拟合，模型会"记住"训练数据而非学到规律。
- **数据量中（1万-10万）**：**微调最后 2-3 层**。让模型的高层特征适配任务，但不动底层（底层特征是通用的，不值得为小数据冒险重训）。
- **数据量大（> 10万）**：**全参数微调**。数据足够支撑全模型重新调整，效果最好。
- **学习率差异**：冻结时可用稍大 lr（5e-5）；全参数微调要用小 lr（1e-5），避免破坏预训练学到的知识。

```
数据少 → 冻结 → 防过拟合，但上限低
数据多 → 全调 → 效果好，但要小心"灾难性遗忘"
```

**关键**：预训练模型是宝贵资产，微调时要"够用就好"——能用冻结解决就别全调。

---

## 运行代码

```bash
cd 02-llm-from-scratch/fundamentals/ch06-finetuning

# 朴素实现（手写训练循环）
python ch06_demo.py

# HuggingFace 版（用 transformers + Trainer）
python ch06_hf_demo.py
```

---

## 下一步

- 第7章：指令微调（Instruction Fine-tuning）
- 第8章：人类反馈强化学习（RLHF）
