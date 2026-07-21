# 第2章：文本数据处理

## 学习目标

完成本章后,你将理解:
- 如何将文本分割成token
- 如何构建词表(vocabulary)
- 如何实现token到ID的映射
- 如何使用BPE tokenizer
- 如何创建数据加载器(dataloader)
- 如何实现token embedding和positional embedding

---

## 2.1 词嵌入简介

LLM无法直接处理文本,需要将文本转换为数值表示。

**核心概念:**
- **Token**: 文本的最小单位(词/子词/字符)
- **Embedding**: 将离散的token映射到连续向量空间

```
文本: "Hello World"
  ↓ Tokenization
Tokens: ["Hello", "World"]
  ↓ Embedding
向量: [[0.1, 0.2, ...], [0.3, 0.4, ...]]
```

---

## 2.2 文本分词 (Tokenization)

### 简单分词器实现
使用正则表达式将文本分割成token:

```python
import re

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\'']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
# ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

**关键点:**
- `re.split` 的第一个参数是正则表达式模式
- `()` 表示捕获分组,保留分隔符
- `|` 表示或,匹配多种标点符号
- `item.strip()` 去除空白
- `if item.strip()` 过滤空字符串

---

## 2.3 Token到ID的转换
构建词表(vocabulary)将token映射到唯一整数ID:

```python
# 所有唯一token
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)  # 1130

# 构建映射字典
vocab = {token: integer for integer, token in enumerate(all_words)}
# {'!': 0, '"': 1, "'": 2, ...}
```

### 简单分词器类
```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text):
        # 文本 → token IDs
        preprocessed = re.split(r'([,.:;?_!"()\'']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        # token IDs → 文本
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\''])', r'\1', text)  # 修复标点前的空格
        return text
```

---

## 2.4 特殊Token
处理未知词和文本边界:

| Token | 用途 |
|-------|------|
| `<|unk|>` | 未知词(不在词表中) |
| `<|endoftext|>` | 文本结束/分隔 |

```python
class SimpleTokenizerV2:
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\'']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # 处理未知词
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
```

---

## 2.5 Byte Pair Encoding (BPE)
GPT-2使用的分词算法,能处理任意词:

**优点:**
- 将未知词分解为子词单元
- 词表大小可控
- 能处理拼写错误和新词

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, ...]

strings = tokenizer.decode(integers)
# "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
```

**BPE原理图:**
```
"someunknownPlace" 
  → ["some", "unknown", "Place"]  (如果在词表中)
  → 或 ["some", "un", "known", "Place"]  (子词分解)
  → 或 ["s", "o", "m", "e", ...]  (字符级分解)
```

---

## 2.6 滑动窗口数据采样
为下一个词预测任务准备数据:

**原理:**
```
输入序列: [1, 2, 3, 4, 5, 6, 7, 8]
context_size = 4

训练样本:
输入:  [1, 2, 3, 4]  → 目标: [2, 3, 4, 5]
输入:  [2, 3, 4, 5]  → 目标: [3, 4, 5, 6]
输入:  [3, 4, 5, 6]  → 目标: [4, 5, 6, 7]
...
```

### Dataset实现
```python
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

**参数说明:**
- `max_length`: 每个样本的token数(上下文长度)
- `stride`: 滑动步长(重叠量)

### DataLoader
```python
def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader
```

---

## 2.7 Token Embedding
将token ID转换为稠密向量:

```python
import torch.nn as nn

vocab_size = 6      # 词表大小
output_dim = 3      # 嵌入维度

embedding_layer = nn.Embedding(vocab_size, output_dim)
# 权重矩阵: [6, 3]

input_ids = torch.tensor([2, 3, 5, 1])
token_embeddings = embedding_layer(input_ids)
# shape: [4, 3]
```

**原理:**
- Embedding层本质上是一个查表操作
- 输入token ID,输出对应的嵌入向量
- 权重在训练中学习

---

## 2.8 Positional Embedding
为token添加位置信息:

**为什么需要?**
- Token embedding不包含位置信息
- "我爱你"和"你爱我"的token embedding相同,但意义不同

```python
vocab_size = 50257
output_dim = 256
max_length = 4

token_embedding_layer = nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = nn.Embedding(max_length, output_dim)

# 输入
input_ids = torch.tensor([1, 2, 3, 4])

# Token embedding
token_embeddings = token_embedding_layer(input_ids)  # [4, 256]

# Positional embedding
pos_embeddings = pos_embedding_layer(torch.arange(max_length))  # [4, 256]

# 最终输入
input_embeddings = token_embeddings + pos_embeddings  # [4, 256]
```

---

## 完整流程图

```
原始文本
  ↓ Tokenization (BPE)
Token IDs
  ↓ Token Embedding
Token Embeddings [seq_len, embed_dim]
  ↓ + Positional Embedding
Input Embeddings [seq_len, embed_dim]
  ↓
送入Transformer
```

---

## 运行代码

1. **运行完整代码:**
   ```bash
   cd 02-llm-from-scratch/fundamentals/ch02-text-data
   python ch02_code.py
   ```

2. **运行测试:**
   ```bash
   python ch02_test.py
   ```

---

## 关键要点总结

1. **Tokenization**: 将文本分割成token
2. **Vocabulary**: 构建token到ID的映射
3. **BPE**: 子词分词,处理未知词
4. **DataLoader**: 滑动窗口生成训练样本
5. **Token Embedding**: 将ID转为向量
6. **Positional Embedding**: 添加位置信息
