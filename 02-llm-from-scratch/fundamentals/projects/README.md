# 学习项目

实践项目，应用所学知识。

---

## 项目列表

### 1. 垃圾短信分类器

**目录**: `spam-classifier/`

**对应章节**: 第 6 章 - 微调（分类任务）

**技术**: GPT-2 微调、类别权重、冻结参数

**成果**: 
- 准确率: 96.7%
- F1 分数: 87.6%

**学习要点**:
- 如何微调预训练模型
- 解决数据不平衡问题
- 冻结策略的应用

**运行**:
```bash
cd spam-classifier
python train.py    # 训练
python predict.py  # 预测
```

---

### 2. 指令微调

**目录**: `instruction-finetuning/`

**对应章节**: 第 7 章 - 指令微调（SFT）

**技术**: GPT-2 指令微调、Prompt 模板、损失掩码

**简介**: 基于 GPT-2 的中文指令微调实践项目。训练一个能理解和执行指令的模型，支持翻译、摘要、问答等 9 种任务类型。数据集包含 50 条中文指令。

**学习要点**:
- 指令数据格式（instruction / input / output）
- Prompt 模板的设计与统一
- 只在回复部分计算损失（loss mask）

**运行**:
```bash
cd instruction-finetuning
python train.py     # 训练
python predict.py   # 推理
```

---

### 3. 诡秘之主 GPT

**目录**: `guimi-gpt/`

**对应章节**: 第 5 章 - 预训练（从零训练）

**技术**: GPT-2 Small (124M) 从零预训练、中文 BPE 分词器

**简介**: 使用完整的《诡秘之主》小说（约 1440 万字符）训练一个能生成小说风格中文文本的 GPT 模型。这是本项目 `03-practice/chinese-gpt/` 的进阶版本——使用更大的数据集和完整的训练流程。

**学习要点**:
- 中文 BPE 分词器训练
- 从零训练 GPT 模型（非微调）
- 长文本生成的采样策略

**运行**:
```bash
cd guimi-gpt
python train.py     # 训练
python generate.py  # 生成
```

---

## 未来项目

- MoE 版本小说 GPT（参考第 9 章实现）
- 文本生成
- 问答系统
- 机器翻译
- 代码补全
