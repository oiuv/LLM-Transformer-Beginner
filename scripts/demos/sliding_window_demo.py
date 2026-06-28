"""
对应 train.py 阶段 3:用 train.py 同款中文 BPE,演示滑动窗口
"""
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. 训练一个迷你 BPE(同 train.py 逻辑)
corpus = [
    "第一章 重生\n\n林凡回到了十年前。\n\n",
    "第二章 修炼\n\n灵气复苏 修仙之路。\n\n",
    "第三章 战斗\n\n剑光如虹 一招破万法。\n\n",
]
tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator(corpus, BpeTrainer(
    vocab_size=200, min_frequency=1,
    special_tokens=["<|pad|>", "<|unk|>", "<s>", "</s>"],
))

# 2. 编码一段测试文本
text = "林凡回到了十年前。灵气复苏,他开始修炼。"
ids = tokenizer.encode(text).ids
tokens = [tokenizer.decode([i]) for i in ids]
print(f"原文: {text}")
print(f"Token 数: {len(ids)}")
print(f"Tokens: {tokens}")
print(f"IDs: {ids}\n")

# 3. 滑动窗口
CONTEXT = 5
print(f"=== 滑动窗口 (context_length={CONTEXT}) ===")
for start in range(0, len(ids) - CONTEXT):
    chunk = ids[start : start + CONTEXT + 1]
    input_ids = chunk[:-1]
    labels = chunk[1:]
    in_tok = [tokenizer.decode([i]) for i in input_ids]
    lbl_tok = [tokenizer.decode([i]) for i in labels]
    print(f"样本 {start}: 输入 {in_tok}  →  预测 {lbl_tok}")