"""
对应 train.py 阶段 2:训练 BPE 分词器(简化版,用小语料演示)
"""
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. 准备语料(模拟 train.py 的 paragraphs)
corpus = [
    "第一章 重生\n\n林凡睁开眼睛,发现自己回到了十年前。\n\n",
    "第二章 修炼\n\n灵气复苏,万物生长。林凡踏入修仙之路。\n\n",
    "第三章 战斗\n\n剑光如虹,一招破万法。林凡大获全胜。\n\n",
]

# 2. 创建 BPE tokenizer(跟 train.py 一模一样)
tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(
    vocab_size=100,  # 故意设小,方便看词表
    special_tokens=["<|pad|>", "<|unk|>", "<s>", "</s>"],
    min_frequency=2,
)

# 3. 训练
tokenizer.train_from_iterator(corpus, trainer)
print(f"词表大小: {tokenizer.get_vocab_size()}")

# 4. 看词表里都有啥
vocab = tokenizer.get_vocab()
print(f"\n部分词表(前 30 个):")
for i, (tok, idx) in enumerate(sorted(vocab.items(), key=lambda x: x[1])[:30]):
    print(f"  {idx:3d} → {tok!r}")

# 5. 编码测试
test = "林凡睁开眼睛,发现自己回到了十年前。"
encoded = tokenizer.encode(test)
print(f"\n原文: {test}")
print(f"Token 数: {len(encoded)}")
print(f"Token ids: {encoded.ids}")
print(f"Token 字符串: {encoded.tokens}")

# 6. 解码
decoded = tokenizer.decode(encoded.ids)
print(f"\n解码: {decoded}")