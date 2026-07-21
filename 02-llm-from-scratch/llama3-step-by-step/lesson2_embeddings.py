"""
阶段 2: 嵌入层 (Embeddings)
理解"数字如何变成向量"
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("阶段 2: 嵌入层 (Embeddings)")
print("=" * 60)

# ============================================
# 第一部分：理解嵌入矩阵
# ============================================

print("\n" + "=" * 60)
print("第一部分：理解嵌入矩阵")
print("=" * 60)

# 定义一个小词汇表（为了演示方便）
vocab = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '我': 3,
    '爱': 4,
    '学': 5,
    '习': 6,
    '猫': 7,
    '狗': 8,
    '动物': 9,
}

vocab_size = len(vocab)
embedding_dim = 8  # 用小维度方便演示（Llama3 用 4096）

print(f"\n【词汇表大小】: {vocab_size}")
print(f"【嵌入维度】: {embedding_dim}")
print(f"【嵌入矩阵形状】: [{vocab_size}, {embedding_dim}]")

# 创建嵌入矩阵（随机初始化）
# 在真实模型中，这是训练出来的
torch.manual_seed(42)  # 固定随机种子，方便复现
embedding_matrix = torch.randn(vocab_size, embedding_dim)

print("\n【嵌入矩阵】（每个Token对应一个向量）:")
print(f"形状: {embedding_matrix.shape}")
print("\n前5个Token的嵌入向量:")
for i in range(min(5, vocab_size)):
    token = list(vocab.keys())[i]
    vector = embedding_matrix[i]
    print(f"  Token {i} ('{token}'): {vector[:4].numpy()}... (显示前4维)")

# ============================================
# 第二部分：嵌入查找过程
# ============================================

print("\n" + "=" * 60)
print("第二部分：嵌入查找过程")
print("=" * 60)

# 模拟阶段1的Token序列
tokens = torch.tensor([1, 3, 4, 5, 6, 2])  # <START> 我 爱 学 习 <END>
token_names = ['<START>', '我', '爱', '学', '习', '<END>']

print(f"\n【输入Token序列】: {tokens.tolist()}")
print(f"【对应词汇】: {token_names}")
print(f"【序列长度】: {len(tokens)}")

# 查找嵌入向量
# 方法1: 直接索引
embeddings_manual = embedding_matrix[tokens]

print(f"\n【查找结果】每个Token变成向量:")
print(f"输入形状: {tokens.shape} → 输出形状: {embeddings_manual.shape}")
print(f"解释: [{len(tokens)}] 个Token，每个变成 [{embedding_dim}] 维向量")

print("\n每个Token的嵌入向量（显示前4维）:")
for i, (token_id, name) in enumerate(zip(tokens, token_names)):
    vec = embeddings_manual[i]
    print(f"  '{name}' (ID={token_id}): {vec[:4].numpy()}...")

# 方法2: 使用 PyTorch 的 Embedding 层
print("\n【使用 PyTorch Embedding 层】")
embedding_layer = nn.Embedding(vocab_size, embedding_dim)
embedding_layer.weight.data = embedding_matrix  # 复制我们的矩阵

embeddings_pytorch = embedding_layer(tokens)
print(f"PyTorch Embedding 输出形状: {embeddings_pytorch.shape}")
print(f"与手动查找结果相同: {torch.allclose(embeddings_manual, embeddings_pytorch)}")

# ============================================
# 第三部分：语义相似度（核心概念）
# ============================================

print("\n" + "=" * 60)
print("第三部分：语义相似度（核心概念）")
print("=" * 60)

print("""
【关键思想】
在嵌入空间中：
- 语义相似的词，向量距离近
- 语义无关的词，向量距离远

距离度量：余弦相似度 (Cosine Similarity)
- 值域: [-1, 1]
- 1: 完全相同方向（语义相似）
- 0: 正交（无关）
- -1: 相反方向（语义相反）
""")

# 计算余弦相似度
def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度"""
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))

# 获取几个词的嵌入
vec_我 = embedding_matrix[vocab['我']]
vec_爱 = embedding_matrix[vocab['爱']]
vec_猫 = embedding_matrix[vocab['猫']]
vec_狗 = embedding_matrix[vocab['狗']]
vec_动物 = embedding_matrix[vocab['动物']]

print("\n【语义相似度计算】")
print("-" * 40)

# 相似的词
sim_猫_狗 = cosine_similarity(vec_猫, vec_狗)
print(f"'猫' vs '狗': {sim_猫_狗:.4f} ← 应该比较接近（都是动物）")

sim_猫_动物 = cosine_similarity(vec_猫, vec_动物)
print(f"'猫' vs '动物': {sim_猫_动物:.4f} ← 应该比较接近")

# 不太相关的词
sim_我_猫 = cosine_similarity(vec_我, vec_猫)
print(f"'我' vs '猫': {sim_我_猫:.4f} ← 可能不太相关")

sim_爱_狗 = cosine_similarity(vec_爱, vec_狗)
print(f"'爱' vs '狗': {sim_爱_狗:.4f} ← 可能不太相关")

print("-" * 40)
print("注意：因为是随机初始化的，这些值没有实际意义")
print("在训练好的模型中，相似词的相似度会很高！")

# ============================================
# 第四部分：可视化嵌入空间
# ============================================

print("\n" + "=" * 60)
print("第四部分：可视化嵌入空间")
print("=" * 60)

# 为了可视化，我们降到2维
# 用 numpy 手写 PCA（避免引入 sklearn 依赖）
# PCA 原理：数据中心化后做 SVD 分解，取前 k 个主成分
def pca_2d(X):
    """numpy 实现的 PCA，降到 2 维。
    步骤：
    1. 中心化：减去均值
    2. SVD 分解：X = U S V^T，V 的列就是主成分方向
    3. 投影：X @ V[:, :2] 得到前两个主成分坐标
    """
    X_centered = X - X.mean(axis=0, keepdims=True)
    # SVD：U (n,k), S (k,), Vt (k, d) —— Vt 的行就是主成分方向
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:2].T  # 投影到前两个主成分

# 提取所有词的嵌入
all_embeddings = embedding_matrix.numpy()
all_labels = list(vocab.keys())

# PCA降维到2D
embeddings_2d = pca_2d(all_embeddings)

# 绘制
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     c=range(len(all_labels)), cmap='tab10', s=200)

# 添加标签
for i, label in enumerate(all_labels):
    ax.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=12, ha='center', va='bottom')

ax.set_xlabel('PCA 维度 1', fontsize=12)
ax.set_ylabel('PCA 维度 2', fontsize=12)
ax.set_title('Token 嵌入空间可视化 (PCA降维)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('embeddings_visualization.png', dpi=150, bbox_inches='tight')
print("\n【可视化】已保存为 'embeddings_visualization.png'")
print("观察：在训练好的模型中，相似的词会在图中聚集在一起")

# ============================================
# 第五部分：Llama3 的嵌入层
# ============================================

print("\n" + "=" * 60)
print("第五部分：Llama3 的嵌入层")
print("=" * 60)

print("""
【Llama3-8B 的嵌入层参数】

词汇表大小: 128,256
嵌入维度: 4,096
嵌入矩阵形状: [128256, 4096]

参数量计算:
  128,256 × 4,096 = 525,336,576 ≈ 5.25 亿参数

这只是输入嵌入！输出层还有一个同样大小的矩阵。

内存占用:
  525,336,576 参数 × 2 字节 (bf16) ≈ 1 GB
""")

# 模拟 Llama3 的嵌入层（用小规模演示）
llama_vocab_size = 128256
llama_embedding_dim = 4096

print(f"\n【模拟 Llama3 嵌入层】")
print(f"词汇表大小: {llama_vocab_size:,}")
print(f"嵌入维度: {llama_embedding_dim}")
print(f"矩阵形状: [{llama_vocab_size}, {llama_embedding_dim}]")

# 计算参数量
total_params = llama_vocab_size * llama_embedding_dim
print(f"\n参数量: {total_params:,}")
print(f"       ≈ {total_params / 1e9:.2f} B (十亿)")

# 模拟一个序列的嵌入转换
seq_length = 17  # 例如 "the answer to the ultimate question..."
sample_tokens = torch.randint(0, llama_vocab_size, (seq_length,))

print(f"\n【示例】Token 序列转换:")
print(f"输入: [{seq_length}] 个 Token IDs")
print(f"      例如: {sample_tokens[:5].tolist()}...")
print(f"输出: [{seq_length}, {llama_embedding_dim}] 的嵌入矩阵")
print(f"      形状: torch.Size([{seq_length}, {llama_embedding_dim}])")

# ============================================
# 第六部分：嵌入层的本质
# ============================================

print("\n" + "=" * 60)
print("第六部分：嵌入层的本质")
print("=" * 60)

print("""
【嵌入层 = 可学习的查找表】

1. 初始化时：随机值（没有语义）
2. 训练过程中：模型学习调整这些向量
3. 训练完成后：
   - "国王" - "男人" + "女人" ≈ "女王"
   - "北京" - "中国" + "日本" ≈ "东京"

【类比理解】

想象每个词是一个"人"：
- Token ID = 身份证号（只是编号，没有信息）
- 嵌入向量 = 这个人的"特征描述"
  * 身高、体重、性格、爱好...（4096个维度）
  * 相似的人（词）会有相似的特征

【数学本质】

嵌入层就是一个矩阵乘法（One-Hot 编码）:

Token ID = 3 → One-Hot = [0, 0, 0, 1, 0, 0, ...]

[0, 0, 0, 1, 0, ...] × 嵌入矩阵 = 嵌入矩阵的第3行

所以嵌入层 = 从矩阵中查一行
""")

# 演示 One-Hot 编码
print("\n【One-Hot 编码演示】")
token_id = 3
one_hot = torch.zeros(vocab_size)
one_hot[token_id] = 1

print(f"Token ID {token_id} ('我') 的 One-Hot 编码:")
print(f"{one_hot.numpy()}")

# One-Hot × 嵌入矩阵 = 嵌入向量
embedding_via_matmul = torch.matmul(one_hot, embedding_matrix)
embedding_direct = embedding_matrix[token_id]

print(f"\nOne-Hot × 嵌入矩阵 = {embedding_via_matmul[:4].numpy()}...")
print(f"直接查找         = {embedding_direct[:4].numpy()}...")
print(f"结果相同: {torch.allclose(embedding_via_matmul, embedding_direct)}")

# ============================================
# 第七部分：总结
# ============================================

print("\n" + "=" * 60)
print("第七部分：总结")
print("=" * 60)

print("""
【本阶段重点】

1. 嵌入层的作用：
   Token ID (整数) → 密集向量 (语义表示)

2. 核心机制：
   - 嵌入矩阵: [词汇表大小, 嵌入维度]
   - 查找操作: matrix[token_id]
   - 等价于: One-Hot 编码 × 嵌入矩阵

3. 语义空间：
   - 相似词在向量空间中距离近
   - 用余弦相似度度量语义相关性

4. Llama3 规模：
   - 128,256 词汇 × 4,096 维度
   - 约 5.25 亿参数

【流程回顾】
文本 → Tokenization → Token IDs → Embedding → 向量序列
"我"              3           [0.2, -0.5, ...]

【下一步】
这些向量序列接下来要进入注意力机制，
学习词与词之间的关系！
""")

print("\n" + "=" * 60)
print("阶段 2 完成！")
print("=" * 60)
