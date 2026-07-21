"""
阶段 4: 位置编码 RoPE (Rotary Position Embedding)
理解"旋转位置编码"：让模型感知位置
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("阶段 4: 位置编码 RoPE (Rotary Position Embedding)")
print("=" * 70)

# ============================================
# 第一部分：为什么需要位置编码
# ============================================

print("\n" + "=" * 70)
print("第一部分：为什么需要位置编码？")
print("=" * 70)

print("""
【问题】

句子 A: "我 爱 学 习"
句子 B: "学 习 让 我 快 乐"

两个句子都有"我"，但：
- 句子 A: "我"是主语（在句首）
- 句子 B: "我"是宾语（在句中）

【注意力机制的问题】

阶段 3 的注意力只关心"词与词的关系"，不关心"词在哪里"。
对两个"我"的处理完全相同！

【解决方案】

给每个词添加"位置信息"，让模型知道：
- "我"在位置 0 → 可能是主语
- "我"在位置 3 → 可能是宾语
""")

# ============================================
# 第二部分：绝对位置编码（简单版）
# ============================================

print("\n" + "=" * 70)
print("第二部分：绝对位置编码（对比理解）")
print("=" * 70)

print("""
【绝对位置编码】

思想：每个位置有一个唯一的编码，直接加到嵌入向量上。

公式：新向量 = 嵌入向量 + 位置编码

示例：
  位置 0: [0.1, 0.2, 0.3, 0.4]
  位置 1: [0.5, 0.6, 0.7, 0.8]
  位置 2: [0.9, 1.0, 1.1, 1.2]
  ...

【缺点】
1. 训练时见过最大长度（如 2048）
2. 推理时超过这个长度就懵了（没学过位置 2049）
3. 不利用 Transformer 的平移不变性
""")

# 演示绝对位置编码
seq_len = 4
dim = 8

# 模拟嵌入向量
embeddings = torch.randn(seq_len, dim)
print(f"\n【嵌入向量】形状: {embeddings.shape}")

# 绝对位置编码（正弦余弦版，Transformer 原版用）
def get_sinusoidal_position(seq_len, dim):
    """正弦余弦位置编码"""
    position = torch.arange(seq_len).unsqueeze(1).float()  # [seq_len, 1]
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
    
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
    return pe

abs_pos_encoding = get_sinusoidal_position(seq_len, dim)
print(f"\n【绝对位置编码】形状: {abs_pos_encoding.shape}")
print(f"位置 0: {abs_pos_encoding[0][:4].numpy()}...")
print(f"位置 1: {abs_pos_encoding[1][:4].numpy()}...")
print(f"位置 2: {abs_pos_encoding[2][:4].numpy()}...")

# 相加
embeddings_with_pos = embeddings + abs_pos_encoding
print(f"\n【添加位置编码后】形状: {embeddings_with_pos.shape}")

# ============================================
# 第三部分：RoPE 核心思想 - 旋转
# ============================================

print("\n" + "=" * 70)
print("第三部分：RoPE 核心思想 - 旋转向量")
print("=" * 70)

print("""
【RoPE (Rotary Position Embedding)】

核心思想：用"旋转"给向量添加位置信息！

想象每个词的向量是一个箭头：
  位置 0: → (0度)
  位置 1: ↗ (45度)
  位置 2: ↑ (90度)
  位置 3: ↖ (135度)

同一个词，在不同位置 → 旋转不同角度 → 不同向量！

【为什么旋转？】

1. 旋转是线性变换，可以用矩阵乘法实现
2. 保持向量的长度（模长）不变
3. 方便计算相对位置（旋转角度相减）
""")

# 可视化旋转
fig, ax = plt.subplots(figsize=(10, 10))

# 原始向量
vec = np.array([1, 0])
angles = [0, 30, 60, 90, 120, 150, 180]
colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))

for i, angle_deg in enumerate(angles):
    angle_rad = np.radians(angle_deg)
    # 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    rotated_vec = rotation_matrix @ vec
    
    ax.arrow(0, 0, rotated_vec[0], rotated_vec[1], 
             head_width=0.05, head_length=0.05, 
             fc=colors[i], ec=colors[i], linewidth=2,
             label=f'位置 {i}: {angle_deg}°')
    ax.text(rotated_vec[0]*1.1, rotated_vec[1]*1.1, 
            f'{angle_deg}°', fontsize=10, color=colors[i])

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('RoPE: 不同位置旋转向量', fontsize=14)
plt.tight_layout()
plt.savefig('rope_rotation.png', dpi=150, bbox_inches='tight')
print("\n【可视化】已保存为 'rope_rotation.png'")

# ============================================
# 第四部分：RoPE 的数学实现
# ============================================

print("\n" + "=" * 70)
print("第四部分：RoPE 的数学实现")
print("=" * 70)

print("""
【RoPE 公式】

对于向量 x = [x_0, x_1, x_2, x_3, ..., x_{d-1}]

步骤 1: 把向量分成 d/2 对
  [(x_0, x_1), (x_2, x_3), ..., (x_{d-2}, x_{d-1})]

步骤 2: 每对看作复数（二维向量）
  (x_0, x_1) → x_0 + i*x_1

步骤 3: 乘以旋转因子 e^{i*m*θ_j}
  m: 位置
  θ_j: 第 j 对的频率

步骤 4: 转回实数
  新向量 = [x_0', x_1', x_2', x_3', ...]

【旋转角度】

θ_j = 1 / (base^{2j/d})

base: 旋转基数（Llama3 用 500000）
j: 维度对索引
""")

# 手动实现 RoPE
def apply_rope(x, positions, base=10000.0):
    """
    手动实现 RoPE
    x: [seq_len, dim] 输入向量
    positions: [seq_len] 位置索引
    base: 旋转基数
    """
    seq_len, dim = x.shape
    
    # 步骤 1: 把向量分成对
    # x: [seq_len, dim] → [seq_len, dim//2, 2]
    x_pairs = x.float().reshape(seq_len, -1, 2)
    
    # 步骤 2: 看作复数
    x_complex = torch.view_as_complex(x_pairs)  # [seq_len, dim//2]
    
    # 步骤 3: 计算旋转角度
    # 频率: θ_j = 1 / (base^{2j/dim})
    dim_half = dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, dim_half).float() / dim_half))
    
    # 每个位置的旋转角度: m * θ_j
    angles = torch.outer(positions.float(), freqs)  # [seq_len, dim//2]
    
    # 旋转因子: e^{i*angle} = cos(angle) + i*sin(angle)
    rotation = torch.polar(torch.ones_like(angles), angles)  # [seq_len, dim//2]
    
    # 步骤 4: 复数乘法（旋转）
    x_rotated = x_complex * rotation  # [seq_len, dim//2]
    
    # 步骤 5: 转回实数
    x_out = torch.view_as_real(x_rotated)  # [seq_len, dim//2, 2]
    x_out = x_out.reshape(seq_len, dim)  # [seq_len, dim]
    
    return x_out, angles, rotation

# 测试 RoPE
seq_len = 4
dim = 8
# ⚠️ 演示用 base=100（小值让旋转角度大、看得清楚）
# 实际模型用的是大值：Transformer 原论文 base=10000，Llama3-8B 用 base=500000
# base 越大，相邻位置的旋转角度差越小，RoPE 变化越平缓（适合长上下文）
base = 100.0

# 创建输入向量（模拟 Query 或 Key）
x = torch.randn(seq_len, dim)
positions = torch.arange(seq_len)

print(f"\n【输入向量】形状: {x.shape}")
print(f"位置索引: {positions.tolist()}")

# 应用 RoPE
x_rope, angles, rotation = apply_rope(x, positions, base)

print(f"\n【RoPE 输出】形状: {x_rope.shape}")
print(f"旋转角度矩阵形状: {angles.shape}")

print("\n【旋转角度】（弧度）:")
for i in range(seq_len):
    print(f"  位置 {i}: {angles[i].numpy()}")

print("\n【旋转前 vs 旋转后】（显示前4维）:")
for i in range(seq_len):
    before = x[i][:4].numpy()
    after = x_rope[i][:4].numpy()
    print(f"  位置 {i}:")
    print(f"    旋转前: {before}")
    print(f"    旋转后: {after}")

# ============================================
# 第五部分：RoPE 的关键特性
# ============================================

print("\n" + "=" * 70)
print("第五部分：RoPE 的关键特性")
print("=" * 70)

print("""
【特性 1: 相对位置编码】

RoPE 编码的是"相对位置"，不是"绝对位置"。

两个词的位置分别是 m 和 n：
- 词 m 旋转了 m*θ
- 词 n 旋转了 n*θ
- 相对旋转: (m-n)*θ

模型学到的是"距离"，不是"绝对位置"！

【特性 2: 与注意力兼容】

RoPE 直接旋转 Q 和 K，不改变注意力计算：
  Scores = (旋转后的 Q) × (旋转后的 K)^T

由于旋转的性质，相对位置信息自然融入！

【特性 3: 外推能力】

训练时见过位置 0-2048
推理时可以处理位置 2049+（因为只关心相对距离）
""")

# 演示相对位置
print("\n【相对位置演示】")
pos_m = 2
pos_n = 5
relative_pos = pos_n - pos_m

print(f"词 m 在位置 {pos_m}")
print(f"词 n 在位置 {pos_n}")
print(f"相对位置: {relative_pos}")
print(f"这意味着：词 n 比词 m 晚出现 {relative_pos} 个位置")

# ============================================
# 第六部分：在注意力中使用 RoPE
# ============================================

print("\n" + "=" * 70)
print("第六部分：在注意力中使用 RoPE")
print("=" * 70)

print("""
【完整流程】

输入: 嵌入向量 X [seq_len, dim]
    ↓
Step 1: 生成 Q, K, V
    Q = X × W_q
    K = X × W_k
    V = X × W_v
    ↓
Step 2: 应用 RoPE 到 Q 和 K
    Q_rotated = RoPE(Q, positions)
    K_rotated = RoPE(K, positions)
    V 不需要旋转！
    ↓
Step 3: 计算注意力
    Scores = Q_rotated × K_rotated^T
    ↓
Step 4: Softmax + 加权求和
    Output = Softmax(Scores) × V

【为什么 V 不需要旋转？】

V 是"内容"，不需要位置信息。
只有 Q 和 K 用于计算"谁关注谁"，需要位置。
""")

# 演示完整流程
class RoPEAttention(nn.Module):
    """带 RoPE 的注意力"""
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        
    def apply_rope(self, x, positions):
        """应用 RoPE"""
        seq_len, dim = x.shape
        
        # 分成对并看作复数
        x_pairs = x.float().reshape(seq_len, -1, 2)
        x_complex = torch.view_as_complex(x_pairs)
        
        # 计算旋转角度
        dim_half = dim // 2
        freqs = 1.0 / (self.base ** (torch.arange(0, dim_half).float() / dim_half))
        angles = torch.outer(positions.float(), freqs)
        
        # 旋转
        rotation = torch.polar(torch.ones_like(angles), angles)
        x_rotated = x_complex * rotation
        
        # 转回实数
        x_out = torch.view_as_real(x_rotated).reshape(seq_len, dim)
        return x_out.to(x.dtype)
    
    def forward(self, x):
        """
        x: [seq_len, dim]
        """
        seq_len, dim = x.shape
        positions = torch.arange(seq_len)
        
        # Step 1: 生成 Q, K, V
        Q = self.W_q(x)  # [seq_len, dim]
        K = self.W_k(x)  # [seq_len, dim]
        V = self.W_v(x)  # [seq_len, dim]
        
        # Step 2: 应用 RoPE 到 Q 和 K
        Q_rotated = self.apply_rope(Q, positions)
        K_rotated = self.apply_rope(K, positions)
        # V 不旋转！
        
        # Step 3: 计算注意力
        scores = torch.matmul(Q_rotated, K_rotated.T) / (dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        
        # Step 4: 加权求和
        output = torch.matmul(weights, V)
        
        return output, weights

# 测试
print("\n【测试带 RoPE 的注意力】")
# 注意：这里 base=100 是演示值，实际 Llama3-8B 用 base=500000
# 小 base 让旋转效应更明显，便于观察位置编码的影响
rope_attn = RoPEAttention(dim=8, base=100.0)

# 创建输入（模拟"我 爱 学 习"）
x = torch.randn(4, 8)
output, weights = rope_attn(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")

print("\n【注意力权重矩阵】:")
words = ["我", "爱", "学", "习"]
print(f"      {'  '.join([f'{w:>8}' for w in words])}")
for i, word in enumerate(words):
    row = weights[i].detach().numpy()
    print(f"{word:>3}: {row}")

# ============================================
# 第七部分：Llama3 的 RoPE 配置
# ============================================

print("\n" + "=" * 70)
print("第七部分：Llama3 的 RoPE 配置")
print("=" * 70)

print("""
【Llama3-8B 的 RoPE 参数】

旋转基数 (rope_theta): 500,000.0
嵌入维度: 4,096
每头维度: 128 (4,096 / 32头)

【为什么基数这么大？】

基数越大，旋转角度越小 → 可以编码更长的序列

base = 500000 时：
  位置 0: 旋转 0 度
  位置 1: 旋转很小的角度
  位置 10000: 旋转较大的角度

这样可以处理长达 128K 的序列！
""")

# 可视化 Llama3 的频率
llama_dim = 128  # 每头维度
llama_base = 500000.0

dim_half = llama_dim // 2
freqs = 1.0 / (llama_base ** (torch.arange(0, dim_half).float() / dim_half))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(dim_half), freqs.numpy(), 'b-', linewidth=2)
ax.set_xlabel('维度对索引', fontsize=12)
ax.set_ylabel('频率', fontsize=12)
ax.set_title(f'Llama3 RoPE 频率分布 (base={llama_base})', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('rope_frequencies.png', dpi=150, bbox_inches='tight')
print("\n【可视化】已保存为 'rope_frequencies.png'")
print("观察：频率随维度指数衰减")

# ============================================
# 第八部分：总结
# ============================================

print("\n" + "=" * 70)
print("第八部分：总结")
print("=" * 70)

print("""
【本阶段重点】

1. 为什么需要位置编码：
   - 让模型知道词在句子中的位置
   - 区分"主语我"和"宾语我"

2. RoPE 核心思想：
   - 用"旋转"给向量添加位置信息
   - 不同位置旋转不同角度
   - 编码的是"相对位置"而非"绝对位置"

3. RoPE 计算步骤：
   Step 1: 向量分成对
   Step 2: 看作复数
   Step 3: 乘以旋转因子 e^{i*m*θ}
   Step 4: 转回实数

4. 在注意力中的使用：
   - 只旋转 Q 和 K
   - V 不旋转
   - 相对位置信息自然融入注意力分数

5. Llama3 配置：
   - 旋转基数: 500,000
   - 支持超长序列（128K）

【形状变化】
输入:     [seq_len, dim]
分成对:   [seq_len, dim//2, 2]
看作复数: [seq_len, dim//2]
旋转后:   [seq_len, dim//2]
转回实数: [seq_len, dim//2, 2]
输出:     [seq_len, dim]

【下一步】
现在模型有了：
✓ 词嵌入（语义）
✓ 注意力（词间关系）
✓ 位置编码（位置信息）

但只有一个注意力头，能力有限。
阶段 5 将学习：多头注意力（Multi-Head Attention）
""")

print("\n" + "=" * 70)
print("阶段 4 完成！")
print("=" * 70)
