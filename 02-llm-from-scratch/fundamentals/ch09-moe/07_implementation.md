# 09-07: MoE代码实现

## 完整MoE实现

### 项目结构

> 📌 **说明**：以下是一个推荐的 MoE 项目目录结构，可用于自己动手实现。本教程提供下方完整的代码实现，但未单独建立 `moe-gpt/` 项目目录——学习者可参考 `projects/instruction-finetuning/` 的组织方式自行搭建。

```
projects/
└── moe-gpt/                 # 推荐结构（需自行创建）
    ├── model.py             # MoE 模型定义（本文已包含）
    ├── train.py             # 训练脚本（可参考 ch05 的 train_long.py）
    └── generate.py          # 生成脚本（可参考 ch05 的 generate.py）
```

## 1. MoE Layer实现

```python
# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class TopKGating(nn.Module):
    """Top-K路由门控网络"""
    
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int, 
                 noisy_gating: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        
        # 门控权重
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # 噪声网络（可选，用于训练稳定性）
        if noisy_gating:
            self.noise_std = nn.Linear(hidden_dim, num_experts)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            top_k_indices: [batch, seq_len, top_k] 选中的专家索引
            top_k_weights: [batch, seq_len, top_k] 归一化权重
        """
        # 门控分数
        logits = self.gate(x)  # [B, L, E]
        
        # 添加噪声（训练时）
        if self.training and self.noisy_gating:
            noise = torch.randn_like(logits)
            noise_std = F.softplus(self.noise_std(x))  # 确保为正
            logits = logits + noise * noise_std
        
        # Softmax转概率
        probs = F.softmax(logits, dim=-1)
        
        # Top-K选择
        top_k_probs, top_k_indices = torch.topk(probs, k=self.top_k, dim=-1)
        
        # 归一化权重
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-6)
        
        return top_k_indices, top_k_weights


class Expert(nn.Module):
    """单个专家网络（标准FFN）"""
    
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class MoELayer(nn.Module):
    """MoE层：多个专家 + Top-K路由"""
    
    def __init__(self, hidden_dim: int, ffn_dim: int, 
                 num_experts: int, top_k: int, noisy_gating: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = TopKGating(hidden_dim, num_experts, top_k, noisy_gating)
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim) 
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            output: [batch, seq_len, hidden_dim]
            aux_loss: 标量，负载均衡辅助损失（用于鼓励专家被均匀使用）
        """
        batch_size, seq_len, hidden_dim = x.shape
        original_shape = x.shape

        # 展平用于批量计算
        x_flat = x.view(-1, hidden_dim)  # [B*L, H]

        # 获取路由
        top_k_indices, top_k_weights = self.gate(x)  # [B, L, K], [B, L, K]
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)  # [B*L, K]
        top_k_weights_flat = top_k_weights.view(-1, self.top_k)  # [B*L, K]

        # 初始化输出
        output_flat = torch.zeros_like(x_flat)

        # 对每个token选择top_k专家加权求和
        for k_idx in range(self.top_k):
            expert_indices = top_k_indices_flat[:, k_idx]  # [B*L]
            expert_weights = top_k_weights_flat[:, k_idx]  # [B*L]

            # 对每个专家收集输出
            for expert_idx, expert in enumerate(self.experts):
                # 找出选择该专家的token
                mask = (expert_indices == expert_idx)  # [B*L]

                if mask.any():
                    # 计算该专家对这个token的输出
                    expert_output = expert(x_flat[mask])  # [N, H]
                    # 加权累加：expert_weights[mask] 是 [N]，需要广播到 [N, H]
                    # 用 unsqueeze(-1) 显式扩展，避免依赖隐式广播
                    output_flat[mask] += expert_weights[mask].unsqueeze(-1) * expert_output  # [N, H]

        # 计算负载均衡辅助损失
        # 目标：让每个专家被选中的频率尽量接近 1/num_experts
        # 若不均衡（少数专家被大量token选中），训练会退化成只用这几个专家
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for expert_idx in range(self.num_experts):
            expert_counts[expert_idx] = (top_k_indices == expert_idx).float().sum()
        expert_freq = expert_counts / expert_counts.sum()  # 归一化为概率分布
        ideal_freq = 1.0 / self.num_experts
        aux_loss = self.num_experts * ((expert_freq - ideal_freq) ** 2).sum()

        return output_flat.view(*original_shape), aux_loss


class LoadBalancingLoss(nn.Module):
    """负载均衡辅助损失"""
    
    def __init__(self, num_experts: int, top_k: int, alpha: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha
    
    def forward(self, gate_probs: torch.Tensor, 
                top_k_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate_probs: [batch, seq_len, num_experts] 门控概率
            top_k_indices: [batch, seq_len, top_k] 选中的专家索引
        Returns:
            aux_loss: 辅助损失
        """
        # 计算每个专家被选中的频率
        # 方法：统计每个expert被选中的token数量 / 总token数量
        
        expert_counts = torch.zeros(self.num_experts, device=gate_probs.device)
        
        for expert_idx in range(self.num_experts):
            count = (top_k_indices == expert_idx).float().sum()
            expert_counts[expert_idx] = count
        
        # 归一化为概率
        expert_freq = expert_counts / expert_counts.sum()
        
        # 理想频率
        ideal_freq = 1.0 / self.num_experts
        
        # 辅助损失：鼓励均匀分布（使用均方误差）
        # 注意：错误的公式是 num_experts * sum(p_i * 1/num_experts) = 1 总是返回1
        # 正确的公式：num_experts * sum((p_i - 1/num_experts)^2)
        aux_loss = self.num_experts * ((expert_freq - ideal_freq) ** 2).sum()
        
        return self.alpha * aux_loss
```

## 2. MoE Transformer Block

```python
class MultiHeadAttention(nn.Module):
    """标准多头自注意力"""
    
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, H = x.shape
        
        # QKV
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        # Output
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        return self.proj(out)


class MoETransformerBlock(nn.Module):
    """MoE版本的Transformer Block
    
    标准Mixtral架构（Pre-LN结构）：
    1. norm1 - Self-Attention前的Pre-LN
    2. norm2 - MoE Layer前的Pre-LN
    3. 共享expert和routed experts的输出相加后通过残差连接
    
    注意：原始实现错误地添加了norm3用于共享FFN，这是多余的。
    共享FFN应该和routed experts一样接收norm2的输出。
    """
    
    def __init__(self, hidden_dim: int, ffn_dim: int, 
                 num_heads: int, num_experts: int, top_k: int):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.moe = MoELayer(hidden_dim, ffn_dim, num_experts, top_k)
        
        # 共享expert（始终参与）
        self.shared_expert = Expert(hidden_dim, ffn_dim)
    
    def forward(self, x: torch.Tensor):
        """
        Returns:
            output: [batch, seq_len, hidden_dim]
            aux_loss: 本层 MoE 的负载均衡损失（标量）
        """
        # Self-Attention + 残差
        x = x + self.attn(self.norm1(x))

        # MoE Layer（包含routed experts + 共享expert）
        # norm2的输出同时送给routed experts和共享expert
        moe_input = self.norm2(x)
        routed_output, aux_loss = self.moe(moe_input)
        shared_output = self.shared_expert(moe_input)

        # 残差连接
        x = x + routed_output + shared_output

        return x, aux_loss
```

## 3. 完整MoE模型

```python
class MoEGPT(nn.Module):
    """完整MoE GPT模型"""
    
    def __init__(self, config):
        super().__init__()
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        
        self.blocks = nn.ModuleList([
            MoETransformerBlock(
                hidden_dim=config.hidden_dim,
                ffn_dim=config.ffn_dim,
                num_heads=config.num_heads,
                num_experts=config.num_experts,
                top_k=config.top_k
            )
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # 权重绑定
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        # Embedding
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(torch.arange(x.size(1), device=x.device))

        # Transformer Blocks（累计所有层的 aux_loss）
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # 主损失：next-token prediction 的交叉熵
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            # 总损失 = LM 损失 + 负载均衡损失
            # alpha 控制辅助损失的权重（通常 0.01）
            loss = lm_loss + 0.01 * total_aux_loss

        return {'loss': loss, 'logits': logits, 'aux_loss': total_aux_loss}


@dataclass
class MoEConfig:
    vocab_size: int = 50000
    hidden_dim: int = 768
    num_heads: int = 12
    ffn_dim: int = 3072
    num_layers: int = 12
    num_experts: int = 8
    top_k: int = 2
    max_seq_len: int = 512
```

## 4. 训练循环

```python
def train_moe(config, train_loader, val_loader, device):
    # 创建模型
    model = MoEGPT(config).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 注意：负载均衡损失（aux_loss）已在 MoEGPT.forward 内部计算并加到 loss 上
    # 这里不需要再单独实例化 LoadBalancingLoss

    best_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # 前向传播：outputs['loss'] 已包含 lm_loss + 0.01 * aux_loss
            outputs = model(input_ids, labels)
            loss = outputs['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, labels)
                val_loss += outputs['loss'].item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'moe_model.pt')
    
    return model
```

## 显存优化技巧

```python
# 1. 梯度检查点（Gradient Checkpointing）
# 用计算换显存
model.gradient_checkpointing_enable()

# 2. 专家分片（需要分布式）
# 每个专家放在不同GPU

# 3. 量化
model = model.half()  # FP16
model = model.quantize('int8')  # INT8
```

## 总结

### 完整MoE实现要点

1. **TopKGating**: 门控网络 + Top-K选择
2. **Expert**: 标准FFN结构
3. **MoELayer**: 多个专家 + 稀疏路由
4. **LoadBalancingLoss**: 负载均衡辅助损失
5. **MoETransformerBlock**: Self-Attention + MoE

---

## 下一步

学习 [09-08: MoE训练技巧](./08_training_tips.md)
