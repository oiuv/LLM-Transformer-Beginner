# 09-08: MoE训练技巧

## 显存优化

### 问题

训练MoE模型时，显存需求来自：

| 来源 | 说明 | 占用 |
|------|------|------|
| 模型参数 | 所有专家+门控 | 高 |
| 梯度 | 所有参数 | 高 |
| 优化器状态 | Adam动量等 | 高 |
| 激活值 | 前向传播中间结果 | 中 |

### 解决方案

#### 1. 梯度检查点 (Gradient Checkpointing)

```python
# 用计算换显存
model.gradient_checkpointing_enable()

# 原理：前向时不保存所有激活值，反向时重新计算
```

#### 2. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        outputs = model(input_ids)
        loss = outputs['loss']
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

#### 3. 专家分片 (Expert Sharding)

```python
# DeepSpeed ZeRO
# 将专家分布到不同GPU

# ZeRO Stage 2: 优化器状态分片
# ZeRO Stage 3: 参数+梯度+优化器状态全分片
```

### 显存占用对比

| 配置 | 7B Dense | 8x7B MoE |
|------|----------|-----------|
| FP32 全参数 | ~28GB | ~188GB |
| FP16 全参数 | ~14GB | ~94GB |
| + Gradient Checkpointing | ~10GB | ~70GB |
| + ZeRO-3 | ~2GB | ~12GB |

## 训练稳定性

### 问题

MoE训练容易不稳定的原因：

| 问题 | 原因 | 表现 |
|------|------|------|
| 路由崩溃 | 专家被选中概率差异大 | 只有1-2个专家被选中 |
| 梯度爆炸 | 部分专家接收梯度过大 | NaN |
| 负载不均 | 门控网络偏向某些专家 | 其他专家训练不足 |

### 解决方案

#### 1. 门控温度

```python
def forward(self, x, temperature=1.0):
    logits = self.gate(x)
    # 温度控制分布平滑度
    probs = F.softmax(logits / temperature, dim=-1)
    return probs
```

| 温度 | 效果 |
|------|------|
| < 1 | 更锐利，路由更确定 |
| = 1 | 标准 |
| > 1 | 更平滑，接近均匀 |

#### 2. 噪声路由

```python
# 训练时添加噪声，探索不同专家组合
class NoisyTopKGating(nn.Module):
    def forward(self, x):
        logits = self.gate(x)
        
        if self.training:
            # 添加高斯噪声
            noise = torch.randn_like(logits) * F.softplus(self.noise_std(x))
            logits = logits + noise
        
        return F.softmax(logits, dim=-1)
```

#### 3. 预训练门控

```python
# 先训练一个初始化的门控，再微调
def pretrain_gate(mixture, train_loader, epochs=5):
    """预训练门控网络"""
    gate_params = list(mixture.gate.parameters())
    optimizer = torch.optim.AdamW(gate_params, lr=1e-3)
    
    for epoch in range(epochs):
        for batch in train_loader:
            outputs = mixture(batch)
            loss = aux_load_balancing_loss(gate_probs, indices)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## 负载均衡实践

### 监控工具

```python
class LoadMonitor:
    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.reset()
    
    def reset(self):
        self.counts = torch.zeros(self.num_experts)
        self.total_calls = 0
    
    def update(self, top_k_indices):
        for i in range(self.num_experts):
            self.counts[i] += (top_k_indices == i).sum().item()
        self.total_calls += top_k_indices.numel()
    
    def report(self):
        rates = self.counts / self.total_calls
        ideal = 1.0 / self.num_experts
        
        print(f"\n专家负载分布:")
        print(f"理想值: {ideal:.4f}")
        print(f"变异系数: {rates.std() / rates.mean():.4f}")
        
        for i, rate in enumerate(rates):
            bar = "█" * int(rate * 100)
            print(f"专家{i}: {rate:.4f} {bar}")
        
        # 检查是否有专家过载或空闲
        max_rate = rates.max().item()
        min_rate = rates.min().item()
        
        if max_rate > ideal * 2:
            print("⚠️ 警告: 有专家过载!")
        if min_rate < ideal * 0.1:
            print("⚠️ 警告: 有专家空闲!")
```

### 自动调整

```python
class AdaptiveLoadBalancer:
    def __init__(self, num_experts, target_rate=0.1):
        self.num_experts = num_experts
        self.target_rate = target_rate  # 理想比例
    
    def get_penalty(self, expert_probs):
        """计算惩罚项，鼓励均衡"""
        mean_prob = expert_probs.mean()
        return ((expert_probs - self.target_rate) ** 2).mean()
```

## 分布式训练

### 数据并行

```python
# 基础数据并行
model = nn.DataParallel(model)
```

### 专家并行 (EP)

```python
# 不同专家在不同GPU
class ExpertParallel(nn.Module):
    def __init__(self, num_experts, devices):
        super().__init__()
        self.devices = devices
        
        # 每个专家在不同设备
        for i, expert in enumerate(self.experts):
            expert.to(devices[i % len(devices)])
    
    def forward(self, x, top_k_indices):
        # 将计算发送到对应专家
        outputs = []
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]
            # ... 路由到对应GPU
        return outputs
```

### DeepSpeed配置

```json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    }
}
```

## 常见问题

### Q1: 只有1-2个专家被选中

**原因**: 门控网络过早收敛

**解决方案**:
```python
# 1. 增加辅助损失权重
aux_loss_fn = LoadBalancingLoss(alpha=0.05)

# 2. 使用噪声路由
gating = NoisyTopKGating(noisy_gating=True)

# 3. 预训练门控
pretrain_gate(model, train_loader, epochs=5)
```

### Q2: 训练OOM (显存不足)

**解决方案**:
```python
# 1. 梯度检查点
model.gradient_checkpointing_enable()

# 2. 减小batch size
batch_size = 2

# 3. 使用DeepSpeed ZeRO
# stage=3 可以大幅降低显存

# 4. 量化
model = model.half()
```

### Q3: 训练不收敛

**原因**: 学习率过高

**解决方案**:
```python
# 1. 降低学习率
lr = 5e-5  # MoE通常需要更小的学习率

# 2. 使用预热
scheduler = WarmupLinearSchedule(
    optimizer,
    warmup_steps=1000,
    t_total=100000
)
```

## 训练检查清单

| 检查项 | 建议 |
|--------|------|
| 学习率 | 1e-4 ~ 5e-5 (比Dense模型小) |
| Batch Size | 1-4 (根据显存) |
| 辅助损失权重 | 0.01 ~ 0.05 |
| 梯度裁剪 | 1.0 |
| 混合精度 | 推荐使用 |
| 梯度检查点 | 推荐开启 |
| 负载均衡监控 | 每100步检查 |

## 总结

### 训练MoE的关键

1. **显存优化**: 梯度检查点 + 混合精度 + ZeRO
2. **训练稳定性**: 噪声路由 + 门控预热 + 温度控制
3. **负载均衡**: Auxiliary Loss + 监控 + 动态调整
4. **分布式**: Expert Parallel + DeepSpeed

### 参考配置

| 模型规模 | 学习率 | Batch | 辅助损失α |
|----------|--------|-------|-----------|
| 1B | 2e-4 | 8 | 0.01 |
| 7B | 1e-4 | 4 | 0.02 |
| 46B (8x7B) | 5e-5 | 1 | 0.05 |

---

## 教程总结

通过ch09的学习，你已经：

- [x] 理解为什么需要MoE架构
- [x] 掌握门控网络和专家网络的原理
- [x] 理解Top-K路由机制
- [x] 学会负载均衡的实现
- [x] 了解MoE Transformer结构
- [x] 分析了Mixtral 8x7B实例
- [x] 实现了完整的MoE代码
- [x] 掌握了MoE训练技巧

## 下一步

恭喜完成MoE章节！接下来可以：

1. **动手实践**：参考本文 07_implementation.md 的代码，自行搭建一个 MoE 模型并训练（可参考 ch05-pretraining/train_long.py 的训练流程）
2. **继续学习**: 探索更多大模型优化技术
3. **实际应用**: 将MoE应用到你的项目中
