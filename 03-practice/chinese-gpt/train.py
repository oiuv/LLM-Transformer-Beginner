"""中文 GPT 训练脚本"""

import os
import re
import glob
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tqdm import tqdm
import time


def parse_args():
    """解析命令行参数"""
    description = """中文 GPT 训练脚本

================================================================================
📊 配置说明与硬件要求
================================================================================

【默认配置】针对 16GB 显存（如 RTX 4090）优化，参考 minimind 训练方案：
    -V 8000 -L 8 -H 8 -b 16 -acc 4  →  等效 batch_size=64，显存占用约 8-10GB

【大模型配置】追求更好效果（需要更多训练数据）：
    -V 10000 -L 12 -H 12 -b 8 -acc 8  →  等效 batch_size=64，显存占用约 12GB

【小模型快速实验】显存不足或快速验证：
    -V 6400 -L 6 -H 6 -b 16 -acc 2  →  等效 batch_size=32，显存占用约 4-6GB

================================================================================
📈 默认模型配置（参考 minimind）
================================================================================

配置参数                | 默认值  | 说明
-----------------------|--------|------------------
词表大小 (vocab_size)  | 8,000  | 中文常用字约3500，8000足够覆盖
上下文长度 (context)   | 512    | 可选 768/1024
嵌入维度 (emb_dim)     | 768    | 与 minimind 一致
Transformer层数        | 8      | 与 minimind 一致
注意力头数             | 8      | 与 minimind 一致
批次大小 (batch_size)  | 16     | 梯度累积4步，等效64
梯度累积               | 4      | 等效 batch_size=16×4=64
训练轮数 (epochs)      | 3      | 观察验证损失早停

================================================================================
🚀 使用示例
================================================================================

# 1. 基础训练（默认配置，推荐）
    python train.py -d data/

# 2. 长文本优化（上下文翻倍）
    python train.py -d data/ -C 1024 -b 8 -acc 8

# 3. 小模型快速实验（低显存）
    python train.py -d data/ -V 6400 -L 6 -H 6

# 4. 单文件训练（支持 txt 和 jsonl）
    python train.py -d data/小说.txt

# 5. 多文件训练（自动合并目录下所有 txt 和 jsonl）
    python train.py -d data/

# 6. 自定义训练轮数和学习率
    python train.py -d data/ -e 5 -lr 3e-4

================================================================================
💡 参数调优建议
================================================================================

• 词表大小 (-V): 8000(推荐) / 6400(最小) / 10000(更大覆盖)
• 上下文长度 (-C): 512(快) / 768(平衡) / 1024(效果好但慢)
• 嵌入维度 (-E) 和头数 (-H): 必须整除，如 -E 768 -H 8
• 层数 (-L): 默认8，减小到6可以大幅降低显存
• 批次大小 (-b): 显存够大用 16，不够减到 8 或 4
• 梯度累积 (-acc): 等效 batch_size = -b × -acc，推荐 64
• 学习率 (-lr): 默认 5e-4，如果发散降到 1e-4，如果收敛慢升到 1e-3
• 训练轮数 (-e): 一般 2-5 轮，观察验证损失早停

================================================================================
"""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # 必需参数
    parser.add_argument("-d", "--data_path", type=str, default="dataset/pretrain_t2t_mini.jsonl", help="训练数据：文件路径或目录（支持 txt/jsonl，默认 dataset/pretrain_t2t_mini.jsonl）")

    # 可选参数
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="输出目录 (default: ./output)")
    parser.add_argument("-V", "--vocab_size", type=int, default=6400, help="词表大小 (default: 6400，与 minimind 默认一致)")
    parser.add_argument("-C", "--context_length", type=int, default=340, help="上下文长度 (default: 340，与 minimind 默认一致；中文 1token≈1.5~1.7字符)")
    parser.add_argument("-E", "--emb_dim", type=int, default=768, help="嵌入维度 (default: 768，与 minimind 默认一致)")
    parser.add_argument("-H", "--n_heads", type=int, default=8, help="注意力头数 (default: 8，与 minimind 默认一致)")
    parser.add_argument("-L", "--n_layers", type=int, default=8, help="Transformer层数 (default: 8，与 minimind 默认一致)")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="批次大小 (default: 32，与 minimind 默认一致；显存不够可降到 16/8)")
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-4, help="学习率 (default: 5e-4)")
    parser.add_argument("-e", "--epochs", type=int, default=2, help="训练轮数 (default: 2，与 minimind 默认一致)")
    parser.add_argument("-s", "--val_split", type=float, default=0.05, help="验证集比例 (default: 0.05)")
    parser.add_argument("--accumulation_steps", "-acc", type=int, default=8, help="梯度累积步数 (default: 8，等效batch_size=32×8=256，与 minimind pretrain 默认一致)")

    return parser.parse_args()


def load_and_preprocess_data(data_path):
    """加载并预处理文本数据（支持文件、目录、jsonl）。

    返回:
        samples: 训练样本列表,每个元素是一条独立文本
                 (jsonl 一行 = 一个样本;txt 一个文件 = 一个样本)
        paragraphs: 用于训练 BPE 的段落集合
                    (从 samples 二次切分,按段落切到细粒度供 BPE 学合并)

    关键设计变更:
        旧实现把所有文本 "join 成一整个字符串" + 按 context_length 滑动切块
        → 相邻块互相重叠,没有显式"样本边界",模型分不清哪里是段尾/新一段开头。
        新实现是 per-sample:每条 jsonl/txt 本身就是一条独立样本,
        用 bos + text + eos 包裹(bos=样本开始 / eos=样本结束),
        模型能清晰学到"样本边界",与 minimind 的 PretrainDataset 风格一致。
    """
    print("\n[阶段1] 加载数据...")
    samples = []  # 每个元素是一条独立样本(字符串)

    # 支持目录输入
    if os.path.isdir(data_path):
        # 递归扫描所有子目录下的 .txt 和 .jsonl
        txt_files = glob.glob(os.path.join(data_path, "**", "*.txt"), recursive=True)
        jsonl_files = glob.glob(os.path.join(data_path, "**", "*.jsonl"), recursive=True)
        if not txt_files and not jsonl_files:
            raise FileNotFoundError(f"目录下没有txt或jsonl文件: {data_path}")
        print(f"  数据目录: {data_path}")
        # txt:整个文件 = 一个样本(去前后空白)
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    samples.append(content)
                    print(f"    - {os.path.basename(txt_file)}: {len(content):,}字符 (1个样本)")
        # jsonl:每行 = 一个样本(与 minimind PretrainDataset 一致)
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                file_count = 0
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        if "text" in obj and obj["text"].strip():
                            samples.append(obj["text"].strip())
                            file_count += 1
                    except json.JSONDecodeError:
                        continue
                print(f"    - {os.path.basename(jsonl_file)}: {file_count:,}个样本")
    elif data_path.endswith(".jsonl"):
        # jsonl 格式:每行 {"text": "..."}
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        print(f"  数据文件: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if "text" in obj and obj["text"].strip():
                        samples.append(obj["text"].strip())
                except json.JSONDecodeError:
                    continue
        print(f"  jsonl记录数: {len(samples):,}")
    else:
        # txt 文件:整个文件 = 一个样本
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        print(f"  数据文件: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            samples.append(content)

    # 把所有样本 join 起来(仅用于统计 + 给 BPE 切段落)
    text = "\n\n".join(samples)

    # 统计信息
    total_chars = sum(len(s) for s in samples)
    print(f"  样本数: {len(samples):,}")
    print(f"  总字符数: {total_chars:,}")

    # 切出 paragraphs 给 BPE 训练用(BPE 需要切到比样本更细的粒度学合并)
    # 按连续空行切,过滤太短的段落
    paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 10]
    print(f"  BPE 段落数(供分词器训练): {len(paragraphs):,}")

    return samples, paragraphs


def train_bpe_tokenizer(paragraphs, vocab_size, output_dir):
    """训练中文BPE分词器"""
    print("\n[阶段2] 训练BPE分词器...")

    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")

    # 检查是否已存在
    if os.path.exists(tokenizer_path):
        temp_tokenizer = Tokenizer.from_file(tokenizer_path)
        if temp_tokenizer.get_vocab_size() != vocab_size:
            print(f"  现有分词器词表({temp_tokenizer.get_vocab_size()})与目标({vocab_size})不符，重新训练")
        else:
            print(f"  分词器已存在: {tokenizer_path}")
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, bos_token="<s>", eos_token="</s>", pad_token="<|pad|>")
            print(f"  词表大小: {tokenizer.vocab_size}")
            return tokenizer

    # 创建BPE分词器 - 预处理器选择（实现细节，与 minimind 对齐）
    #
    # BPE 算法本身是"在已切好的小段上做频率合并"，pre_tokenizer 负责"先怎么切"。
    # 对中文有两个常见选择:
    #
    # (1) Whitespace():按"空格/换行"切
    #   - 优点:英文/中英混合时表现自然,代码直观
    #   - 致命问题:中文没有空格 → 整段汉字变成一个"词"丢给 BPE
    #     → BPE 几乎只能学到整句级别的合并,词汇碎片化严重
    #     → 同样的 context_length 下,中文 token 序列更长、信息密度更低
    #
    # (2) ByteLevel():按 UTF-8 字节切(GPT-2 同款)
    #   - 每个中文字符 → 2~3 字节(UTF-8 中文字符固定 3 字节) → 3 个"字节级 token"
    #   - BPE 在字节上做频率合并 → 学到"常用字/常用偏旁"作为 token
    #   - 中文压缩率约 1.5~1.7 char/token(README 注释),是 Whitespace 的 ~3 倍
    #   - 跨语言一致:同一套 token 表能处理中英混排,不会出现"中文字符整个 OOV"
    #   - 副作用:token 数看着多,但 embedding 层(vocab_size=6400)照常工作
    #
    # minimind 默认就用 ByteLevel;chinese-gpt 之前用 Whitespace 是为了代码简洁。
    # 切到 ByteLevel 是实现选型差异,不是新概念:BPE 训练/合并/解码流程完全一致。
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # 配置训练器
    # min_frequency=1: 即使生僻字只出现 1 次也保留进词表,大幅减少 <unk> 输出
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<|pad|>", "<|unk|>", "<s>", "</s>"], min_frequency=1)

    # 训练
    print(f"  训练中... (词表大小: {vocab_size})")
    start_time = time.time()
    tokenizer.train_from_iterator(paragraphs, trainer)
    print(f"  训练完成! 耗时: {time.time() - start_time:.1f}秒")

    # 保存
    tokenizer.save(tokenizer_path)
    print(f"  已保存: {tokenizer_path}")
    print(f"  实际词表大小: {tokenizer.get_vocab_size()}")

    # 转换为transformers格式
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, bos_token="<s>", eos_token="</s>", pad_token="<|pad|>")

    # 显式设置 pad_token_id，避免后续使用时报错
    # 兼容 transformers 5.x:优先用 convert_tokens_to_ids,回退到 token_to_id
    if hasattr(hf_tokenizer, "convert_tokens_to_ids"):
        hf_tokenizer.pad_token_id = hf_tokenizer.convert_tokens_to_ids("<|pad|>")
    else:
        hf_tokenizer.pad_token_id = hf_tokenizer.token_to_id("<|pad|>")

    return hf_tokenizer


class NovelDataset(Dataset):
    """文本数据集(通用,支持小说、对话、百科等)。

    设计:per-sample + bos+text+eos 边界

    旧实现把全文本拼成一个长字符串,按 context_length 滑动切块:
        相邻块互相重叠,没有显式的"样本边界"标记。
        模型分不清"这是一段的结尾"还是"这是另一段的开头",
        位置编码学到的是"在第 N 个 token",而不是"在第 N 条样本内"。

    新实现每条样本独立处理:
        1. 取第 idx 条样本(一段 jsonl text 或一个 txt 文件内容)
        2. tokenizer.encode() 一次性编码,max_length = context_length - 2(留位置给 bos/eos)
        3. 用 bos_id + tokens + eos_id 包裹,pad 不足的位置
        4. labels[i] = tokens[i+1] 风格的 LM 目标;labels[pad] = -100(不计入 loss)

    这样模型能清晰学到:
        - bos 之后是新样本开头
        - eos 处必停(生成时撞到 eos 就结束)
        - 一个 context_length 窗口 = 一条完整样本,位置编码更稳定

    注:删掉了原来的 .npy cache 逻辑 —— per-sample 不需要预编码整个语料,
       每次 __getitem__ 重新 tokenize 即可(minimind 同款做法)。
       首次训练时会比旧版稍慢(逐条 tokenize),但样本数也少得多,差距不大。
    """

    def __init__(self, samples, tokenizer, context_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id

        print("\n[阶段3] 创建数据集(per-sample 模式)...")
        print(f"  样本数: {len(self.samples):,}")
        print(f"  上下文长度: {self.context_length}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. 取一条样本(整段字符串)
        text = self.samples[idx]

        # 2. 编码:在 encode 时就截断到 context_length - 2,留位置给 bos/eos
        #    add_special_tokens=False,手动加 bos/eos
        #    truncation=True + max_length 是 tokenizer 内部的截断,与 PretrainDataset 等价
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.context_length - 2,
            truncation=True,
        )

        # 3. 用 bos + text + eos 包裹,pad 到 context_length
        #    长度上限就是 context_length(bos + (ctx-2) + eos)
        full = [self.bos_id] + token_ids + [self.eos_id]
        if len(full) < self.context_length:
            full = full + [self.pad_id] * (self.context_length - len(full))
        else:
            full = full[: self.context_length]

        input_ids = torch.tensor(full, dtype=torch.long)

        # 4. LM 目标:下一 token 预测
        #    labels[i] = input_ids[i+1],最后一位无目标
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100

        # 忽略 padding 的损失
        labels[labels == self.pad_id] = -100

        return {"input_ids": input_ids, "labels": labels}


def create_model(vocab_size, config, output_dir):
    """创建GPT-2模型"""
    print("\n[阶段4] 创建模型...")

    model_path = os.path.join(output_dir, "model")

    # 检查是否已存在训练好的模型
    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        print(f"  加载已训练模型: {model_path}")
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model = model.to(config["device"])
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
        return model

    # 创建新模型
    # 注意：GPT2Config 使用 n_ctx 表示上下文长度，而非 n_positions
    gpt_config = GPT2Config(
        vocab_size=vocab_size,
        n_ctx=config["context_length"],
        n_embd=config["emb_dim"],
        n_layer=config["n_layers"],
        n_head=config["n_heads"],
    )

    model = GPT2LMHeadModel(gpt_config)
    model = model.to(config["device"])

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    return model


def build_optimizer(model, config):
    """构造 AdamW 优化器,对齐 minimind 训练侧配方。

    与 minimind 训练配方对齐的 3 项关键调整(架构不动前提下):
      1. betas=(0.9, 0.95):LLM 训练的事实标准,PyTorch 默认 (0.9, 0.999) 在长 tail
         任务里容易出现 lr 调度与 loss 曲线的耦合震荡。
      2. weight_decay=0.01:对矩阵参数(Linear/Conv 等 nn > 1 维权重)做 L2 衰减,
         这是 LLM 训练的另一条经验值。
      3. 参数分组:矩阵权重走 weight_decay,1D 参数(bias / LayerNorm.weight 等
         标量形状的张量)走 weight_decay=0。这是标准做法——1D 参数在 weight_decay
         下容易被"过度惩罚",反而拖慢收敛。

    ⚠️ checkpoint 兼容性:
       weight_decay / betas 改了之后,旧的 checkpoint.pt 里
       optimizer_state_dict(param_groups 结构不同)load_state_dict 会报错。
       如需恢复旧训练,删 output/model/checkpoint.pt,模型权重也一并从这次起从头训。
    """
    # 1. 参数分组:2D+ 矩阵参数走 weight_decay,1D (bias / norm.weight 等) 走 0
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 仅对矩阵参数应用 weight decay:bias / norm weight 是 1D,直接跳过
        # (nDim > 1 比 param.dim() > 1 更宽松,匹配 nn.Linear.weight / nn.Embedding.weight)
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    weight_decay = float(config.get("weight_decay", 0.01))
    lr = config["learning_rate"]

    optim_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # 2. AdamW:betas=(0.9, 0.95),eps 显式固定,避免依赖 PyTorch 默认
    optimizer = AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"  ✓ AdamW:lr={lr:.1e}, betas=(0.9, 0.95), eps=1e-8")
    print(f"  weight_decay={weight_decay} 仅作用于矩阵权重  | 1D 参数(bias / norm.weight)不衰减")
    print(f"  decay 参数: {n_decay:,}  |  no-decay 参数: {n_no_decay:,}")

    return optimizer


def train_model(
    model, train_loader, val_loader, tokenizer, config, output_dir, start_epoch=0, best_val_loss=float("inf"), no_improve_epochs=0, optimizer=None, scheduler=None, loaded_scheduler_state=None
):
    """训练模型"""
    print("\n[阶段5] 训练模型...")

    # bf16 混合精度:仅在 CUDA 上启用(RTX 4090 原生支持 bf16,显存减半、速度近翻倍)
    # 权重保持 fp32,只是前向计算时用 bf16 → checkpoint 兼容性好
    use_bf16 = config["device"] == "cuda" and torch.cuda.is_bf16_supported()
    if use_bf16:
        print("  ✓ 启用 bf16 混合精度(权重 fp32 + 计算 bf16)")

    if optimizer is None:
        optimizer = build_optimizer(model, config)
        # Cosine schedule + Warmup:前 3% 步线性升 lr,之后余弦衰减到 0
        if scheduler is None:
            # 注意：梯度累积时，实际更新次数 = 数据步数 / accumulation_steps
            accumulation_steps = config.get("accumulation_steps", 1)
            total_steps = config["epochs"] * len(train_loader) // accumulation_steps
            warmup_steps = max(1, int(0.03 * total_steps))
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            print(f"  ✓ 启用 Cosine schedule (warmup={warmup_steps} 步, total={total_steps} 步)")

    # 断点续训时,把已存的 scheduler state 加载到新 scheduler
    if scheduler is not None and loaded_scheduler_state is not None:
        scheduler.load_state_dict(loaded_scheduler_state)
        print(f"  ✓ 恢复 scheduler 状态,当前 lr = {scheduler.get_last_lr()[0]:.2e}")

    accumulation_steps = config.get("accumulation_steps", 1)
    if accumulation_steps > 1:
        print(f"  ✓ 启用梯度累积（每 {accumulation_steps} 步更新一次，等效 batch_size={config['batch_size'] * accumulation_steps}）")

    patience = 3
    model_path = os.path.join(output_dir, "model")
    os.makedirs(model_path, exist_ok=True)

    # 日志打印间隔：每100步打印一次loss
    # 作用：大epoch可能有几千步，每个epoch才打印一次看不到中间变化趋势
    # 及时发现 loss spike 或发散，方便调参
    log_interval = 100

    for epoch in range(start_epoch, config["epochs"]):
        current_epoch = epoch + 1
        print(f"\nEpoch {current_epoch}/{config['epochs']}")
        print("-" * 40)

        # 训练
        model.train()
        train_loss = 0
        train_steps = 0

        for batch in tqdm(train_loader, desc="训练中"):
            input_ids = batch["input_ids"].to(config["device"])
            labels = batch["labels"].to(config["device"])

            # bf16 autocast:仅前向计算降精度,权重/loss/梯度保持 fp32
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

            # 梯度累积：把 loss 按累积步数缩小，保证梯度量级一致
            # 例如 accumulation_steps=4 时，4 步的梯度等效于 4 倍 batch_size
            loss = loss / accumulation_steps
            loss.backward()

            # 每 accumulation_steps 步才更新一次参数
            if (train_steps + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()  # 每步更新 lr
                # set_to_none=True: 直接把梯度设为 None，而不是填零
                # 比默认行为更高效，减少一次内存写操作，省显存省时间
                optimizer.zero_grad(set_to_none=True)

            # 记录真实的 loss（还原缩放）
            train_loss += loss.item() * accumulation_steps
            train_steps += 1

            # 每 log_interval 步打印一次中间loss，方便观察训练趋势
            if train_steps % log_interval == 0:
                current_loss = loss.item() * accumulation_steps
                current_lr = optimizer.param_groups[-1]['lr']
                print(f"    Step {train_steps}, Loss: {current_loss:.4f}, LR: {current_lr:.2e}")

        # 处理末尾不足 accumulation_steps 的残余梯度
        if train_steps % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        avg_train_loss = train_loss / train_steps

        # 验证
        model.eval()
        val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(config["device"])
                labels = batch["labels"].to(config["device"])

                # 验证也用 bf16(保持一致,数值差异可忽略)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    outputs = model(input_ids=input_ids, labels=labels)
                val_loss += outputs.loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps

        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证损失: {avg_val_loss:.4f}")
        if scheduler is not None:
            print(f"  当前 lr: {scheduler.get_last_lr()[0]:.2e}")

        # 保存训练日志
        with open(os.path.join(output_dir, "training.log"), "a") as f:
            f.write(f"{current_epoch} {avg_train_loss:.4f} {avg_val_loss:.4f}\n")

        # 保存最佳模型 + 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(model_path)
            print(f"  ✓ 保存最佳模型 (Val Loss: {avg_val_loss:.4f})")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"  验证损失未下降 ({no_improve_epochs}/{patience})")
            if no_improve_epochs >= patience:
                print(f"  验证损失连续{patience}轮未下降，提前终止训练")
                break

        # 保存checkpoint（支持断点续训）
        checkpoint = {
            "epoch": current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "no_improve_epochs": no_improve_epochs,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(checkpoint, os.path.join(model_path, "checkpoint.pt"))

        # 显示显存使用(driver 级口径,与 nvidia-smi 一致)
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            used_gb = (total - free) / 1024**3
            total_gb = total / 1024**3
            allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  显存: {used_gb:.2f}GB / {total_gb:.2f}GB (PyTorch 持有: {allocated_gb:.2f}GB)")

    # 删除checkpoint文件（训练完成）
    checkpoint_path = os.path.join(model_path, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("  checkpoint已清理")

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型已保存: {model_path}")

    # 生成样例文本
    print("\n" + "=" * 60)
    print("生成样例文本")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        sample_prompt = "第一章"
        sample_ids = tokenizer.encode(sample_prompt, return_tensors="pt").to(config["device"])
        sample_output = model.generate(
            sample_ids,
            max_length=sample_ids.shape[1] + 100,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        sample_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
        print(f"提示: {sample_prompt}")
        print(f"生成: {sample_text}")
    print("=" * 60)

    return model, best_val_loss


def main():
    # 解析参数
    args = parse_args()

    # 设置随机种子，保证实验可复现
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 配置
    config = {
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "val_split": args.val_split,
        "accumulation_steps": args.accumulation_steps,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # 打印配置
    print("=" * 60)
    print("中文 GPT 训练")
    print("=" * 60)
    print(f"设备: {config['device']}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("-" * 60)
    print(f"数据文件: {config['data_path']}")
    print(f"输出目录: {config['output_dir']}")
    print(f"词表大小: {config['vocab_size']}")
    print(f"上下文长度: {config['context_length']}")
    print(f"嵌入维度: {config['emb_dim']}")
    print(f"注意力头数: {config['n_heads']}")
    print(f"Transformer层数: {config['n_layers']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"学习率: {config['learning_rate']}")
    print(f"训练轮数: {config['epochs']}")
    print("=" * 60)

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据 (samples 给训练用, paragraphs 给 BPE 用)
    samples, paragraphs = load_and_preprocess_data(config["data_path"])

    # 2. 训练分词器
    tokenizer = train_bpe_tokenizer(paragraphs, config["vocab_size"], output_dir)

    # 3. 创建数据集 (per-sample 模式,无 .npy 缓存)
    full_dataset = NovelDataset(samples, tokenizer, config["context_length"])

    # 划分训练/验证集
    train_size = int((1 - config["val_split"]) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    print(f"  训练集: {len(train_dataset):,}")
    print(f"  验证集: {len(val_dataset):,}")

    # 创建DataLoader
    # num_workers: 数据加载的并行进程数
    # - 0: 主进程串行加载（GPU要等CPU，训练速度慢30-50%）
    # - 4: 4个子进程并行加载，GPU不用等（推荐值，可根据CPU核心数调整）
    # 注意：大数据集时 num_workers 会占用大量内存（每个子进程复制数据集引用）
    # 当前数据集 2.2亿 token，num_workers=2 占约 28GB，平衡速度和内存
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # 4. 创建模型
    model = create_model(tokenizer.vocab_size, config, output_dir)

    # torch.compile: PyTorch 2.x 的编译优化，一行代码提速 10-30%
    # 原理：把 Python 字节码编译成优化的计算图，减少 Python 开销
    # 首次编译会慢几十秒，之后每个 epoch 都会快不少
    # 注意：需要 PyTorch >= 2.0 + Triton 库（Windows 上 Triton 支持有限，暂不启用）
    # if config["device"] == "cuda":
    #     model = torch.compile(model)
    #     print("  ✓ 启用 torch.compile 加速")

    # 检查checkpoint，支持断点续训
    model_path = os.path.join(output_dir, "model")
    checkpoint_path = os.path.join(model_path, "checkpoint.pt")
    start_epoch = 0
    best_val_loss = float("inf")
    no_improve_epochs = 0
    optimizer = None
    scheduler = None
    loaded_scheduler_state = None  # 暂存 checkpoint 里的 scheduler state

    if os.path.exists(checkpoint_path):
        print("\n  发现checkpoint，正在加载...")
        checkpoint = torch.load(checkpoint_path, map_location=config["device"])
        model.load_state_dict(checkpoint["model_state_dict"])
        # 先用 build_optimizer 构造(同训练配方),再加载 state
        # 若旧版 checkpoint 形状不匹配,下面这行会抛 param_groups mismatch,提示删 checkpoint.pt
        optimizer = build_optimizer(model, config)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        no_improve_epochs = checkpoint["no_improve_epochs"]
        # 旧 checkpoint 可能没有 scheduler_state_dict(向前兼容)
        if "scheduler_state_dict" in checkpoint:
            loaded_scheduler_state = checkpoint["scheduler_state_dict"]
        print(f"  从第 {start_epoch + 1} 轮继续训练")
        print(f"  最佳验证损失: {best_val_loss:.4f}")
        # 注意：这里不立即删除 checkpoint
        # 原因：若加载后、第一个 epoch 完成前训练崩溃（OOM/断电/Ctrl+C），
        #       checkpoint 已删则无法再次恢复。让每个 epoch 末尾的保存逻辑
        #       自然覆盖它，训练正常结束时统一清理（见下方 train_model 末尾）。
        # ── optimizer 兼容性提示 ──
        # build_optimizer 的 weight_decay / betas 与旧版 AdamW(model.parameters())
        # 不同,param_groups 形状也变了。本代码会自动让新 optimizer 加载新 state;
        # 如果 load_state_dict 报 param_groups mismatch 多半是旧版 checkpoint,
        # 删 output/model/checkpoint.pt 重训即可(模型权重也建议从头训以匹配新配方)

    # 5. 训练(传入 loaded_scheduler_state 以便 train_model 恢复 lr 调度)
    model, best_val_loss = train_model(model, train_loader, val_loader, tokenizer, config, output_dir, start_epoch, best_val_loss, no_improve_epochs, optimizer, scheduler, loaded_scheduler_state)

    # 保存tokenizer配置
    tokenizer.save_pretrained(os.path.join(output_dir, "model"))

    # 保存训练配置
    import json
    config_to_save = {k: v for k, v in config.items() if k != "device"}
    config_to_save["best_val_loss"] = best_val_loss
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_to_save, f, indent=2, ensure_ascii=False)
    print(f"配置已保存: {os.path.join(output_dir, 'config.json')}")

    print("\n下一步: 运行 python generate.py 生成文本")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练已手动中断。")
        print("如需恢复训练，请使用相同的命令重新运行（支持断点续训）。")
        import sys
        sys.exit(0)
