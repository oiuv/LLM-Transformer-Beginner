"""中文小说 GPT 训练脚本"""

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
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
import time


def parse_args():
    """解析命令行参数"""
    description = """中文小说 GPT 训练脚本

================================================================================
📊 配置说明与硬件要求
================================================================================

【默认配置】针对 16GB 显存（如 RTX 4090）优化：
    -C 512 -b 8  →  显存占用约 10-12GB，训练速度最快

【平衡配置】上下文和速度的平衡：
    -C 768 -b 6  →  显存占用约 12GB，比512长50%上下文，batch适中

【推荐配置】追求更好的长文本生成效果：
    -C 1024 -b 4  →  显存占用约 10-12GB，上下文翻倍，适合小说生成

【小模型快速实验】显存不足或快速验证：
    -C 256 -L 6  →  显存占用约 4-6GB，只改上下文和层数，其他默认

================================================================================
📈 模型规模对比（与 GPT-2 系列）
================================================================================

配置参数                | 你的模型(默认) | GPT-2 Small | GPT-2 Medium
-----------------------|---------------|-------------|--------------
词表大小 (vocab_size)  | 50,000        | 50,257      | 50,257
上下文长度 (context)   | 512 (可1024)  | 1024        | 1024
嵌入维度 (emb_dim)     | 768           | 768         | 1024
Transformer层数        | 12            | 12          | 24
注意力头数             | 12            | 12          | 16
参数量                 | ~124M         | ~117M       | ~345M
显存需求 (参考)        | ~10-12GB      | ~10-12GB    | ~30GB+

默认配置 ≈ GPT-2 Small 级别，但针对中文优化（BPE分词器）

================================================================================
🚀 使用示例
================================================================================

# 1. 基础训练（默认配置，适合 16GB 显存）
    python train.py -d data/

# 2. 长文本优化（推荐，上下文翻倍）
    python train.py -d data/ -C 1024 -b 4

# 3. 小模型快速实验（低显存）
    python train.py -d data/ -C 256 -L 6

# 4. 单文件训练
    python train.py -d data/小说.txt

# 5. 多文件训练（自动合并目录下所有txt）
    python train.py -d data/

# 6. 自定义训练轮数和学习率
    python train.py -d data/ -e 15 -lr 3e-4

================================================================================
💡 参数调优建议
================================================================================

• 上下文长度 (-C): 512(快) / 768(平衡，推荐搭配b6) / 1024(效果好但慢)
• 嵌入维度 (-E) 和头数 (-H): 必须整除，如 -E 768 -H 12 或 -E 512 -H 8
• 层数 (-L): 默认12，减小到6可以大幅降低显存
• 批次大小 (-b): 显存够大用 8，不够减到 4 或 2
• 学习率 (-lr): 默认 5e-4，如果发散降到 1e-4，如果收敛慢升到 1e-3
• 训练轮数 (-e): 一般 10-20 轮，观察验证损失早停

================================================================================
"""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # 必需参数
    parser.add_argument("-d", "--data_path", type=str, default="dataset/pretrain_t2t_mini.jsonl", help="训练数据：文件路径或目录（支持 txt/jsonl，默认 dataset/pretrain_t2t_mini.jsonl）")

    # 可选参数
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="输出目录 (default: ./output)")
    parser.add_argument("-V", "--vocab_size", type=int, default=50000, help="词表大小 (default: 50000)")
    parser.add_argument("-C", "--context_length", type=int, default=512, help="上下文长度 (default: 512)")
    parser.add_argument("-E", "--emb_dim", type=int, default=768, help="嵌入维度 (default: 768)")
    parser.add_argument("-H", "--n_heads", type=int, default=12, help="注意力头数 (default: 12)")
    parser.add_argument("-L", "--n_layers", type=int, default=12, help="Transformer层数 (default: 12)")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="批次大小 (default: 8)")
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-4, help="学习率 (default: 5e-4)")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="训练轮数 (default: 20)")
    parser.add_argument("-s", "--val_split", type=float, default=0.05, help="验证集比例 (default: 0.05)")

    return parser.parse_args()


def load_and_preprocess_data(data_path):
    """加载并预处理文本数据（支持文件、目录、jsonl）"""
    print("\n[阶段1] 加载数据...")

    # 支持目录输入
    if os.path.isdir(data_path):
        # 递归扫描所有子目录下的 .txt 和 .jsonl
        txt_files = glob.glob(os.path.join(data_path, "**", "*.txt"), recursive=True)
        jsonl_files = glob.glob(os.path.join(data_path, "**", "*.jsonl"), recursive=True)
        if not txt_files and not jsonl_files:
            raise FileNotFoundError(f"目录下没有txt或jsonl文件: {data_path}")
        print(f"  数据目录: {data_path}")
        all_text = []
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                all_text.append(content)
                print(f"    - {os.path.basename(txt_file)}: {len(content):,}字符")
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                texts = []
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        if "text" in obj:
                            texts.append(obj["text"])
                    except json.JSONDecodeError:
                        continue
                content = "\n\n".join(texts)
                all_text.append(content)
                print(f"    - {os.path.basename(jsonl_file)}: {len(texts):,}条, {len(content):,}字符")
        text = "\n\n".join(all_text)
    elif data_path.endswith(".jsonl"):
        # jsonl 格式：每行 {"text": "..."}
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        print(f"  数据文件: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            texts = []
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if "text" in obj:
                        texts.append(obj["text"])
                except json.JSONDecodeError:
                    continue
        text = "\n\n".join(texts)
        print(f"  jsonl记录数: {len(texts):,}")
    else:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        print(f"  数据文件: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

    text = text.strip()

    # 统计信息
    total_chars = len(text)
    total_lines = text.count("\n") + 1

    print(f"  总字符数: {total_chars:,}")
    print(f"  总行数: {total_lines:,}")

    # 分割成段落（用于训练BPE）- 按连续空行分割
    paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 10]
    print(f"  有效段落: {len(paragraphs):,}")

    return text, paragraphs


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

    # 创建BPE分词器 - 使用字节级适配中文
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    # 注意：中文文本没有空格分隔，必须使用 Whitespace() 进行预处理
    # CharDelimiterSplit(" ") 对中文无效，会导致整个段落被当作一个token
    tokenizer.pre_tokenizer = Whitespace()

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
    """小说数据集"""

    def __init__(self, text, tokenizer, context_length):
        self.tokenizer = tokenizer
        self.context_length = context_length

        print("\n[阶段3] 创建数据集...")
        print("  编码文本...")

        # 分批编码（避免内存问题）
        chunk_size = 100000
        all_ids = []

        for i in tqdm(range(0, len(text), chunk_size), desc="  编码进度"):
            chunk = text[i : i + chunk_size]
            ids = tokenizer.encode(chunk)
            all_ids.extend(ids)

        self.token_ids = all_ids
        print(f"  总token数: {len(self.token_ids):,}")

        # 计算样本数
        self.num_samples = (len(self.token_ids) - 1) // context_length
        print(f"  样本数: {self.num_samples:,}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length + 1

        chunk = self.token_ids[start:end]

        # 填充
        if len(chunk) < self.context_length + 1:
            chunk = chunk + [self.tokenizer.pad_token_id] * (self.context_length + 1 - len(chunk))

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)

        # 忽略padding的损失
        labels[labels == self.tokenizer.pad_token_id] = -100

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
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
        # Cosine schedule + Warmup:前 3% 步线性升 lr,之后余弦衰减到 0
        if scheduler is None:
            total_steps = config["epochs"] * len(train_loader)
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

    patience = 3
    model_path = os.path.join(output_dir, "model")
    os.makedirs(model_path, exist_ok=True)

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

            optimizer.zero_grad()

            # bf16 autocast:仅前向计算降精度,权重/loss/梯度保持 fp32
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # 每步更新 lr

            train_loss += loss.item()
            train_steps += 1

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
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # 打印配置
    print("=" * 60)
    print("中文小说 GPT 训练")
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

    # 1. 加载数据
    text, paragraphs = load_and_preprocess_data(config["data_path"])

    # 2. 训练分词器
    tokenizer = train_bpe_tokenizer(paragraphs, config["vocab_size"], output_dir)

    # 3. 创建数据集
    full_dataset = NovelDataset(text, tokenizer, config["context_length"])

    # 划分训练/验证集
    train_size = int((1 - config["val_split"]) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    print(f"  训练集: {len(train_dataset):,}")
    print(f"  验证集: {len(val_dataset):,}")

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    # 4. 创建模型
    model = create_model(tokenizer.vocab_size, config, output_dir)

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
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        no_improve_epochs = checkpoint["no_improve_epochs"]
        # 旧 checkpoint 可能没有 scheduler_state_dict(向前兼容)
        if "scheduler_state_dict" in checkpoint:
            loaded_scheduler_state = checkpoint["scheduler_state_dict"]
        print(f"  从第 {start_epoch + 1} 轮继续训练")
        print(f"  最佳验证损失: {best_val_loss:.4f}")
        os.remove(checkpoint_path)

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
