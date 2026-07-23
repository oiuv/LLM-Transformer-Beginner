"""
中文小说 GPT 文本生成脚本
使用方法：
    # 基础生成
    python generate.py --model output/model --prompt "第一章"
    
    # 交互模式
    python generate.py --model output/model -i
    
    # 自定义参数
    python generate.py --model output/model --prompt "第一章" --length 1000 --temperature 0.9
"""

import argparse
import os
import sys
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


class GPTGenerator:
    """GPT 文本生成器"""

    def __init__(self, model_dir, device=None):
        """
        初始化生成器
        Args:
            model_dir: 模型目录路径
            device: 计算设备，None则自动选择
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)

        print("加载模型...")

        # 查找 tokenizer.json
        possible_paths = [
            os.path.join(model_dir, "tokenizer.json"),
            os.path.join(os.path.dirname(model_dir), "tokenizer.json"),
        ]

        tokenizer_file = None
        for path in possible_paths:
            if os.path.exists(path):
                tokenizer_file = path
                break

        if tokenizer_file is None:
            raise FileNotFoundError(f"找不到 tokenizer.json，尝试路径: {possible_paths}")

        # 加载分词器（使用与训练一致的 PreTrainedTokenizerFast）
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

        # 加载模型
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"设备: {self.device}")
        print(f"词表大小: {self.tokenizer.vocab_size}")
        print("模型加载完成!\n")

    def encode(self, text):
        """编码文本"""
        return self.tokenizer.encode(text)

    def decode(self, ids, skip_special_tokens=True):
        """解码token，并进行中文后处理"""
        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        # 中文后处理:清理 ByteLevel pre_tokenizer 留下的"词首空格"标记
        # ByteLevel 用 Ġ(U+0120)代替 ASCII 空格表示"这个 token 原本是词首"
        # 中文输出里这些字符没有语义,必须删掉,否则会污染生成结果
        # 注意:不是替换为 ASCII 空格(会让中文词之间多出空格),而是直接删除
        text = text.replace("Ġ", " ").replace(" ", "")
        return text

    def generate(self, prompt, max_length=500, temperature=0.8, top_p=0.9, repetition_penalty=1.1):
        """
        生成文本
        
        Args:
            prompt: 提示文本
            max_length: 生成token数量
            temperature: 温度参数（越高越随机）
            top_p: Top-p采样阈值
            repetition_penalty: 重复惩罚系数
            
        Returns:
            生成的完整文本（包含prompt）
        """
        # 编码输入
        input_ids = self.encode(prompt)
        generated_ids = input_ids.copy()

        with torch.no_grad():
            for _ in range(max_length):
                # 准备输入
                input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(self.device)

                # 前向传播
                outputs = self.model(input_tensor)
                logits = outputs.logits[:, -1, :]  # 取最后一个token的logits

                # 应用温度
                # temperature 越小，分布越尖锐（更确定）；越大，越平坦（更随机）
                # 用 max 防止 temperature=0 时除零；接近 0 时退化为 greedy（argmax）
                logits = logits / max(temperature, 1e-5)

                # 应用重复惩罚
                # 正确实现：需要根据logits的符号区分处理
                # 正向logits：除以penalty（降低概率）
                # 负向logits：乘以penalty（提升概率，避免过度惩罚）
                if repetition_penalty != 1.0:
                    for token_id in set(generated_ids):
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= repetition_penalty
                        else:
                            logits[0, token_id] *= repetition_penalty

                # Top-p 采样（nucleus sampling）
                # 标准的 nucleus sampling 实现
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

                # 创建掩码：保留第一个超过 top_p 的token及其之前的所有token
                # 移除第一个超过阈值的token之后的所有token
                sorted_indices_to_remove = cumsum_probs > top_p
                # 左移掩码，确保第一个超过阈值的token也被移除（只保留不超过阈值的）
                sorted_indices_to_remove = torch.cat([
                    sorted_indices_to_remove[..., :1] * False,
                    sorted_indices_to_remove[..., :-1]
                ], dim=-1)

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = float('-inf')

                # 重新计算概率并采样
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                generated_ids.append(next_token)

        # 解码并返回
        return self.decode(generated_ids)


def demo(generator):
    """演示生成多个示例"""
    print("=" * 60)
    print("生成示例")
    print("=" * 60)

    # 测试提示
    prompts = [
        "第一章",
        "话说",
        "江湖",
        "那一年",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}] 提示: {prompt}")
        print("-" * 40)

        output = generator.generate(prompt, max_length=150)
        print(output)
        print()


def interactive(generator):
    """交互模式"""
    print("=" * 60)
    print("交互模式")
    print("输入提示文本，模型将续写故事")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'demo' 查看示例")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n请输入提示: ").strip()

            if prompt.lower() in ("quit", "exit", "q"):
                print("再见!")
                break

            if prompt.lower() == "demo":
                demo(generator)
                continue

            if not prompt:
                continue

            print("\n生成中...")
            output = generator.generate(prompt, max_length=200)

            print("\n" + "=" * 60)
            print(output)
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="中文小说 GPT 文本生成")

    # 必需参数
    parser.add_argument("--model", "-m", type=str, required=True, help="模型路径（必需）")

    # 可选参数
    parser.add_argument("--prompt", "-p", type=str, help="生成提示文本")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--length", "-l", type=int, default=500, help="生成token数量 (default: 500)")
    parser.add_argument("--temperature", "-t", type=float, default=0.8, help="温度参数，越高越随机 (default: 0.8)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p采样，控制多样性 (default: 0.9)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚 (default: 1.1)")
    parser.add_argument("--device", type=str, default=None, help="设备：cuda/cpu (default: auto)")

    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化生成器
    print("=" * 60)
    print("中文小说 GPT 文本生成")
    print("=" * 60)
    print(f"模型路径: {args.model}")

    try:
        generator = GPTGenerator(args.model, device=args.device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        sys.exit(1)

    # 交互模式
    if args.interactive:
        interactive(generator)
        return

    # 单次生成模式
    if not args.prompt:
        print("错误: 请提供 --prompt 参数或使用 --interactive 进入交互模式")
        print("示例: python generate.py -m output/model -p '第一章'")
        sys.exit(1)

    print(f"提示: {args.prompt}")
    print(f"生成长度: {args.length}")
    print(f"温度: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print("=" * 60)

    print("\n生成中...")
    output = generator.generate(
        args.prompt,
        max_length=args.length,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    print("\n" + "=" * 60)
    print("生成结果")
    print("=" * 60)
    print(output)
    print("\n" + "=" * 60)
    print(f"生成完成！总长度: {len(output)} 字符")
    print("=" * 60)


if __name__ == "__main__":
    main()
