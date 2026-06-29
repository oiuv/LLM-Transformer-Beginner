# AGENTS.md — 给 AI 协作者的工程说明

> 本文件面向 AI 工具（Claude、Codex、Hermes 等）和未来的协作者，明确本项目的**性质**与**演化边界**，避免误判项目类型而提出过度工程化的修改建议。
> 人类读者的入门指南请看 [`README.md`](README.md)。

---

## 1. 项目性质（必读）

这是一个**个人学习教程仓库**，记录作者从零开始学习 ML/DL/LLM 的过程。它**不是**工程化产品，**不是**开源框架，**不是**为生产部署准备的代码库。

由此推出三条不可违反的演化边界：

1. **任何修改不得增加复杂度**。如果某个改动是"工程化最佳实践但会让初学者更难读"——**不做**。
2. **朴素实现是教学价值，不是缺陷**。手写 loop、显式参数、字典 config 都有教学意义——**别为换 `dataclass` / `pydantic` / `hydra` 重构**。
3. **依赖保持最小**。`torch` + `transformers` + `tokenizers` + `jupyter` + `tqdm` 之外，**不要引入新依赖**——它们会产生初学者的环境门槛。

---

## 2. 文件地图

| 路径 | 用途 |
|---|---|
| [`01-basics/`](01-basics/) | 从 `y=wx` 到神经网络的入门教程，4 个 `.ipynb`，不使用 ML 框架 |
| [`02-llm-from-scratch/`](02-llm-from-scratch/) | LLM 原理三条平行路径：fundamentals（9 章系统）/ llama3-step-by-step（10 课渐进）/ llama3（单 notebook 完整） |
| [`03-practice/chinese-gpt/`](03-practice/chinese-gpt/) | 中文小说 GPT 实战，用 `transformers` 训练 + 推理 |
| [`scripts/`](scripts/) | 编码扫描转换 (GBK→UTF-8) + 一键训练启动脚本（bash + PowerShell） |
| `04-future/` | 空目录 |
| `tests/` | playground，不是 unit test |

### 关键文件

- `03-practice/chinese-gpt/train.py` — 训练脚本。最近反复迭代，**注释里"为什么这样做"比代码本身更有价值**（看 `bf16` / `min_frequency=1` / 递归扫描几处注释）。
- `03-practice/chinese-gpt/generate.py` — 推理脚本。朴素手写版，无 KV-Cache，这是有意的——初学者看到每步重算，自然会问"难道不能缓存吗"，引出 KV-Cache 概念。
- `scripts/run_full_training.sh` / `.ps1` — Linux/macOS 与 Windows 的一键训练启动器（含环境检查、CUDA/bf16 检测、checkpoint 冲突确认）。

---

## 3. 演化原则

### ✅ 鼓励的方向

- **把踩过的坑沉淀成代码默认值**（参见 `train.py` 近期 commits）：
  - `bf16 混合精度`（RTX 4090+ 加速）
  - `Cosine schedule + Warmup`（收敛更稳）
  - BPE `min_frequency=1`（中文生僻字不进 `<unk>`）
  - 数据目录**递归扫描**（解决 ebook/ 子目录扫不到的问题）
  - 断点续训 + 早停 + 显存监控
- **修复明显错误**：typo、broken link、会 silent failure 的 bug、dead import
- **补充注释里"为什么这样做"的背景**——这是教学价值
- **新增教程或可视化 PNG**
- **调整 README / 文档结构**——对齐 README 在项目中的定位（"面向零基础读者的入门指南"）即可，不必拘泥现有措辞

### ❌ 禁止的方向

- 引入新的第三方依赖（在 §1 列出的 5 个之外）
- 把朴素实现"优化"成工程化版本：加 KV-Cache、换 `dataclass`、加 type hints 全覆盖
- 把单个脚本拆成多文件 package、加抽象层（base class / plugin / config schema）
- 加 `requirements.txt` / `pyproject.toml` / CI / lint / test framework 之类的工程化基础设施

### ⚠️ 需要先确认才做

- 任何 §3 没覆盖到、可能影响学习曲线或项目性质的改动——先问用户
- 未经用户明确要求，不执行 `git commit` / `git push`

---

## 4. 常用命令

```bash
# 训练
python 03-practice/chinese-gpt/train.py -d data/

# 生成
python 03-practice/chinese-gpt/generate.py --model output/model --prompt "第一章"

# 一键训练（含环境检查）
bash scripts/run_full_training.sh                    # Linux/macOS
powershell -File scripts/run_full_training.ps1       # Windows
```

---

## 5. 上下文备注

- Windows 上的 bash 是 MSYS（Git Bash），用 POSIX 路径
- `python` 指 python3，`py` 也是有效 launcher
- 数据和输出目录在 `.gitignore` 中显式排除
- 项目**不含**任何 LLM 权重文件（`02-llm-from-scratch/llama3/` 是讲解性 notebook，不加载真实权重）
