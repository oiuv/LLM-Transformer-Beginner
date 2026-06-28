#!/usr/bin/env bash
# ============================================================
# 中文小说 GPT 全量训练一键启动脚本
# ============================================================
# 用法:
#   bash scripts/run_full_training.sh             # 默认配置
#   bash scripts/run_full_training.sh --check-only # 只检查环境,不训练
#
# 默认配置:
#   - 数据: 全量 UTF-8 小说(7461 个文件,几百 MB)
#   - 模型: 124M(≈ GPT-2 Small)
#   - 上下文: 1024
#   - batch: 8  (显存 ~14-15GB,留 1-2GB 余量)
#   - lr:   8e-4 (batch 翻倍,lr 同步调高)
#   - 轮数: 10   (大数据量下,10 epoch 就够)
# ============================================================

set -euo pipefail

# 路径解析(相对于脚本位置)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$PROJECT_ROOT/03-practice/chinese-gpt/data"
TRAIN_SCRIPT="$PROJECT_ROOT/03-practice/chinese-gpt/train.py"
OUTPUT_DIR="$PROJECT_ROOT/output"

# 默认训练参数
CONTEXT_LENGTH=1024
BATCH_SIZE=8
LEARNING_RATE=8e-4
EPOCHS=10
EMB_DIM=768
N_HEADS=12
N_LAYERS=12
VOCAB_SIZE=50000

# 解析参数(简化版,只支持 --check-only)
CHECK_ONLY=false
for arg in "$@"; do
    case $arg in
        --check-only) CHECK_ONLY=true ;;
        *) echo "未知参数: $arg(目前只支持 --check-only)"; exit 1 ;;
    esac
done

echo "============================================================"
echo "中文小说 GPT 全量训练 — 一键启动"
echo "============================================================"
echo "项目根: $PROJECT_ROOT"
echo "数据:   $DATA_DIR"
echo "输出:   $OUTPUT_DIR"
echo "============================================================"
echo

# ---------- 1. 环境检查 ----------
echo "[1/4] 环境检查..."

# Python:优先用 py(Windows 上 torch 装在 py 指向的环境),再退回 python3 / python
PY=""
for cmd in py python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        if "$cmd" -c "import torch, transformers, tokenizers" >/dev/null 2>&1; then
            PY="$cmd"
            break
        elif [ -z "$PY" ]; then
            # 记录候选,即使依赖没装
            PY="$cmd"
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "❌ 找不到 python/py 命令"
    exit 1
fi
echo "  ✓ Python: $($PY --version 2>&1) (命令: $PY)"

# 关键依赖(必须装好)
if ! $PY -c "import torch, transformers, tokenizers" >/dev/null 2>&1; then
    echo "❌ 缺少依赖(torch/transformers/tokenizers)"
    echo "  安装: $PY -m pip install torch transformers tokenizers"
    exit 1
fi
echo "  ✓ torch=$($PY -c 'import torch;print(torch.__version__)'), transformers=$($PY -c 'import transformers;print(transformers.__version__)')"

# CUDA
CUDA_AVAILABLE=$($PY -c "import torch;print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo "⚠️  未检测到 CUDA,将用 CPU 训练(会非常慢,不推荐)"
    read -p "  继续吗?(y/N) " ans
    [ "$ans" = "y" ] || [ "$ans" = "Y" ] || exit 1
else
    GPU_NAME=$($PY -c "import torch;print(torch.cuda.get_device_name(0))" 2>/dev/null)
    GPU_MEM=$($PY -c "import torch;print(f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}')" 2>/dev/null)
    BF16_OK=$($PY -c "import torch;print(torch.cuda.is_bf16_supported())" 2>/dev/null || echo "False")
    echo "  ✓ GPU: $GPU_NAME (${GPU_MEM}GB)"
    [ "$BF16_OK" = "True" ] && echo "  ✓ bf16 支持(训练会启用)" || echo "  ⚠️  不支持 bf16(会回退到 fp32,显存占用翻倍)"
fi

# 数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 数据目录不存在: $DATA_DIR"
    exit 1
fi
TXT_COUNT=$(find "$DATA_DIR" -name "*.txt" 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh "$DATA_DIR" 2>/dev/null | awk '{print $1}')
echo "  ✓ 数据: $TXT_COUNT 个 txt 文件,共 $TOTAL_SIZE"

# 输出目录
if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/model/checkpoint.pt" ]; then
    echo "⚠️  发现已有 checkpoint.pt(断点续训): $OUTPUT_DIR/model/checkpoint.pt"
    read -p "  继续会从断点恢复,是否继续?(y/N) " ans
    [ "$ans" = "y" ] || [ "$ans" = "Y" ] || exit 1
elif [ -d "$OUTPUT_DIR" ]; then
    echo "  ⚠️  输出目录已存在: $OUTPUT_DIR(会复用已有分词器/model)"
fi

if [ "$CHECK_ONLY" = true ]; then
    echo
    echo "[check-only 模式] 环境检查完成,未启动训练"
    exit 0
fi

# ---------- 2. 显示配置 ----------
echo
echo "[2/4] 训练配置:"
echo "  数据:       全量 ($DATA_DIR)"
echo "  模型:       ~124M(emb=$EMB_DIM, heads=$N_HEADS, layers=$N_LAYERS)"
echo "  词表:       $VOCAB_SIZE"
echo "  上下文:     $CONTEXT_LENGTH"
echo "  batch:      $BATCH_SIZE"
echo "  学习率:     $LEARNING_RATE"
echo "  轮数:       $EPOCHS"
echo "  输出:       $OUTPUT_DIR"

# ---------- 3. 预估时间 ----------
echo
echo "[3/4] 预估时间(根据上轮经验 ~6.4 it/s,新 batch ~10 it/s):"
# 假设全量数据 token 数与单本相似(约 350 万 token × 100 倍数据量)
EST_TOKEN_COUNT=$((3500000 * 50))   # 粗估,实际看 log
EST_SAMPLES=$((EST_TOKEN_COUNT / CONTEXT_LENGTH))
EST_STEPS_PER_EPOCH=$((EST_SAMPLES / BATCH_SIZE))
EST_TIME_PER_EPOCH=$((EST_STEPS_PER_EPOCH / 10 / 60))
EST_TOTAL_TIME=$((EST_TIME_PER_EPOCH * EPOCHS))
echo "  每 epoch 约 $EST_TIME_PER_EPOCH 分钟,共 $EPOCHS 轮 ≈ $EST_TOTAL_TIME 分钟"
echo "  (实际时间取决于数据加载速度,首次跑会偏长——分词器训练 + 数据集构建)"

# ---------- 4. 启动训练 ----------
echo
echo "[4/4] 启动训练...(Ctrl+C 可中断,会保存 checkpoint 供下次续训)"
echo "============================================================"
echo

cd "$PROJECT_ROOT"
$PY "$TRAIN_SCRIPT" \
    -d "$DATA_DIR" \
    -o "$OUTPUT_DIR" \
    -V "$VOCAB_SIZE" \
    -C "$CONTEXT_LENGTH" \
    -E "$EMB_DIM" \
    -H "$N_HEADS" \
    -L "$N_LAYERS" \
    -b "$BATCH_SIZE" \
    -lr "$LEARNING_RATE" \
    -e "$EPOCHS"