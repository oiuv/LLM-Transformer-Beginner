# ============================================================
# Chinese Novel GPT Full Training — One-Click Starter (PowerShell)
# ============================================================
# Usage (from project root):
#   .\scripts\run_full_training.ps1                # default: keep tokenizer, retrain model
#   .\scripts\run_full_training.ps1 -CleanOutput   # delete output/ first (rebuild everything)
#   .\scripts\run_full_training.ps1 -CheckOnly
#   .\scripts\run_full_training.ps1 -SmallModel    # small model for quick test
#
# Default config (51M model + 11.5B tokens data):
#   - Data:  full UTF-8 novels (7461 files, 1.65GB chars, ~1.15B tokens)
#   - Model: 51M (emb=512, heads=8, layers=8)
#   - Context: 1024
#   - batch:  8
#   - lr:     5e-4
#   - Epochs: 1  (matches Chinchilla optimal ratio for 51M model)
# ============================================================

param(
    [switch]$CheckOnly,
    [switch]$SmallModel,
    [switch]$CleanOutput
)

$ErrorActionPreference = "Stop"

# Paths (relative to project root)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$DataDir = Join-Path $ProjectRoot "03-practice\chinese-gpt\data"
$TrainScript = Join-Path $ProjectRoot "03-practice\chinese-gpt\train.py"
$OutputDir = Join-Path $ProjectRoot "output"

# Model config
if ($SmallModel) {
    $EmbDim = 384
    $NHeads = 6
    $NLayers = 6
    $ContextLength = 1024
    $BatchSize = 4
    $LearningRate = 1e-3
    $Epochs = 30
    $ModelName = "30M Small (Quick Test)"
}
else {
    $EmbDim = 512
    $NHeads = 8
    $NLayers = 8
    $ContextLength = 1024
    $BatchSize = 8
    $LearningRate = 5e-4
    $Epochs = 1
    $ModelName = "51M Standard"
}

$VocabSize = 50000

Write-Host "============================================================"
Write-Host "Chinese Novel GPT Full Training"
Write-Host "============================================================"
Write-Host "Project:  $ProjectRoot"
Write-Host "Data:     $DataDir"
Write-Host "Output:   $OutputDir"
Write-Host "Mode:     $ModelName"
Write-Host "============================================================"
Write-Host ""

# ---------- 1. Environment check ----------
Write-Host "[1/4] Checking environment..."

# Find Python (prefer py, fallback to python/python3)
$Py = $null
foreach ($cmd in @("py", "python", "python3")) {
    $found = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($found) {
        $testOutput = & $cmd -c "import torch, transformers, tokenizers" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $Py = $cmd
            break
        }
        elseif (-not $Py) {
            $Py = $cmd
        }
    }
}

if (-not $Py) {
    Write-Host "[X] Python not found"
    exit 1
}

$pyVer = & $Py --version 2>&1
Write-Host "  [OK] Python: $pyVer (cmd: $Py)"

# Check key deps
$depCheck = & $Py -c "import torch, transformers, tokenizers" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[X] Missing deps (torch/transformers/tokenizers)"
    Write-Host "    Install: $Py -m pip install torch transformers tokenizers"
    exit 1
}

$torchVer = & $Py -c "import torch; print(torch.__version__)"
$hfVer = & $Py -c "import transformers; print(transformers.__version__)"
Write-Host "  [OK] torch=$torchVer, transformers=$hfVer"

# CUDA check
$cudaAvail = & $Py -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($cudaAvail -eq "True") {
    $gpuName = & $Py -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null
    $gpuMem = & $Py -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}')" 2>$null
    $bf16Ok = & $Py -c "import torch; print(torch.cuda.is_bf16_supported())" 2>$null
    Write-Host "  [OK] GPU: $gpuName (${gpuMem}GB)"
    if ($bf16Ok -eq "True") {
        Write-Host "  [OK] bf16 supported (will be enabled)"
    }
    else {
        Write-Host "  [WARN] bf16 NOT supported (falls back to fp32, ~2x VRAM)"
    }
}
else {
    Write-Host "[WARN] CUDA not detected, will train on CPU (very slow)"
    $ans = Read-Host "  Continue? (y/N)"
    if ($ans -ne "y" -and $ans -ne "Y") { exit 1 }
}

# Data dir
if (-not (Test-Path $DataDir)) {
    Write-Host "[X] Data dir not found: $DataDir"
    exit 1
}
$txtFiles = Get-ChildItem -Path $DataDir -Filter "*.txt" -Recurse -File
$txtCount = $txtFiles.Count
$totalBytes = ($txtFiles | Measure-Object -Property Length -Sum).Sum
$totalSize = "{0:N1} MB" -f ($totalBytes / 1MB)
Write-Host "  [OK] Data: $txtCount txt files, $totalSize total"

# Output dir
if ((Test-Path "$OutputDir\model\checkpoint.pt")) {
    Write-Host "[WARN] checkpoint.pt found (will resume): $OutputDir\model\checkpoint.pt"
    $ans = Read-Host "  Resume from checkpoint? (y/N)"
    if ($ans -ne "y" -and $ans -ne "Y") { exit 1 }
}
elseif (Test-Path $OutputDir) {
    Write-Host "  [INFO] Output dir exists: $OutputDir (will reuse tokenizer/model)"
}

if ($CheckOnly) {
    Write-Host ""
    Write-Host "[CheckOnly] Environment check complete, training NOT started"
    exit 0
}

# ---------- 1.5. Clean output if requested ----------
if ($CleanOutput) {
    if (Test-Path $OutputDir) {
        Write-Host ""
        Write-Host "[1.5/4] -CleanOutput specified, removing $OutputDir ..."
        Remove-Item -Path $OutputDir -Recurse -Force
        Write-Host "  [OK] output/ removed (next start will rebuild tokenizer + model)"
    }
    else {
        Write-Host ""
        Write-Host "[1.5/4] -CleanOutput specified, but $OutputDir doesn't exist (nothing to remove)"
    }
}

# ---------- 2. Show config ----------
Write-Host ""
Write-Host "[2/4] Training config:"
Write-Host "  Data:      Full ($DataDir)"
Write-Host "  Model:     $ModelName (emb=$EmbDim, heads=$NHeads, layers=$NLayers)"
Write-Host "  Vocab:     $VocabSize"
Write-Host "  Context:   $ContextLength"
Write-Host "  Batch:     $BatchSize"
Write-Host "  LR:        $LearningRate"
Write-Host "  Epochs:    $Epochs"
Write-Host "  Output:    $OutputDir"

# ---------- 3. Estimate time ----------
Write-Host ""
if ($SmallModel) {
    Write-Host "[3/4] Time estimate (small model ~20 it/s):"
    $itPerSec = 20
}
else {
    Write-Host "[3/4] Time estimate (51M model ~12 it/s):"
    $itPerSec = 12
}
$estTokenCount = 180000000
$estSamples = [math]::Floor($estTokenCount / $ContextLength)
$estStepsPerEpoch = [math]::Floor($estSamples / $BatchSize)
$estSecsPerEpoch = $estStepsPerEpoch / $itPerSec
$estMinsPerEpoch = [math]::Round($estSecsPerEpoch / 60)
$estTotalMins = $estMinsPerEpoch * $Epochs
Write-Host "  Per epoch ~$estMinsPerEpoch min, $Epochs epochs total ~$estTotalMins min"
Write-Host "  (Actual time depends on data loading; first run is slower due to BPE + dataset build)"

# ---------- 4. Launch ----------
Write-Host ""
Write-Host "[4/4] Starting training... (Ctrl+C to interrupt, will save checkpoint for resume)"
Write-Host "============================================================"
Write-Host ""

Set-Location $ProjectRoot
& $Py $TrainScript `
    -d $DataDir `
    -o $OutputDir `
    -V $VocabSize `
    -C $ContextLength `
    -E $EmbDim `
    -H $NHeads `
    -L $NLayers `
    -b $BatchSize `
    -lr $LearningRate `
    -e $Epochs