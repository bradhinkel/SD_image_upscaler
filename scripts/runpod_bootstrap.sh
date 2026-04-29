#!/usr/bin/env bash
# Phase 4c bootstrap for a fresh RunPod RTX 5090 pod.
# Idempotent: re-runs land on a working state without duplicating effort.
#
# Expected pre-state:
#   - Pod has uv, git, python3.12 (RunPod's PyTorch template ships these).
#   - HF_TOKEN export set in the calling shell or in /workspace/.env.
#   - data/pairs.tar + data/captions.jsonl already SCP'd into /workspace/.
#
# Usage on the pod:
#   bash /workspace/runpod_bootstrap.sh

set -euo pipefail

WORKSPACE=/workspace
REPO_DIR=$WORKSPACE/SD_image_upscaler
REPO_URL=https://github.com/bradhinkel/SD_image_upscaler.git

cd "$WORKSPACE"

echo "==[1/5]== Cloning repo (or pulling latest)..."
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --rebase
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

echo "==[2/5]== Stage dataset under repo data/ ..."
mkdir -p "$REPO_DIR/data"
if [ -f "$WORKSPACE/pairs.tar" ] && [ ! -d "$REPO_DIR/data/pairs" ]; then
  echo "  extracting pairs.tar -> data/pairs/ (this takes a moment for 15k files)"
  tar -xf "$WORKSPACE/pairs.tar" -C "$REPO_DIR/data/"
fi
if [ -f "$WORKSPACE/captions.jsonl" ] && [ ! -f "$REPO_DIR/data/captions.jsonl" ]; then
  cp "$WORKSPACE/captions.jsonl" "$REPO_DIR/data/captions.jsonl"
fi
echo "  pairs: $(ls "$REPO_DIR/data/pairs" 2>/dev/null | wc -l) files"
echo "  captions: $(wc -l < "$REPO_DIR/data/captions.jsonl" 2>/dev/null) lines"

echo "==[3/5]== Stage .env (HF_TOKEN) ..."
if [ -f "$WORKSPACE/.env" ] && [ ! -f "$REPO_DIR/.env" ]; then
  cp "$WORKSPACE/.env" "$REPO_DIR/.env"
fi

echo "==[4/5]== Install deps with uv (cu128 torch wheels)..."
cd "$REPO_DIR"
uv venv --clear .venv
# shellcheck disable=SC1091
. .venv/bin/activate
uv pip install -e ".[dev]"

echo "==[5/5]== Verify CUDA + sm_120 ..."
python -c "
import torch
print(f'torch {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'device: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'capability: sm_{cap[0]}{cap[1]}')
    assert cap[0] >= 12, f'expected Blackwell sm_12x, got sm_{cap[0]}{cap[1]}'
"

echo
echo "==[ready]== Bootstrap complete."
echo "  Repo:    $REPO_DIR"
echo "  Activate: source $REPO_DIR/.venv/bin/activate"
echo "  Train:   cd $REPO_DIR && nohup .venv/bin/python -m upscaler.lora_train --config configs/sd15_main.yaml > /workspace/training.log 2>&1 &"
