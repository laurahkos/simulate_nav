#!/bin/bash
# Build the env and download the models: bash setup_env.sh [--no-models]
# Llama is gated: accept its licence on Hugging Face and export HF_TOKEN first.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

python3 -m venv "$LLMNAV_ENV"
source "$LLMNAV_ENV/bin/activate"
pip install -q --upgrade pip
if [ "$(uname -s)" = Darwin ]; then
    pip install -q torch
else
    pip install -q torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
fi
pip install -q -r requirements.txt
pip check
python -m playwright install chromium

if [ "${1:-}" != --no-models ]; then
    python - <<'EOF'
from huggingface_hub import snapshot_download

for m in ["gpt2", "microsoft/Phi-3-mini-4k-instruct",
          "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.1-8B-Instruct"]:
    snapshot_download(m, ignore_patterns=["*.pth", "original/*", "consolidated*"])
EOF
fi
