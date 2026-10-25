#!/usr/bin/env bash
# Download the model weights (~1.0 GB) from Hugging Face into ./weights before `docker build`.
set -euo pipefail
cd "$(dirname "$0")"
REPO=${HF_REPO:-seantangth/WACV2027-RetailAction-Challenge-weights}
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --local-dir weights
else
  python3 -m pip install -q "huggingface_hub>=0.24"
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$REPO', local_dir='weights')"
fi
echo "weights: $(find weights -name '*.pt' | wc -l) checkpoints, $(du -sh weights | cut -f1)"
