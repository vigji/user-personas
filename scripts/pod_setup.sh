#!/usr/bin/env bash
# Runs ON the pod (fresh RunPod PyTorch image). Idempotent.
set -euo pipefail

# Model cache on /workspace (so weights survive pod restarts when a volume is
# attached) — but only if it is actually writable with room; some hosts mount
# a tiny or full /workspace, and the default /root/.cache works fine there.
if touch /workspace/.writetest 2>/dev/null; then
  rm -f /workspace/.writetest
  grep -q HF_HOME ~/.bashrc 2>/dev/null || echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc
  export HF_HOME=/workspace/hf
fi

pip install -q -U vllm transformers accelerate

python - <<'EOF'
import torch, transformers, vllm
print(f"vllm {vllm.__version__} | transformers {transformers.__version__} | "
      f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
EOF
