#!/usr/bin/env bash
# Pull results back from the pod into local data/activations/.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
POD_DIR="${POD_DIR:-/workspace/user-personas}"
rsync -rltvz -e "ssh -p ${POD_SSH_PORT}" \
  "${POD_SSH_USER}@${POD_SSH_HOST}:${POD_DIR}/data/activations/" \
  data/activations/
