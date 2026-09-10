#!/usr/bin/env bash
# Push the repo to the pod. Code + small data only: never .env (secrets stay
# local), never large artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
POD_DIR="${POD_DIR:-/workspace/user-personas}"
ssh -p "${POD_SSH_PORT}" "${POD_SSH_USER}@${POD_SSH_HOST}" \
  'command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)'
# -rltvz, not -avz: pod /workspace may be a network FS that forbids chown/chmod
rsync -rltvz -e "ssh -p ${POD_SSH_PORT}" \
  --exclude '.git' --exclude '.env' --exclude '__pycache__' \
  --exclude 'data/activations' --exclude 'data/weights' \
  ./ "${POD_SSH_USER}@${POD_SSH_HOST}:${POD_DIR}/"
