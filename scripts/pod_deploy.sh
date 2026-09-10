#!/usr/bin/env bash
# Create a RunPod pod via the REST API, wait until it is reachable, and write
# POD_ID / POD_SSH_HOST / POD_SSH_PORT back into .env. Needs curl + jq and
# RUNPOD_API_KEY in .env. Account SSH keys (RunPod Settings) are injected into
# the official image automatically.
#
# Defaults target the cheap mock pod; override via env, e.g.:
#   GPU_TYPES='["NVIDIA H100 80GB HBM3"]' GPU_COUNT=2 DISK_GB=300 bash scripts/pod_deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY in .env}"
command -v jq >/dev/null || { echo "jq required (brew install jq)"; exit 1; }

API=https://rest.runpod.io/v1
AUTH="Authorization: Bearer ${RUNPOD_API_KEY}"
NAME="${NAME:-persona-mock}"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
GPU_TYPES="${GPU_TYPES:-[\"NVIDIA GeForce RTX 4090\", \"NVIDIA RTX A5000\", \"NVIDIA RTX A4000\"]}"
GPU_COUNT="${GPU_COUNT:-1}"
DISK_GB="${DISK_GB:-40}"

set_env() { # update or append KEY=VALUE in .env
  if grep -q "^$1=" .env; then
    sed -i.bak "s|^$1=.*|$1=$2|" .env && rm -f .env.bak
  else
    echo "$1=$2" >> .env
  fi
}

# The official image installs $PUBLIC_KEY into authorized_keys on boot;
# account-level SSH keys are not injected into REST-created pods.
PUBKEY=$(cat "${SSH_PUBKEY_FILE:-$HOME/.ssh/id_ed25519.pub}")

body=$(jq -n \
  --arg name "$NAME" --arg image "$IMAGE" --arg pubkey "$PUBKEY" \
  --argjson gpus "$GPU_TYPES" --argjson count "$GPU_COUNT" --argjson disk "$DISK_GB" \
  '{name: $name, imageName: $image, gpuTypeIds: $gpus, gpuCount: $count,
    containerDiskInGb: $disk, ports: ["22/tcp"], cloudType: "SECURE",
    env: {PUBLIC_KEY: $pubkey}}')

resp=$(curl -sf -X POST "$API/pods" -H "$AUTH" -H "Content-Type: application/json" -d "$body") \
  || { echo "pod creation failed"; exit 1; }
pod_id=$(echo "$resp" | jq -r .id)
echo "created pod $pod_id ($(echo "$resp" | jq -r .costPerHr) \$/hr), waiting for SSH..."

for _ in $(seq 1 60); do
  sleep 5
  pod=$(curl -sf "$API/pods/$pod_id" -H "$AUTH")
  ip=$(echo "$pod" | jq -r '.publicIp // empty')
  port=$(echo "$pod" | jq -r '.portMappings["22"] // empty')
  if [[ -n "$ip" && -n "$port" ]]; then
    set_env POD_ID "$pod_id"
    set_env POD_SSH_HOST "$ip"
    set_env POD_SSH_PORT "$port"
    echo "ready: ssh -p $port ${POD_SSH_USER:-root}@$ip   (written to .env)"
    exit 0
  fi
done

echo "pod $pod_id not reachable after 5 min; check the RunPod console (it is still billing)"
exit 1
