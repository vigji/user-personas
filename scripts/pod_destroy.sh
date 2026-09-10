#!/usr/bin/env bash
# Terminate the pod recorded in .env (POD_ID) and blank its coordinates.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY in .env}"
: "${POD_ID:?no POD_ID in .env}"

curl -sf -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" > /dev/null
for key in POD_ID POD_SSH_HOST POD_SSH_PORT; do
  sed -i.bak "s|^${key}=.*|${key}=|" .env && rm -f .env.bak
done
echo "terminated pod $POD_ID"
