# RunPod runbook

**Pods** (rented GPU with SSH, per-second billing while running) are the right
tool for the research loop: activation capture is custom transformers code, the
iteration cycle is edit-file → rsync → rerun over SSH, and results come back as
`.pt` files via rsync. You pay for idle time and must remember to kill the pod.

## Secrets policy

- `.env` (copy of `.env.example`, gitignored) lives only on the local machine.
- `sync_to_pod.sh` explicitly excludes `.env`; nothing secret is ever written
  to pod disk. `HF_TOKEN` is passed inline per SSH command (see below) and
  dies with the shell. Never run `huggingface-cli login` on the pod.
- The HF model cache on the pod volume contains only public weights.

## First mock test (Qwen3-0.6B, ungated, no HF_TOKEN needed)

1. Local: `cp .env.example .env`, set `RUNPOD_API_KEY`, add your SSH public
   key in RunPod Settings → SSH Keys.
2. `bash scripts/pod_deploy.sh` — creates a cheap 4090/A5000/A4000 pod via the
   REST API, waits for SSH, and writes `POD_ID`/`POD_SSH_HOST`/`POD_SSH_PORT`
   into `.env`. (Web-UI deployment works too; then fill those by hand from
   Connect → "SSH over exposed TCP".)
3. `bash scripts/sync_to_pod.sh`
4. `ssh -p $POD_SSH_PORT root@$POD_SSH_HOST 'cd /workspace/user-personas && bash scripts/pod_setup.sh'`
5. Run the two halves:
   `ssh ... 'cd /workspace/user-personas/scripts && python generate_vllm.py --model Qwen/Qwen3-0.6B'`
   `ssh ... 'cd /workspace/user-personas/scripts && python capture_activations.py --model Qwen/Qwen3-0.6B'`
6. `bash scripts/sync_from_pod.sh` → `data/activations/mock_capture.pt` locally.
7. `bash scripts/pod_destroy.sh` — terminates the pod and blanks its
   coordinates in `.env` (stop ≠ free: stopped pods still bill disk).

Success criteria: generation prints the rendered Qwen tags
(`<|im_start|>system` …) and an in-character response; capture prints
`seq_len`, the response span, and a 1024-dim vector; the `.pt` lands locally.
Total cost well under $1.

## Scaling to the paper models

Same flow, three changes: attach a network volume mounted at `/workspace`
(≥250GB; the HF cache in `/workspace/hf` then survives across pods), pick the
GPU per the sizing table (27B/32B → 1× A100/H100 80GB; 70B → 2× H100,
`--tensor-parallel 2`), and prefix run commands with the token for gated
models: `HF_TOKEN=$HF_TOKEN python generate_vllm.py --model google/gemma-2-27b-it ...`
(source `.env` locally first so ssh expands it client-side).
