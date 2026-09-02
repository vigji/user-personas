"""(2) Preload a prefilled system/user/assistant interaction, run one forward
pass with transformers, and capture the residual stream at an arbitrary
(layer, position).

The transcript is rendered with the model's exact chat template tags via
scripts/chat_format.py (the same renderer used for vLLM generation), so the
teacher-forced forward pass sees exactly the tokens the model would have
generated over. ``output_hidden_states=True`` returns the residual stream at
every layer and position in one pass — no hooks needed; arbitrary
(layer, position) selection happens after the fact.

Position semantics (--position):
  response-mean  mean over the final assistant message's tokens (what the
                 paper's role vectors use)         [default]
  last           the last token
  <int>          an absolute token index

Deps: torch, transformers, accelerate. Smoke test on a laptop:
    python scripts/capture_activations.py --model Qwen/Qwen3-0.6B
"""

import argparse
from pathlib import Path

from chat_format import MOCK_TRANSCRIPT, MODEL_CONFIGS, render, response_span

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layer", type=int, default=None,
                    help="layer index (0 = embeddings); default: the model's paper target layer")
    ap.add_argument("--position", default="response-mean")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "activations" / "mock_capture.pt")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    messages = MOCK_TRANSCRIPT
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = render(messages, tokenizer, args.model, add_generation_prompt=False)
    span = response_span(messages, tokenizer, args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    # [n_layers+1, seq, d_model]; index 0 is the embedding layer output.
    # Per-layer .cpu() before stacking: with device_map="auto" on multiple
    # GPUs the layers' hidden states live on different devices.
    hidden = torch.stack([h.squeeze(0).float().cpu() for h in out.hidden_states])
    layer = args.layer if args.layer is not None else MODEL_CONFIGS[args.model]["target_layer"]

    if args.position == "response-mean":
        vector = hidden[layer, span[0] : span[1]].mean(dim=0)
    elif args.position == "last":
        vector = hidden[layer, -1]
    else:
        vector = hidden[layer, int(args.position)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vector": vector,
            "model": args.model,
            "layer": layer,
            "position": args.position,
            "response_span": span,
            "seq_len": hidden.shape[1],
            "messages": messages,
        },
        args.out,
    )
    print(
        f"model={args.model} seq_len={hidden.shape[1]} response_span={span}\n"
        f"captured layer={layer} position={args.position} -> {tuple(vector.shape)}\n"
        f"saved to {args.out}"
    )


if __name__ == "__main__":
    main()
