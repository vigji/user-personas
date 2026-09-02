"""(1) Efficient token generation with vLLM for the assistant-axis models.

Prompts are pre-rendered with scripts/chat_format.py (exact model tags), then
handed to vLLM as plain strings — vLLM's continuous batching does the rest, and
prefix caching makes many conversations sharing a prefix (e.g. 275 personas x 8
standard requests with the same system prompt) nearly free to prefill.

Deps: vllm (pulls torch + transformers). Smoke test on a small GPU box:
    python scripts/generate_vllm.py --model Qwen/Qwen3-0.6B
Paper models: --model google/gemma-2-27b-it | Qwen/Qwen3-32B |
meta-llama/Llama-3.3-70B-Instruct (set --tensor-parallel to the GPU count).
"""

import argparse
import json
from pathlib import Path

from chat_format import MOCK_MESSAGES, render


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--conversations", type=Path, default=None,
                    help="JSON file: list of message lists; defaults to the mock interaction")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="optional output JSONL")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    conversations = (
        json.loads(args.conversations.read_text())
        if args.conversations
        else [MOCK_MESSAGES]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [render(m, tokenizer, args.model, add_generation_prompt=True) for m in conversations]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        enable_prefix_caching=True,
        max_model_len=4096,
    )
    outputs = llm.generate(
        prompts, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    )

    records = []
    for messages, out in zip(conversations, outputs):
        response = out.outputs[0].text
        records.append({"messages": messages, "response": response})
        print(f"--- prompt (rendered) ---\n{out.prompt}\n--- response ---\n{response}\n")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in records))
        print(f"wrote {len(records)} generations to {args.out}")


if __name__ == "__main__":
    main()
