"""Shared chat formatting for the assistant-axis models.

One renderer used by both generation (vLLM) and activation capture
(transformers), so both see token-identical prompts built with each model's own
chat template tags. Handles the two quirks of the paper's models:

- Gemma 2 has no system role: a system message is folded into the first user
  turn (same trick as upstream assistant-axis generation.py).
- Qwen 3 gets ``enable_thinking=False``.

MODEL_CONFIGS target layers come from upstream assistant_axis/models.py.
"""

MODEL_CONFIGS = {
    "google/gemma-2-27b-it": {"target_layer": 22, "total_layers": 46},
    "Qwen/Qwen3-32B": {"target_layer": 32, "total_layers": 64},
    "meta-llama/Llama-3.3-70B-Instruct": {"target_layer": 40, "total_layers": 80},
    # tiny stand-in for smoke tests on a laptop; same template family as Qwen3-32B
    "Qwen/Qwen3-0.6B": {"target_layer": 14, "total_layers": 28},
}

# Mock scenario for this project: the persona lives in the USER, not the model.
# A fake user posing as the "pirate" persona (data/processed/) issues the
# standard "dinner" request (data/processed/standard_user_requests.json).
MOCK_MESSAGES = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {
        "role": "user",
        "content": (
            "Arr, listen up ye clever contraption — what should I be cookin' "
            "for me supper tonight? All I've got left in the galley is eggs, "
            "rice, and a fistful of spinach."
        ),
    },
]
MOCK_TRANSCRIPT = MOCK_MESSAGES + [
    {
        "role": "assistant",
        "content": (
            "Ahoy! With eggs, rice, and spinach ye can whip up a fine "
            "galley-style egg fried rice: scramble the eggs, fry the rice, "
            "toss the spinach in at the end, and season it well."
        ),
    },
]


def _template_kwargs(model_name: str) -> dict:
    return {"enable_thinking": False} if "qwen" in model_name.lower() else {}


def supports_system(tokenizer) -> bool:
    probe = "PROBE_SYSTEM_PROMPT_XYZ"
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": probe}, {"role": "user", "content": "hi"}],
            tokenize=False,
        )
        return probe in rendered
    except Exception:
        return False


def normalize(messages: list[dict], tokenizer) -> list[dict]:
    """Fold a leading system message into the first user turn if unsupported."""
    if messages and messages[0]["role"] == "system" and not supports_system(tokenizer):
        system, user, *rest = messages
        assert user["role"] == "user"
        merged = {"role": "user", "content": f"{system['content']}\n\n{user['content']}"}
        return [merged, *rest]
    return messages


def render(messages: list[dict], tokenizer, model_name: str, add_generation_prompt: bool) -> str:
    """Render a conversation to the exact prompt string for this model.

    With add_generation_prompt=True the final message must be a user turn (the
    model is about to answer); with False the final message may be a completed
    assistant turn (for teacher-forced forward passes) and is rendered with
    continue_final_message=True, i.e. without the closing turn tag, so the
    response span ends at the last real response token.
    """
    kwargs = _template_kwargs(model_name)
    if not add_generation_prompt:
        kwargs["continue_final_message"] = True
    return tokenizer.apply_chat_template(
        normalize(messages, tokenizer),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def response_span(messages: list[dict], tokenizer, model_name: str) -> tuple[int, int]:
    """Token span [start, end) of the final assistant message in the rendered
    transcript — the positions the paper averages over for role vectors."""
    assert messages[-1]["role"] == "assistant"
    prefix = render(messages[:-1], tokenizer, model_name, add_generation_prompt=True)
    full = render(messages, tokenizer, model_name, add_generation_prompt=False)
    assert full.startswith(prefix), "chat template prefix mismatch"
    n_prefix = len(tokenizer(prefix, add_special_tokens=False).input_ids)
    n_full = len(tokenizer(full, add_special_tokens=False).input_ids)
    return n_prefix, n_full
