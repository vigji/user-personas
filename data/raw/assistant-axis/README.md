# Vendored data from safety-research/assistant-axis

Copied verbatim from https://github.com/safety-research/assistant-axis (MIT), the
code release for "The Assistant Axis: Situating and Stabilizing the Default
Persona of Language Models" (arXiv:2601.10387; Lu, Gallagher, Michala, Fish,
Lindsey, 2026). Retrieved 2026-09-01.

- `role_list.json` — the 275 character roles and their one-line descriptions
  (upstream `data/roles/role_list.json`). These are exactly the roles shipped as
  role vectors on HF: https://huggingface.co/datasets/lu-christina/assistant-axis-vectors
- `role_instructions/` — per-role files (upstream `data/roles/instructions/`),
  each with 5 model-defining system prompts (`instruction[].pos`) and ~20
  role-typical user questions (`questions`). Includes the extra `default.json`
  baseline (plain Assistant, no role) used to define the axis.
- `extraction_questions.jsonl` — the shared question set used by the upstream
  pipeline when extracting activations.

The paper PDF is in `references/2601.10387_assistant-axis.pdf` at the repo root.
