"""Build user-side persona definitions from the assistant-axis role data.

Reads the upstream role definitions vendored in data/raw/assistant-axis/
(from https://github.com/safety-research/assistant-axis, data for
arXiv:2601.10387) and emits, under data/processed/:

- personas.txt                    the 275 persona names shipped as role
                                  vectors on HF (lu-christina/assistant-axis-vectors)
- persona_descriptions.json       persona -> original one-line description
- user_persona_instructions.json  persona -> instructions adapted to define a
                                  simulated *user* posing as that persona
                                  (instead of a model embodying it)

The upstream "pos" instructions are model-defining system prompts
("You are an accountant..."). Here each is mechanically rewritten to define a
fake user: the leading directive is stripped and the persona core is wrapped
in a user-simulation template.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "raw" / "assistant-axis"
OUT = REPO / "data" / "processed"

# Leading directives found across all upstream instructions, longest first.
PREFIXES = [
    "Please take on the role of ",
    "Take on the role of ",
    "Take on the persona of ",
    "Embody the role of ",
    "Play the role of ",
    "Assume the role of ",
    "Channel the essence of ",
    "Please respond as ",
    "Please embody ",
    "Please act as ",
    "Please be ",
    "Respond as ",
    "You are ",
    "Act as ",
    "Behave as ",
    "Function as ",
    "Serve as ",
    "Embody ",
    "Be ",
]

USER_TEMPLATE = (
    "You are simulating a human user chatting with an AI assistant. "
    "The user presents themselves as {core} "
    "Write every message in this user's voice, in the first person, making the "
    "request you are given. Never break character and never answer as the assistant."
)


def adapt(pos: str) -> str:
    core = pos
    for p in PREFIXES:
        if pos.startswith(p):
            core = pos[len(p) :]
            break
    else:
        raise ValueError(f"unmatched directive: {pos[:80]}")
    first = core.split()[0]
    if first in ("A", "An", "The"):  # keep proper nouns (e.g. "Leviathan") intact
        core = core[0].lower() + core[1:]
    return USER_TEMPLATE.format(core=core)


def main() -> None:
    descriptions = json.loads((SRC / "role_list.json").read_text())
    # role_list.json holds exactly the 275 personas with HF role vectors;
    # the instructions dir additionally holds the "default" baseline.
    personas = sorted(descriptions)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "personas.txt").write_text("\n".join(personas) + "\n")
    (OUT / "persona_descriptions.json").write_text(
        json.dumps({p: descriptions[p] for p in personas}, indent=2) + "\n"
    )

    adapted = {}
    for p in personas:
        instr = json.loads((SRC / "role_instructions" / f"{p}.json").read_text())
        adapted[p] = {
            "description": descriptions[p],
            "user_simulation_instructions": [adapt(i["pos"]) for i in instr["instruction"]],
        }
    (OUT / "user_persona_instructions.json").write_text(json.dumps(adapted, indent=2) + "\n")
    print(f"wrote {len(personas)} personas to {OUT}")


if __name__ == "__main__":
    main()
