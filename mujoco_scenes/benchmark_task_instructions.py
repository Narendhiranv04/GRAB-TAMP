"""The one task instruction each benchmark domain issues, verbatim.

Every method -- the proposed functional-grounding pipeline and every
comparison baseline -- must receive the *same* instruction for a domain, or the
comparison measures the instruction rather than the method.  The text was
previously written out separately in the batch runner, the scene modules, the
pipeline's domain definitions and several baselines, and those copies drifted:
the Workshop instruction handed to the baselines named the object categories
and prescribed the insertion geometry ("the compatible screw", "the first
compatible driver", "tip-down", "head/recess on top") where the published
instruction says only "the compatible components required to complete the
fastening".  That gave the baselines information the proposed method never
received, and it inflated them.

These strings are the published Table I instructions.  Import them; do not
restate them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml


# One source of truth: the scene configuration files.  ViLaIn-TAMP already
# reads the instruction from these (runtime._TASK_CONFIG), and so do the other
# baselines' planning runtimes, so restating the text in Python would recreate
# exactly the drift this module exists to prevent.
_CONFIG_SOURCES = {
    "kitchen": ("kitchen_feasibility_variants.yaml", "goal_instruction"),
    "living_room": ("living_room_variants.yaml", "task"),
    "workshop": ("workshop_variants.yaml", "canonical_task_instruction"),
}

_CONFIG_ROOT = Path(__file__).resolve().parent / "configs"


def _load(domain: str) -> str:
    filename, key = _CONFIG_SOURCES[domain]
    document = yaml.safe_load(
        (_CONFIG_ROOT / filename).read_text(encoding="utf-8")
    )
    value = document.get(key) if isinstance(document, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{filename} is missing a usable {key!r} task instruction"
        )
    return " ".join(value.split())


KITCHEN_TASK_INSTRUCTION = _load("kitchen")
LIVING_ROOM_TASK_INSTRUCTION = _load("living_room")
WORKSHOP_TASK_INSTRUCTION = _load("workshop")

TASK_INSTRUCTIONS = {
    "kitchen": KITCHEN_TASK_INSTRUCTION,
    "living_room": LIVING_ROOM_TASK_INSTRUCTION,
    "workshop": WORKSHOP_TASK_INSTRUCTION,
}


def task_instruction(domain: str) -> str:
    """Return the published instruction for one benchmark domain."""
    try:
        return TASK_INSTRUCTIONS[domain]
    except KeyError:
        raise ValueError(
            f"No benchmark task instruction for domain {domain!r}; "
            f"expected one of {sorted(TASK_INSTRUCTIONS)}"
        ) from None
