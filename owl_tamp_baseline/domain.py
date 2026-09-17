"""Small domain compiler and OWL-TAMP relaxed grounding."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping

from .models import Action, ValidationError


@dataclass(frozen=True)
class Operator:
    name: str
    argument_kinds: tuple[str, ...]


DOMAINS: dict[str, tuple[Operator, ...]] = {
    "kitchen": (
        Operator("PICK", ("object",)),
        Operator("PLACE", ("object", "destination")),
        Operator("POUR", ("object", "object")),
        Operator("STIR", ("object", "object")),
        Operator("PLACE_SERVING_UTENSIL", ("object", "object")),
        Operator("OPEN", ("inspectable_region",)),
    ),
    "living_room": (
        Operator("PICK", ("object",)),
        Operator("PLACE", ("object", "region")),
    ),
    "workshop": (
        Operator("INSPECT", ("inspectable_region",)),
        Operator("PICK", ("object",)),
        Operator("PLACE", ("object", "destination")),
        # The thing being repaired is a fixture, not a movable object: it can
        # never be picked, but it is the only admissible target of an insertion
        # or a fastening.  Grounding these over "object" -- the movable pool --
        # left the frame joint out of every grounded action, so the two
        # operators that can satisfy the goal did not exist, and a sketch that
        # named the joint would have been rejected by `validate_sketch` as
        # not-in-the-grounded-set.  See `relaxed_ground`.
        Operator("INSERT", ("object", "fixture")),
        Operator("FASTEN", ("object", "object", "fixture")),
    ),
}


def relaxed_ground(
    scene: str,
    object_ids: Iterable[str],
    region_ids: Iterable[str],
    inspectable_regions: Iterable[str] = (),
    fixed_object_ids: Iterable[str] = (),
) -> tuple[Action, ...]:
    """Ground reachable operator shapes with optimistic continuous values.

    The paper's relaxed reachability uses placeholders for continuous values.
    This planning-only adaptation omits those placeholders from the model-facing
    action signature and adds them during continuous refinement.

    `object_ids` is the *movable* pool -- what the robot may pick up.
    `fixed_object_ids` are objects that exist and may be referred to but never
    picked, such as the Workshop frame joint.  Passing the joint only as
    "not movable" removed it from every argument position, which made
    `INSERT(screw, joint)` and `FASTEN(driver, screw, joint)` -- the only two
    actions that can satisfy the Workshop goal -- absent from the grounded set
    entirely.  The model was then choosing targets from a menu that did not
    contain the right answer, and `validate_sketch` would have rejected the
    right answer had it been produced anyway.  Empty for scenes that pass no
    movable subset, so their grounded sets are unchanged.
    """
    if scene not in DOMAINS:
        raise ValueError(f"Unsupported OWL-TAMP scene {scene!r}")
    fixed = set(fixed_object_ids)
    pools = {
        "object": tuple(sorted(set(object_ids))),
        "fixture": tuple(sorted(fixed)),
        "region": tuple(sorted(set(region_ids))),
        # A fixture is deliberately NOT a placement destination.  The physical
        # layer routes a placement onto the frame joint to its insertion
        # primitive, but the symbolic transition model does not: PLACE records
        # `at(screw, joint)` where FASTEN's precondition needs
        # `inserted(screw, joint)`.  Grounding both spellings therefore gave
        # the domain two operators that mean the same thing physically and
        # different things symbolically, and the search would satisfy a sketch
        # by placing onto the joint and then never be able to apply the FASTEN
        # that sketch asked for -- observed live, repeating
        # `PLACE(driver, joint)` for three cycles.  INSERT is the one way to
        # seat a fastener.
        "destination": tuple(sorted(set(object_ids) | set(region_ids))),
        "inspectable_region": tuple(sorted(set(inspectable_regions))),
    }
    result = []
    for operator in DOMAINS[scene]:
        for arguments in product(*(pools[kind] for kind in operator.argument_kinds)):
            if len(arguments) == 2 and arguments[0] == arguments[1]:
                continue
            result.append(Action(operator.name, tuple(arguments)))
    return tuple(result)


def validate_sketch(
    scene: str,
    actions: Iterable[Action],
    grounded_actions: Iterable[Action],
) -> tuple[Action, ...]:
    allowed = set(grounded_actions)
    arities = {operator.name: len(operator.argument_kinds) for operator in DOMAINS[scene]}
    result = []
    for action in actions:
        if action.operator not in arities:
            raise ValidationError(f"unknown {scene} operator {action.operator!r}")
        if len(action.arguments) != arities[action.operator]:
            raise ValidationError(f"{action.operator} has the wrong arity")
        if action not in allowed:
            raise ValidationError(
                f"action is not in the relaxed grounded set: {action.as_dict()}"
            )
        result.append(action)
    return tuple(result)


def executed_encoding(actions: Iterable[Action]) -> tuple[dict[str, object], ...]:
    """Return the paper's Executed(i) subsequence compilation."""
    rows = []
    for index, action in enumerate(actions):
        rows.append(
            {
                "action": action.as_dict(),
                "extra_precondition": "Executed(0)" if index == 0 else f"Executed({index})",
                "extra_effect": f"Executed({index + 1})",
            }
        )
    return tuple(rows)
