"""Shadow: let one FM region role realize a different canonical region per operation.

Auxiliary study. The pipeline is not modified; this patches one function in
process and restores it on exit.

The problem it probes
---------------------
Canonicalization resolves each FM role to exactly one canonical role, globally.
Living Room needs two region roles that differ by purpose rather than by kind --
PERSONAL_CUP_SAUCER_REGION, where one person's drinkware goes, and
SHARED_REMOTE_REGION, the surface reachable from both seats -- and the model
routinely writes a single role covering both, whose own stated function says so:

    placement_surface  REGION  count=2
      "Supports the refreshment items and the entertainment control."

That role resolves to the personal form. SUPPORT_ENTERTAINMENT_CONTROL then finds
no capability whose target signature accepts it, the operation is dropped as
NO_CAPABILITY_SIGNATURE_ACCEPTS_THESE_PARTICIPANTS, and the remote is left with
nowhere to be put -- 10 of 60 feasible Living Room trials, all of which place the
drinkware correctly and simply stop two actions short.

What this changes
-----------------
Only the endpoint filter in interpret_operation, and only for a region standing
in for a sibling region. The operation's own capability says which form it needs,
so the substitution is read off the signature rather than guessed.

Why a region and not an object. Substituting a sibling of the same family settles
which of several receiving places an operation meant; it is not a way to decide
what the operation acts on. The registry already carries that scar: allowing the
substitution for objects read a drinkware set as the remote control, on the
strength of both being things one carries. A region is only ever a destination,
so that objection does not reach it.

Four conditions, all required, so this cannot quietly become a general loosening:

  * the operation's text already nominates the capability -- the endpoint is a
    filter, never the source of operation semantics, and that rule is unchanged;
  * substituting changes the target only, never source or anchor;
  * the role named and the role substituted are both REGION kind, in the same
    canonical family;
  * exactly one capability survives, so an ambiguous reading still fails closed.

Status: DOES NOT WORK YET -- do not run this as an arm
-------------------------------------------------------
`_substitute_target` returns None on every Living Room trial it was written
for, so the shadow reproduces the baseline exactly and would read as a clean
null result.

The cause is that it inspects the wrong slot. For the operation this is meant
to rescue, the capability signature is

    SUPPORT_ENTERTAINMENT_CONTROL
      source = (SHARED_REMOTE_REGION, REMOTE_SUPPORT)
      target = (REMOTE, REMOTE_CONTROL)

The region is the **source** and the remote is the target. The first guard
here -- `if not target_role or not _is_region(domain, target_role)` -- sees a
non-region target and bails before anything else runs. Making it fire means
substituting on the source slot, which also invalidates the "changes the
target only, never source or anchor" condition in the list above: that
condition was written from the assumption that a region is always a
destination, and this signature is the counterexample. The safety argument
needs rewriting, not just the slot index.

Committed unrun, because the diagnosis is worth more than the file: it is the
third time in this work that a plausible patch point turned out not to be the
one that decides. Verify the patch point on one trial before running 309.
"""
from __future__ import annotations

import contextlib
import functools
from typing import Any

STATS: dict[str, int] = {}


def _bump(key: str) -> None:
    STATS[key] = STATS.get(key, 0) + 1


def _is_region(domain: str, role: str) -> bool:
    from mujoco_scenes.functional_tamp_pipeline.operation_slot_completion import _slot_role_kind
    return _slot_role_kind(domain, role) == "REGION"


def _same_family(domain: str, a: str, b: str) -> bool:
    from mujoco_scenes.functional_tamp_pipeline.semantic_typing import canonical_role_family
    fa, fb = canonical_role_family(domain, a), canonical_role_family(domain, b)
    return fa == fb and fa != "OTHER"


def _substitute_target(domain: str, raw_phrase: str, source_role: str,
                       target_role: str, anchor_role: str | None) -> str | None:
    """The region this operation's own capability needs in place of the one named."""
    from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
        extract_operation_semantic_candidates, get_robot_capabilities,
    )
    if not target_role or not _is_region(domain, target_role):
        return None
    # The phrase must nominate the capability on its own; endpoints stay a filter.
    nominated = {c.capability_id for c in extract_operation_semantic_candidates(domain, raw_phrase)}
    if not nominated:
        return None
    survivors = set()
    for cap in get_robot_capabilities(domain):
        if cap.capability_id not in nominated:
            continue
        if source_role not in cap.allowed_source_roles:
            continue
        if target_role in cap.allowed_target_roles:
            return None          # already accepted; nothing to substitute
        if anchor_role is not None and cap.allowed_anchor_roles and anchor_role not in cap.allowed_anchor_roles:
            continue
        for candidate in cap.allowed_target_roles:
            if _is_region(domain, candidate) and _same_family(domain, candidate, target_role):
                survivors.add(candidate)
    return next(iter(survivors)) if len(survivors) == 1 else None


@contextlib.contextmanager
def operation_relative_regions():
    """Install the operation-relative region substitution for one evaluation pass."""
    from mujoco_scenes.functional_tamp_pipeline import robot_capability_registry as RC

    STATS.clear()
    original = RC.interpret_operation

    @functools.wraps(original)
    def patched(domain, raw_phrase, source_role, target_role,
                anchor_role=None, capability_hint=None):
        result = original(domain, raw_phrase, source_role, target_role,
                          anchor_role=anchor_role, capability_hint=capability_hint)
        if getattr(result, "status", "") != "UNMAPPABLE_OPERATION":
            _bump("unchanged")
            return result
        try:
            swap = _substitute_target(domain, raw_phrase, source_role, target_role, anchor_role)
        except Exception:
            swap = None
        if swap is None:
            _bump("no_substitution")
            return result
        retried = original(domain, raw_phrase, source_role, swap,
                           anchor_role=anchor_role, capability_hint=capability_hint)
        if getattr(retried, "status", "") == "UNMAPPABLE_OPERATION":
            # The substitution did not rescue it; keep the original refusal so the
            # reported reason stays the true one.
            _bump("substitution_did_not_help")
            return result
        _bump("substituted")
        return retried

    RC.interpret_operation = patched
    restores = [(RC, "interpret_operation", original)]
    # Rebind wherever the name was imported at module load, or the old function
    # keeps being called and the arm silently reproduces the baseline.
    for module_name in (
        "mujoco_scenes.functional_tamp_pipeline.semantic_compiler",
        "mujoco_scenes.functional_tamp_pipeline.operation_slot_completion",
        "mujoco_scenes.functional_tamp_pipeline.functional_constraint_interpreter",
    ):
        try:
            module = __import__(module_name, fromlist=["_"])
        except Exception:
            continue
        if getattr(module, "interpret_operation", None) is original:
            module.interpret_operation = patched
            restores.append((module, "interpret_operation", original))
    try:
        yield STATS
    finally:
        for module, attr, fn in restores:
            setattr(module, attr, fn)
