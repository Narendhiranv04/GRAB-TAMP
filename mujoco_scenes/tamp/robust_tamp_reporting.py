"""Shared outcome reporting for the ROBUST-TAMP scene runners.

ROBUST-TAMP wrote only `discovery_replanning_result.json`, its own schema,
which meant it carried no feasible/infeasible verdict into the artifact the
metrics read.  `success` is physical goal satisfaction and is false by
construction on a GT-infeasible variant, so without the verdict a method that
correctly rejects an impossible task is scored identically to one that blunders
through it -- 6 of Kitchen's 12 variants, 4 of Living Room's 10, and 2 of
Workshop's 10.  That is the same defect that made every infeasible episode
score as a plain failure for VLM-TAMP and OWL-TAMP until it was fixed.

One helper for all three scenes so the verdict rule cannot drift between them.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from baseline_common.physical_benchmark import (
    GOAL_COMPLETE_STATUS,
    write_execution_result,
)

METHOD = "discovery_replanning"

_GT_ROOT = Path(__file__).resolve().parents[2] / "EXPECTED_GT_ACTIONS"


def expected_outcome_for(scene: str, variant: str) -> str:
    """The frozen GT feasibility verdict for one variant.

    Read from the same `EXPECTED_GT_ACTIONS` tree the other baselines compare
    against, so the three methods cannot disagree about what a variant is.
    This is evaluator-private and never reaches a model.
    """
    path = _GT_ROOT / scene / variant / "expected_gt_actions.json"
    if not path.is_file():
        raise ValueError(
            f"No frozen GT feasibility verdict for {scene}/{variant}: {path}"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    outcome = str(document.get("intended_outcome", "")).upper()
    if outcome not in {"FEASIBLE", "INFEASIBLE"}:
        raise ValueError(f"{path} has no usable intended_outcome: {outcome!r}")
    return outcome


def predicted_outcome(
    *,
    success: bool,
    terminal_failure: Mapping[str, Any] | None,
    infeasibility_proven: bool | None = None,
) -> str:
    """The method's own three-way verdict on the task.

    FEASIBLE when the goal was physically satisfied.  INFEASIBLE only when the
    method actually concluded the task cannot be done -- for ROBUST-TAMP that
    is the planner returning `NO_VALID_PLAN`, the direct counterpart of
    VLM-TAMP's `no_valid_subgoals`.  Everything else is UNRESOLVED, including a
    plain budget overrun: exhausting the replanning budget is not a rejection
    and must not earn the credit for one.

    `infeasibility_proven` is the stricter scene-level test where a runtime
    offers it (Workshop), which additionally requires every storage region to
    have been inspected.  When it is available it is the authority, because a
    claim of infeasibility that skipped a region has not been earned.
    """
    if success:
        return "FEASIBLE"
    if infeasibility_proven is not None:
        return "INFEASIBLE" if infeasibility_proven else "UNRESOLVED"
    if terminal_failure and bool(terminal_failure.get("no_valid_plan")):
        return "INFEASIBLE"
    return "UNRESOLVED"


def write_robust_tamp_artifacts(
    output_dir,
    *,
    scene: str,
    protocol: str,
    variant: str,
    camera_count: int,
    seed: int,
    success: bool,
    executed_actions: int,
    model_calls: int,
    raw_vlm_requests: int,
    replans: int,
    planning_latency_s: float,
    elapsed_seconds: float,
    status: str,
    terminal_failure: Mapping[str, Any] | None,
    expected_outcome: str | None,
    infeasibility_proven: bool | None = None,
) -> dict[str, Any]:
    """Emit the shared artifact with the verdict, and return the verdict fields.

    `terminal_status` must be GOAL_COMPLETE exactly when success is true --
    `write_execution_result` enforces that so failure-mode breakdowns stay
    comparable across methods.
    """
    predicted = predicted_outcome(
        success=success,
        terminal_failure=terminal_failure,
        infeasibility_proven=infeasibility_proven,
    )
    write_execution_result(
        output_dir,
        scene=scene,
        method=METHOD,
        protocol=protocol,
        variant=variant,
        camera_count=camera_count,
        seed=seed,
        success=bool(success),
        executed_actions=executed_actions,
        model_calls=model_calls,
        # Counted from the planner, not inferred from model_calls.  Those
        # coincided until transport faults gained their own retry budget: a
        # failed request is real HTTP traffic that produces no completion, so
        # `model_calls` now undercounts what was actually sent.  VLM-TAMP and
        # OWL-TAMP both count from their transports for the same reason.
        raw_vlm_requests=raw_vlm_requests,
        replans=replans,
        planning_latency_s=planning_latency_s,
        elapsed_seconds=elapsed_seconds,
        terminal_status=GOAL_COMPLETE_STATUS if success else str(status).upper(),
        terminal_failure=terminal_failure,
        expected_outcome=expected_outcome,
        predicted_outcome=predicted,
    )
    return {
        "expected_outcome": expected_outcome,
        "predicted_outcome": predicted,
        "outcome_match": (
            None if expected_outcome is None else predicted == expected_outcome
        ),
    }
