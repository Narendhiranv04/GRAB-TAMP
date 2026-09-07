"""A transport fault must not be charged to ROBUST-TAMP's planning budget.

`ModelTransportError` subclasses `PlanningError`, and
`discovery_planner.plan()` caught `PlanningError` wholesale and re-raised it as
`RecoverablePlanningError` -- which the executive treats as the model having
emitted an unusable plan, consuming a replan.  `_model_calls` was also
incremented *before* the request, so a failed call cost a model call too.

Measured consequence: when the inference server went away mid-run, an episode
burned all 5 model calls and 4 replans in 26 seconds and was recorded as
FAILED with 0 executed actions -- an infrastructure outage reported as the
method planning badly.

VLM-TAMP's executive already draws transport faults on a separate bounded
budget, commented "A transport failure produced no completion, so it is an
infrastructure fault rather than a model call", and BASELINE_FIDELITY.md
describes truncation as retried "in the same way a transport fault already
did".
"""

from __future__ import annotations

import pytest

from baseline_common.inference import ModelTransportError, PlanningError
from mujoco_scenes.tamp.discovery_replanning import (
    RecoverablePlanningError,
    TransportFaultError,
)


def test_a_the_two_failure_kinds_are_distinct_types():
    """Folding them together is what caused the mis-accounting."""
    assert not issubclass(TransportFaultError, RecoverablePlanningError)
    assert not issubclass(RecoverablePlanningError, TransportFaultError)


def test_b_model_transport_error_is_still_a_planning_error_upstream():
    """The subclassing is why order matters in the planner's except clauses.

    If this ever stops being true the ordering guard below is moot, so it is
    asserted rather than assumed.
    """
    assert issubclass(ModelTransportError, PlanningError)


def test_c_the_planner_catches_the_transport_case_first():
    """`except PlanningError` before `except ModelTransportError` would shadow it."""
    import inspect

    from mujoco_scenes.tamp.discovery_planner import OpenAIDiscoveryPlanner

    source = inspect.getsource(OpenAIDiscoveryPlanner.plan)
    transport_at = source.index("except ModelTransportError")
    planning_at = source.index("except PlanningError")
    assert transport_at < planning_at, (
        "ModelTransportError is a PlanningError subclass, so it must be "
        "caught first or it is silently treated as bad model output"
    )
    assert "raise TransportFaultError" in source


def test_d_a_model_call_is_charged_only_for_a_real_completion():
    """The increment must sit after `planner.plan()` returned, not before."""
    import inspect

    from mujoco_scenes.tamp.discovery_replanning import (
        DiscoveryReplanningExecutive,
    )

    source = inspect.getsource(DiscoveryReplanningExecutive._request_plan)
    call_at = source.index("result = self.planner.plan(request)")
    charge_at = source.index("self._model_calls += 1")
    assert charge_at > call_at, (
        "charging before the request means a failed request costs a model call"
    )


def test_e_the_transport_retry_bypasses_the_replan_budget():
    """It must re-request directly, not go through `_begin_replan`.

    `_begin_replan` spends a replan and checks the model-call budget; a
    transport fault is entitled to neither.
    """
    import inspect

    from mujoco_scenes.tamp.discovery_replanning import (
        DiscoveryReplanningExecutive,
    )

    source = inspect.getsource(DiscoveryReplanningExecutive._request_plan)
    block = source[source.index("except TransportFaultError") :]
    block = block[: block.index("except RecoverablePlanningError")]
    assert "self._request_plan(event)" in block, "must retry the request itself"
    assert "_begin_replan" not in block, "must not spend a replan"
    assert "self._transport_retries += 1" in block, "must draw on its own budget"


def test_f_the_retry_budget_is_bounded_and_validated():
    """Unbounded retries would loop forever against a dead server."""
    from mujoco_scenes.tamp.discovery_replanning import (
        DiscoveryReplanningExecutive,
    )
    import inspect

    signature = inspect.signature(DiscoveryReplanningExecutive.__init__)
    default = signature.parameters["max_transport_retries"].default
    assert default == 2, "mirrors VLM-TAMP's max_transport_retries"
    source = inspect.getsource(DiscoveryReplanningExecutive.__init__)
    assert "max_transport_retries must be a non-negative integer" in source


def test_g_exhausting_the_budget_is_reported_as_a_transport_fault():
    """It must be distinguishable from a planning failure in the artifact.

    Otherwise a dead server shows up in the method's planning-failure count.
    """
    import inspect

    from mujoco_scenes.tamp.discovery_replanning import (
        DiscoveryReplanningExecutive,
    )

    source = inspect.getsource(DiscoveryReplanningExecutive._request_plan)
    assert '"transport_fault": True' in source


def test_h_a_transport_fault_never_earns_an_infeasible_verdict():
    """An unreachable server is not evidence the task is impossible."""
    from mujoco_scenes.tamp.robust_tamp_reporting import predicted_outcome

    verdict = predicted_outcome(
        success=False,
        terminal_failure={"code": "inference_failed", "transport_fault": True},
    )
    assert verdict == "UNRESOLVED"
